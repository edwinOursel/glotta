import torch
from transformers import LogitsProcessor
from typing import Set, Optional


class VocabularyConstraintLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor qui contraint la génération aux tokens connus par l'utilisateur.

    Permet trois modes de contrainte :
    - HARD: bloque complètement les tokens inconnus (-inf)
    - SOFT: réduit la probabilité des tokens inconnus (penalité)
    - ADAPTIVE: ajuste dynamiquement en fonction du contexte
    """

    MODE_HARD = "hard"
    MODE_SOFT = "soft"
    MODE_ADAPTIVE = "adaptive"

    def __init__(
        self,
        allowed_token_ids: Set[int],
        mode: str = MODE_HARD,
        penalty_weight: float = 10.0,
        min_allowed_tokens: int = 5
    ):
        """
        Args:
            allowed_token_ids: Set des IDs de tokens autorisés
            mode: Mode de contrainte (hard/soft/adaptive)
            penalty_weight: Poids de la pénalité en mode soft (plus élevé = plus restrictif)
            min_allowed_tokens: Nombre minimum de tokens à garder disponibles en mode adaptive
        """
        self.allowed_token_ids = allowed_token_ids
        self.mode = mode
        self.penalty_weight = penalty_weight
        self.min_allowed_tokens = min_allowed_tokens
        # Lazy-initialized boolean mask tensor; rebuilt if vocab_size changes
        self._forbidden_mask: Optional[torch.BoolTensor] = None
        self._cached_vocab_size: int = -1

    def _get_forbidden_mask(self, vocab_size: int, device: torch.device) -> torch.BoolTensor:
        """
        Retourne un masque booléen 1-D [vocab_size] : True = token interdit.

        Le masque est construit une seule fois et réutilisé pour toute la génération,
        ce qui évite des milliers d'itérations Python par step.
        """
        if self._forbidden_mask is None or self._cached_vocab_size != vocab_size:
            allowed_list = [t for t in self.allowed_token_ids if t < vocab_size]
            allowed_tensor = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            if allowed_list:
                allowed_tensor[torch.tensor(allowed_list, dtype=torch.long, device=device)] = True
            self._forbidden_mask = ~allowed_tensor
            self._cached_vocab_size = vocab_size
        return self._forbidden_mask.to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Modifie les scores (logits) pour contraindre la génération.

        Args:
            input_ids: Tokens d'entrée [batch_size, sequence_length]
            scores: Logits de sortie [batch_size, vocab_size]

        Returns:
            Scores modifiés [batch_size, vocab_size]
        """
        if self.mode == self.MODE_HARD:
            return self._apply_hard_constraint(scores)
        elif self.mode == self.MODE_SOFT:
            return self._apply_soft_constraint(scores)
        elif self.mode == self.MODE_ADAPTIVE:
            return self._apply_adaptive_constraint(scores)
        else:
            raise ValueError(f"Mode inconnu: {self.mode}")

    def _apply_hard_constraint(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Bloque complètement les tokens non autorisés."""
        forbidden = self._get_forbidden_mask(scores.shape[-1], scores.device)
        scores = scores.clone()
        scores[:, forbidden] = float('-inf')
        return scores

    def _apply_soft_constraint(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Pénalise les tokens non autorisés sans les bloquer complètement."""
        forbidden = self._get_forbidden_mask(scores.shape[-1], scores.device)
        scores = scores.clone()
        scores[:, forbidden] -= self.penalty_weight
        return scores

    def _apply_adaptive_constraint(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mode adaptatif : hard constraint, mais si trop peu de tokens disponibles,
        on garde les N meilleurs tokens même s'ils ne sont pas dans le vocabulaire connu.
        """
        vocab_size = scores.shape[-1]
        forbidden = self._get_forbidden_mask(vocab_size, scores.device)
        allowed_mask = ~forbidden

        # Compter combien de tokens autorisés ont une probabilité raisonnable
        num_viable_allowed = (allowed_mask & (scores > float('-inf'))).sum(dim=-1)

        if num_viable_allowed.min() >= self.min_allowed_tokens:
            scores = scores.clone()
            scores[:, forbidden] = float('-inf')
            return scores

        # Fallback : étendre l'autorisation aux top-k tokens pour éviter le blocage
        top_k_indices = torch.topk(scores, k=self.min_allowed_tokens, dim=-1).indices
        # Construire un masque étendu sans muter self.allowed_token_ids (thread-safe)
        extended_allowed = allowed_mask.clone()
        extended_allowed[top_k_indices.reshape(-1)] = True

        scores = scores.clone()
        scores[:, ~extended_allowed] = float('-inf')
        return scores


class GrammarGuidedLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor qui peut favoriser certaines catégories grammaticales
    en fonction du contexte (par exemple, favoriser les verbes après は).

    Plus avancé : nécessite une analyse grammaticale du contexte.
    """

    def __init__(self, tokenizer, grammar_rules: Optional[dict] = None):
        """
        Args:
            tokenizer: Tokenizer pour décoder les tokens
            grammar_rules: Règles grammaticales {pattern: boost_tokens}
        """
        self.tokenizer = tokenizer
        self.grammar_rules = grammar_rules or {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Analyse le contexte et booste certains tokens selon les règles grammaticales.

        Exemple : si le dernier token est は (particle de sujet),
        on peut booster les verbes et adjectifs.
        """
        # Pour l'instant, retourne les scores inchangés
        # À implémenter selon les besoins spécifiques
        return scores
