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
        mask = torch.full_like(scores, 0.0)

        # Créer un masque de tous les tokens NON autorisés
        all_token_ids = set(range(scores.shape[-1]))
        forbidden_token_ids = all_token_ids - self.allowed_token_ids

        # Appliquer -inf aux tokens interdits
        for token_id in forbidden_token_ids:
            mask[:, token_id] = float('-inf')

        return scores + mask

    def _apply_soft_constraint(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Pénalise les tokens non autorisés sans les bloquer complètement."""
        mask = torch.zeros_like(scores)

        all_token_ids = set(range(scores.shape[-1]))
        forbidden_token_ids = all_token_ids - self.allowed_token_ids

        # Appliquer une pénalité aux tokens interdits
        for token_id in forbidden_token_ids:
            mask[:, token_id] = -self.penalty_weight

        return scores + mask

    def _apply_adaptive_constraint(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mode adaptatif : hard constraint, mais si trop peu de tokens disponibles,
        on garde les N meilleurs tokens même s'ils ne sont pas dans le vocabulaire connu.
        """
        # Compter combien de tokens autorisés ont une probabilité raisonnable
        scores_copy = scores.clone()
        allowed_mask = torch.full_like(scores, False, dtype=torch.bool)

        for token_id in self.allowed_token_ids:
            allowed_mask[:, token_id] = True

        # Si on a assez de tokens autorisés, mode hard
        num_viable_allowed = (allowed_mask & (scores > float('-inf'))).sum(dim=-1)

        if num_viable_allowed.min() >= self.min_allowed_tokens:
            return self._apply_hard_constraint(scores)

        # Sinon, on garde les top-k tokens même si inconnus
        # Pour éviter de bloquer complètement la génération
        top_k_values, top_k_indices = torch.topk(scores, k=self.min_allowed_tokens, dim=-1)

        # Créer un masque qui garde soit les tokens autorisés, soit les top-k
        adaptive_allowed = self.allowed_token_ids.copy()
        for batch_idx in range(scores.shape[0]):
            for idx in top_k_indices[batch_idx]:
                adaptive_allowed.add(idx.item())

        # Appliquer le hard constraint avec le vocabulaire étendu
        original_allowed = self.allowed_token_ids
        self.allowed_token_ids = adaptive_allowed
        result = self._apply_hard_constraint(scores)
        self.allowed_token_ids = original_allowed

        return result


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
