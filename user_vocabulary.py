import json
from typing import Set, List, Dict, Optional
from pathlib import Path


class UserVocabulary:
    """
    Gère le vocabulaire connu par l'utilisateur et le mappe vers les tokens du modèle.

    Supporte :
    - Mots individuels (ex: 猫, 食べる)
    - Expressions (ex: こんにちは)
    - Catégories grammaticales (ex: particules de base)
    - Niveaux de difficulté (JLPT N5-N1)
    """

    # Particules de base qu'on veut probablement toujours autoriser
    BASIC_PARTICLES = ["は", "が", "を", "に", "へ", "と", "で", "の", "も", "や", "か"]

    # Ponctuations et caractères spéciaux toujours autorisés
    BASIC_PUNCTUATION = ["。", "、", "！", "？", ".", ",", "!", "?", " ", "\n"]

    def __init__(self, tokenizer, vocab_file: Optional[str] = None):
        """
        Args:
            tokenizer: Tokenizer du modèle (pour mapper mots → token IDs)
            vocab_file: Chemin vers un fichier JSON contenant le vocabulaire de l'utilisateur
        """
        self.tokenizer = tokenizer
        self.known_words: Set[str] = set()
        self.known_expressions: Set[str] = set()
        self.vocab_file = vocab_file

        # Cache pour get_allowed_token_ids() — invalidé à chaque modification du vocabulaire
        self._token_ids_cache: Optional[Set[int]] = None

        # Ajouter les éléments de base
        self._add_basic_elements()

        # Charger le vocabulaire depuis le fichier si fourni
        if vocab_file and Path(vocab_file).exists():
            self.load_from_file(vocab_file)

    def _add_basic_elements(self):
        """Ajoute les particules et ponctuation de base."""
        for particle in self.BASIC_PARTICLES:
            self.known_words.add(particle)
        for punct in self.BASIC_PUNCTUATION:
            self.known_words.add(punct)
        self._token_ids_cache = None

    def add_word(self, word: str):
        """Ajoute un mot au vocabulaire connu."""
        self.known_words.add(word)
        self._token_ids_cache = None

    def add_words(self, words: List[str]):
        """Ajoute plusieurs mots au vocabulaire connu."""
        self.known_words.update(words)
        self._token_ids_cache = None

    def add_expression(self, expression: str):
        """Ajoute une expression (plusieurs mots) au vocabulaire connu."""
        self.known_expressions.add(expression)
        self._token_ids_cache = None

    def remove_word(self, word: str):
        """Retire un mot du vocabulaire connu."""
        self.known_words.discard(word)
        self._token_ids_cache = None

    def add_jlpt_level(self, level: str):
        """
        Ajoute tous les mots d'un niveau JLPT.

        Args:
            level: N5, N4, N3, N2, ou N1

        Note: Cette méthode est un placeholder. Pour une vraie implémentation,
        il faudrait charger des listes de vocabulaire JLPT depuis des ressources externes.
        """
        # TODO: Intégrer avec une base de données JLPT
        print(f"⚠️  JLPT {level} vocabulary loading not yet implemented")
        print(f"   You can manually add words with add_words() or load from JSON")

    def get_allowed_token_ids(self) -> Set[int]:
        """
        Convertit le vocabulaire connu en IDs de tokens utilisables par le modèle.

        Le résultat est mis en cache et n'est recalculé qu'après une modification
        du vocabulaire (add_word, remove_word, load_from_file, etc.).

        Returns:
            Set des IDs de tokens autorisés
        """
        if self._token_ids_cache is not None:
            return self._token_ids_cache

        allowed_ids: Set[int] = set()

        # Tokeniser chaque mot connu et récupérer ses token IDs
        for word in self.known_words:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            allowed_ids.update(token_ids)

        # Tokeniser chaque expression connue
        for expression in self.known_expressions:
            token_ids = self.tokenizer.encode(expression, add_special_tokens=False)
            allowed_ids.update(token_ids)

        # Toujours autoriser les tokens spéciaux (BOS, EOS, PAD, etc.)
        allowed_ids.update(self.tokenizer.all_special_ids)

        self._token_ids_cache = allowed_ids
        return allowed_ids

    def save_to_file(self, filepath: Optional[str] = None):
        """
        Sauvegarde le vocabulaire dans un fichier JSON.

        Args:
            filepath: Chemin du fichier (utilise self.vocab_file si non fourni)
        """
        filepath = filepath or self.vocab_file
        if not filepath:
            raise ValueError("No filepath provided and no vocab_file set")

        data = {
            "words": list(self.known_words),
            "expressions": list(self.known_expressions)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Vocabulary saved to {filepath}")

    def load_from_file(self, filepath: Optional[str] = None):
        """
        Charge le vocabulaire depuis un fichier JSON.

        Args:
            filepath: Chemin du fichier (utilise self.vocab_file si non fourni)
        """
        filepath = filepath or self.vocab_file
        if not filepath:
            raise ValueError("No filepath provided and no vocab_file set")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.known_words.update(data.get("words", []))
        self.known_expressions.update(data.get("expressions", []))
        self._token_ids_cache = None

        print(f"✓ Loaded {len(self.known_words)} words and {len(self.known_expressions)} expressions")

    def get_stats(self) -> Dict:
        """Retourne des statistiques sur le vocabulaire."""
        return {
            "total_words": len(self.known_words),
            "total_expressions": len(self.known_expressions),
            "total_token_ids": len(self.get_allowed_token_ids()),
            "sample_words": list(self.known_words)[:10]
        }

    def __len__(self):
        """Retourne le nombre total de mots connus."""
        return len(self.known_words) + len(self.known_expressions)

    def __contains__(self, word: str):
        """Vérifie si un mot est dans le vocabulaire connu."""
        return word in self.known_words or word in self.known_expressions

    def __repr__(self):
        return f"UserVocabulary(words={len(self.known_words)}, expressions={len(self.known_expressions)})"
