import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from typing import Optional, List
from logits_processor import VocabularyConstraintLogitsProcessor
from user_vocabulary import UserVocabulary


class JapaneseGenerator:
    """
    Générateur de texte japonais avec contraintes de vocabulaire.

    Utilise un modèle de langage pré-entraîné et contraint la génération
    aux mots/expressions connus par l'utilisateur via manipulation des logits.
    """

    # Modèles japonais disponibles sur Hugging Face
    AVAILABLE_MODELS = {
        "gpt2-small": "rinna/japanese-gpt2-small",
        "gpt2-medium": "rinna/japanese-gpt2-medium",
        "gpt2-large": "rinna/japanese-gpt2-1b",
        "gpt-neox": "rinna/japanese-gpt-neox-small",
    }

    def __init__(
        self,
        model_name: str = "gpt2-small",
        user_vocabulary: Optional[UserVocabulary] = None,
        constraint_mode: str = "hard",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: Nom du modèle (voir AVAILABLE_MODELS)
            user_vocabulary: Instance de UserVocabulary
            constraint_mode: Mode de contrainte (hard/soft/adaptive)
            device: Device PyTorch (cuda/cpu, auto-détecté si None)
        """
        # Résoudre le nom du modèle
        if model_name in self.AVAILABLE_MODELS:
            model_path = self.AVAILABLE_MODELS[model_name]
        else:
            model_path = model_name  # Assume it's a full path/name

        print(f"📦 Loading model: {model_path}")

        # Charger le modèle et le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Déterminer le device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")

        # Gérer le vocabulaire utilisateur
        if user_vocabulary is None:
            self.user_vocabulary = UserVocabulary(self.tokenizer)
            print("⚠️  No user vocabulary provided, using basic elements only")
        else:
            self.user_vocabulary = user_vocabulary

        self.constraint_mode = constraint_mode

        # Configuration du tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        use_constraints: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Génère du texte japonais à partir d'un prompt.

        Args:
            prompt: Texte de départ
            max_length: Longueur maximale de la génération
            temperature: Température de sampling (plus haut = plus aléatoire)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            num_return_sequences: Nombre de séquences à générer
            use_constraints: Activer/désactiver les contraintes de vocabulaire
            **kwargs: Arguments additionnels pour model.generate()

        Returns:
            Liste de textes générés
        """
        # Encoder le prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Préparer le logits processor si contraintes activées
        logits_processor = None
        if use_constraints:
            allowed_token_ids = self.user_vocabulary.get_allowed_token_ids()
            vocab_processor = VocabularyConstraintLogitsProcessor(
                allowed_token_ids=allowed_token_ids,
                mode=self.constraint_mode
            )
            logits_processor = LogitsProcessorList([vocab_processor])

        # Générer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                **kwargs
            )

        # Décoder les résultats
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts

    def generate_conversational(
        self,
        user_input: str,
        conversation_history: Optional[List[str]] = None,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """
        Génère une réponse conversationnelle.

        Args:
            user_input: Message de l'utilisateur
            conversation_history: Historique de la conversation
            max_length: Longueur maximale
            **kwargs: Arguments pour generate()

        Returns:
            Réponse générée
        """
        # Construire le contexte
        if conversation_history:
            context = "\n".join(conversation_history) + "\n" + user_input
        else:
            context = user_input

        # Générer
        responses = self.generate(context, max_length=max_length, **kwargs)
        return responses[0]

    def interactive_mode(self):
        """
        Mode interactif pour tester la génération.

        L'utilisateur peut entrer des prompts et voir les résultats générés.
        """
        print("\n" + "=" * 60)
        print("🗣️  GLOTTA - Mode interactif")
        print("=" * 60)
        print(f"Vocabulaire: {len(self.user_vocabulary)} éléments")
        print(f"Mode contrainte: {self.constraint_mode}")
        print("Commandes: 'quit' pour quitter, 'stats' pour voir les stats")
        print("=" * 60 + "\n")

        conversation_history = []

        while True:
            try:
                user_input = input("Vous: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("👋 Sayonara!")
                    break

                if user_input.lower() == 'stats':
                    stats = self.user_vocabulary.get_stats()
                    print(f"\n📊 Stats: {stats}\n")
                    continue

                # Générer sans contraintes
                print("\n🔓 Sans contraintes:")
                unconstrained = self.generate(
                    user_input,
                    max_length=50,
                    use_constraints=False,
                    num_return_sequences=1
                )[0]
                print(f"   {unconstrained}")

                # Générer avec contraintes
                print(f"\n🔒 Avec contraintes ({self.constraint_mode}):")
                constrained = self.generate(
                    user_input,
                    max_length=50,
                    use_constraints=True,
                    num_return_sequences=1
                )[0]
                print(f"   {constrained}\n")

                conversation_history.append(f"User: {user_input}")
                conversation_history.append(f"Bot: {constrained}")

            except KeyboardInterrupt:
                print("\n👋 Sayonara!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")

    def set_constraint_mode(self, mode: str):
        """Change le mode de contrainte (hard/soft/adaptive)."""
        if mode not in ["hard", "soft", "adaptive"]:
            raise ValueError(f"Mode invalide: {mode}")
        self.constraint_mode = mode
        print(f"✓ Constraint mode set to: {mode}")

    def get_model_info(self) -> dict:
        """Retourne des informations sur le modèle."""
        return {
            "model_name": self.model.config._name_or_path,
            "vocab_size": self.model.config.vocab_size,
            "device": self.device,
            "user_vocab_size": len(self.user_vocabulary),
            "constraint_mode": self.constraint_mode
        }
