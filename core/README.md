# 🗣️ Glotta - Apprentissage du japonais avec LLM contraint

Glotta est un système d'apprentissage de langues (focus japonais) qui utilise des modèles de langage (LLM) avec contraintes de vocabulaire. Le concept : **forcer le modèle à générer du texte uniquement avec les mots et structures grammaticales que vous connaissez déjà**.

## 💡 Concept

L'idée est simple mais puissante : au lieu de vous exposer à du contenu trop complexe, Glotta génère du texte japonais **adapté à votre niveau** en manipulant directement les probabilités de sortie du LLM (logits de la dernière couche).

### Comment ça marche ?

```
Utilisateur → Vocabulaire connu → Modèle LLM japonais
                                         ↓
                                  Logits Processor
                                  (contraintes de vocabulaire)
                                         ↓
                                  Génération contrainte
                                  (seulement les mots connus)
```

**Mécanisme technique** :
- Le modèle calcule normalement les probabilités pour tous les tokens
- Avant le sampling, on applique un masque sur les logits :
  - Mode **HARD** : `-inf` sur les tokens inconnus → probabilité = 0
  - Mode **SOFT** : pénalité sur les tokens inconnus → moins probable
  - Mode **ADAPTIVE** : ajuste dynamiquement selon le contexte

## 🚀 Installation

### Avec uv (recommandé)

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv pip install -e .

# Note: Le premier lancement téléchargera le modèle (~500MB pour gpt2-small)
```

### Avec pip classique

```bash
# Installer les dépendances depuis pyproject.toml
pip install -e .
```

Voir [INSTALL.md](./INSTALL.md) pour plus de détails.

### Dépendances principales
- `torch` : Framework de deep learning
- `transformers` : Librairie Hugging Face pour les LLMs
- `sentencepiece` : Tokenisation
- `fugashi` + `ipadic` : Analyse morphologique japonaise (optionnel)
- `fastapi` + `uvicorn` : API REST pour l'app mobile

## 📖 Utilisation rapide

### 1. Demo basique

```bash
uv run python demo.py 1
# ou simplement: python demo.py 1
```

Génère du texte avec un vocabulaire minimal (débutant).

### 2. Créer votre vocabulaire

```python
from japanese_generator import JapaneseGenerator
from user_vocabulary import UserVocabulary

# Initialiser
generator = JapaneseGenerator(model_name="gpt2-small")
vocab = generator.user_vocabulary

# Ajouter vos mots
vocab.add_words(["猫", "犬", "食べる", "好き"])

# Sauvegarder
vocab.save_to_file("my_vocabulary.json")
```

### 3. Générer du texte

```python
# Générer avec contraintes
result = generator.generate(
    prompt="私は猫が",
    max_length=50,
    use_constraints=True
)
print(result[0])
```

### 4. Mode interactif

```bash
python demo.py 4
```

Testez la génération en temps réel avec différentes contraintes.

## 📁 Structure du projet

```
glotta/
├── README.md                           # Ce fichier
├── requirements.txt                    # Dépendances
│
├── japanese_generator.py               # Générateur principal
├── logits_processor.py                 # Manipulation des logits
├── user_vocabulary.py                  # Gestion du vocabulaire utilisateur
│
├── demo.py                             # Scripts de démonstration
│
├── vocabulary_n5.json                  # Vocabulaire JLPT N5 (exemple)
└── vocabulary_intermediate.json        # Vocabulaire intermédiaire (exemple)
```

## 🎯 Fonctionnalités

### ✅ Implémenté

- [x] Manipulation des logits pour contraindre le vocabulaire
- [x] Trois modes de contrainte (hard/soft/adaptive)
- [x] Gestion du vocabulaire utilisateur (save/load JSON)
- [x] Support des modèles japonais pré-entraînés (rinna/japanese-gpt2)
- [x] Mode interactif pour tester
- [x] Génération avec/sans contraintes pour comparaison

### 🚧 À venir

- [ ] Intégration bases de données JLPT (N5-N1)
- [ ] Support des règles grammaticales (GrammarGuidedLogitsProcessor)
- [ ] Analyse morphologique automatique (MeCab)
- [ ] Interface web (Gradio/Streamlit)
- [ ] Système de progression automatique (détecter les mots maîtrisés)
- [ ] Support multi-langues (au-delà du japonais)
- [ ] Fine-tuning sur corpus spécifiques
- [ ] Mode conversation (chatbot adaptatif)

## 🔧 Configuration avancée

### Modes de contrainte

```python
# Mode HARD : bloque complètement les tokens inconnus
generator.set_constraint_mode("hard")

# Mode SOFT : pénalise mais n'interdit pas
generator.set_constraint_mode("soft")

# Mode ADAPTIVE : ajuste selon le contexte
generator.set_constraint_mode("adaptive")
```

### Choix du modèle

```python
# Modèles disponibles
models = {
    "gpt2-small": "rinna/japanese-gpt2-small",      # ~350MB, rapide
    "gpt2-medium": "rinna/japanese-gpt2-medium",    # ~800MB, meilleur
    "gpt2-large": "rinna/japanese-gpt2-1b",         # ~3GB, excellent
    "gpt-neox": "rinna/japanese-gpt-neox-small",    # ~1.4GB, moderne
}

generator = JapaneseGenerator(model_name="gpt2-medium")
```

### Paramètres de génération

```python
result = generator.generate(
    prompt="今日は",
    max_length=100,           # Longueur max
    temperature=0.8,          # Créativité (0.7-1.0 recommandé)
    top_p=0.9,               # Nucleus sampling
    top_k=50,                # Top-k sampling
    use_constraints=True,     # Activer contraintes
    num_return_sequences=3    # Nombre de variantes
)
```

## 🧪 Exemples d'utilisation

### Exemple 1 : Débutant absolu

```python
# Vocabulaire ultra-basique
vocab = UserVocabulary(tokenizer)
vocab.add_words([
    "私", "猫", "好き", "です", "食べる", "ご飯", "美味しい"
])

generator = JapaneseGenerator(user_vocabulary=vocab)
result = generator.generate("私は", max_length=30)
# → "私は猫が好きです。" (seulement des mots connus)
```

### Exemple 2 : Niveau intermédiaire

```python
# Charger vocabulaire N4-N3
vocab = UserVocabulary(tokenizer, vocab_file="vocabulary_n4.json")

# Générer plusieurs variantes
results = generator.generate(
    "週末に友達と",
    num_return_sequences=3,
    temperature=0.9
)
# → Trois phrases différentes avec le même vocabulaire
```

### Exemple 3 : Comparaison

```python
prompt = "今日は"

# Sans contraintes
free = generator.generate(prompt, use_constraints=False)[0]
# → Peut contenir des mots complexes, kanji rares, etc.

# Avec contraintes
constrained = generator.generate(prompt, use_constraints=True)[0]
# → Uniquement votre vocabulaire connu
```

## 🎓 Cas d'usage pédagogiques

### 1. Pratique de lecture graduée
Générez des textes de complexité croissante au fur et à mesure de votre apprentissage.

### 2. Exercices de compréhension
Créez des dialogues adaptés à votre niveau pour pratiquer la lecture.

### 3. Découverte de structures
Voyez comment vos mots connus peuvent se combiner naturellement.

### 4. Renforcement du vocabulaire
Exposez-vous uniquement aux mots que vous étudiez actuellement.

## 🧠 Détails techniques

### Logits Processing

Le cœur du système est le `VocabularyConstraintLogitsProcessor` :

```python
class VocabularyConstraintLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # scores = logits de shape [batch_size, vocab_size]

        # Créer un masque
        mask = torch.zeros_like(scores)

        # Appliquer -inf aux tokens interdits
        for forbidden_id in forbidden_tokens:
            mask[:, forbidden_id] = float('-inf')

        # Modifier les scores
        return scores + mask
```

Les logits modifiés sont ensuite passés à la fonction de sampling (softmax + categorical).

### Tokenisation japonaise

Le japonais pose des défis spécifiques :
- Pas d'espaces entre les mots
- Mélange de hiragana, katakana, kanji
- Particules grammaticales collées aux mots

Solution actuelle : **SentencePiece** (subword tokenization)
- Un mot peut = plusieurs tokens
- Les particules sont souvent des tokens séparés
- Nécessite de mapper "mot connu" → "tous ses tokens"

### Performance

- **Overhead du masquage** : ~5-10ms par génération (négligeable)
- **Chargement du modèle** : ~2-5s (une fois au démarrage)
- **Génération** : ~0.5-2s pour 50 tokens (selon GPU/CPU)

GPU recommandé pour une utilisation intensive, mais CPU tout à fait utilisable.

## 🤝 Contribution

Les contributions sont bienvenues ! Quelques idées :

- **Bases de données JLPT** : Intégrer des listes officielles
- **Analyse grammaticale** : Utiliser MeCab pour des contraintes plus fines
- **Interface utilisateur** : Créer une app web
- **Benchmarks** : Évaluer la qualité des textes générés
- **Support multi-langues** : Adapter pour le coréen, chinois, etc.

## 📚 Ressources

### Modèles utilisés
- [rinna/japanese-gpt2](https://huggingface.co/rinna/japanese-gpt2-medium)
- [Documentation Transformers](https://huggingface.co/docs/transformers)

### Vocabulaire japonais
- [JLPT Resources](https://jlptsensei.com/)
- [Core 10k](https://ankiweb.net/shared/info/2141233552)

### Papers de référence
- [CTRL: Conditional Transformer Language Model](https://arxiv.org/abs/1909.05858)
- [Controlled Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/)

## 📝 License

MIT License - Faites-en ce que vous voulez !

## 🙏 Remerciements

- **rinna Co.** pour les modèles japonais pré-entraînés
- **Hugging Face** pour la librairie Transformers
- Communauté d'apprentissage du japonais

---

**Note** : Ce projet est expérimental et en développement actif. Les modèles actuels ne sont pas parfaits et peuvent générer du contenu bizarre ou grammaticalement incorrect. C'est un outil d'**assistance** à l'apprentissage, pas un remplacement d'un professeur !

Pour toute question : ouvrez une issue sur GitHub.

頑張ってください！(Ganbatte kudasai! - Bon courage !)
