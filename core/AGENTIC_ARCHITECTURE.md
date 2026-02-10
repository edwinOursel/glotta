# Architecture Agentique - Glotta

Ce document décrit l'architecture agentique multi-LLM de Glotta utilisant LangGraph.

## Vue d'ensemble

L'architecture agentique remplace le simple pipeline API → LLM par un système orchestré de plusieurs agents spécialisés qui collaborent pour produire des réponses de haute qualité adaptées à l'utilisateur.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. Intent Detection Agent (GPT-4 Mini)                          │
│  - Détecte l'intention: practice, correction, conversation,     │
│    quiz, translation, explanation                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Error Correction Agent (GPT-4 Mini)                          │
│  - Corrige les fautes de japonais si nécessaire                 │
│  - Fournit des explications pédagogiques                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. System Prompt Builder                                        │
│  - Construit dynamiquement le prompt système selon:              │
│    * L'intention détectée                                        │
│    * Le niveau JLPT de l'utilisateur (N5-N1)                     │
│    * Le contexte de la conversation                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. Constrained Generation (Rinna Japanese GPT-2)                │
│  - Génère du texte japonais avec contraintes vocabulaire        │
│  - Utilise VocabularyConstraintLogitsProcessor                   │
│  - Modes: hard, soft, adaptive                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Response Validation Agent (GPT-4 Mini)                       │
│  - Évalue la qualité de la réponse:                              │
│    * Grammaire correcte                                          │
│    * Appropriée au niveau JLPT                                   │
│    * Pertinence par rapport à l'input                            │
│    * Japonais naturel                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  Score < 0.6?   │
                    └─────────────────┘
                       ↓            ↓
                    YES (retry)    NO
                       ↓            ↓
                   Regenerate   Finalize
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. Response Finalizer                                           │
│  - Combine la réponse avec le feedback de correction            │
│  - Ajoute des notes de validation si nécessaire                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Final Response                              │
└─────────────────────────────────────────────────────────────────┘
```

## Composants

### 1. IntentDetector
**LLM**: GPT-4 Mini (OpenAI)
**Rôle**: Analyser l'input utilisateur et déterminer l'intention

**Intentions supportées**:
- `practice`: Utilisateur veut pratiquer l'écriture/conversation
- `correction`: Utilisateur veut des corrections
- `conversation`: Conversation naturelle
- `quiz`: Utilisateur veut être testé sur le vocabulaire
- `translation`: Aide à la traduction
- `explanation`: Explications grammaticales/vocabulaire

### 2. ErrorCorrector
**LLM**: GPT-4 Mini (OpenAI)
**Rôle**: Corriger les erreurs dans l'input japonais de l'utilisateur

**Fonctionnalités**:
- Détecte et corrige les fautes de grammaire
- Fournit des explications pédagogiques
- S'active principalement pour l'intention "correction"

### 3. SystemPromptBuilder
**LLM**: N/A (logique déterministe)
**Rôle**: Construire dynamiquement le prompt système

**Personnalisation**:
- Template spécifique par intention
- Adaptation au niveau JLPT (N5, N4, N3, N2, N1)
- Contexte conversationnel

### 4. ConstrainedGenerator
**LLM**: Rinna Japanese GPT-2 (local)
**Rôle**: Générer du texte japonais avec contraintes vocabulaire

**Caractéristiques**:
- Utilise le `VocabularyConstraintLogitsProcessor`
- Trois modes de contrainte:
  - **hard**: Bloque totalement les tokens inconnus (-inf)
  - **soft**: Pénalise les tokens inconnus (-2.0)
  - **adaptive**: Pénalisation contextuelle (-1.0 à -5.0)
- Génération basée sur le vocabulaire connu de l'utilisateur

### 5. ResponseValidator
**LLM**: GPT-4 Mini (OpenAI)
**Rôle**: Valider la qualité de la réponse générée

**Critères d'évaluation**:
- Grammaire correcte (0-1)
- Appropriée au niveau utilisateur (0-1)
- Pertinence par rapport à l'input (0-1)
- Japonais naturel (0-1)

**Seuils**:
- Score < 0.6 → Régénération (max 3 itérations)
- Score ≥ 0.6 → Acceptation

### 6. ResponseFinalizer
**LLM**: N/A (logique de formatage)
**Rôle**: Préparer la réponse finale avec metadata

## Flux de données (GraphState)

```python
class GraphState(TypedDict):
    # Input
    user_input: str              # Input utilisateur
    user_level: str              # Niveau JLPT (N5-N1)
    use_constraints: bool        # Activer contraintes vocabulaire
    constraint_mode: str         # Mode: hard/soft/adaptive

    # Intermediate
    intent: str                  # Intention détectée
    corrected_input: str         # Input corrigé
    system_prompt: str           # Prompt système dynamique

    # Output
    generated_text: str          # Texte généré
    validation_result: dict      # Résultats de validation
    feedback: str                # Feedback de correction
    final_response: str          # Réponse finale

    # Metadata
    iterations: int              # Nombre de tentatives
    max_iterations: int          # Maximum 3 itérations
```

## API Endpoint

### POST `/api/agentic/generate`

**Request**:
```json
{
  "user_input": "こんにちは、元気ですか？",
  "user_level": "N5",
  "use_constraints": true,
  "constraint_mode": "hard",
  "max_iterations": 3
}
```

**Response**:
```json
{
  "response": "はい、元気です。ありがとうございます。",
  "intent": "conversation",
  "corrected_input": "こんにちは、元気ですか？",
  "validation": {
    "valid": true,
    "score": 0.85,
    "feedback": "Natural and appropriate response"
  },
  "iterations": 1
}
```

## Avantages de l'architecture agentique

1. **Modularité**: Chaque agent a une responsabilité unique et peut être amélioré indépendamment

2. **Adaptabilité**: Le système s'adapte automatiquement à:
   - L'intention de l'utilisateur
   - Son niveau de compétence
   - Le contexte de la conversation

3. **Qualité**: Validation automatique avec régénération si nécessaire

4. **Pédagogie**: Feedback constructif avec corrections et explications

5. **Personnalisation**: Prompts système dynamiques basés sur le profil utilisateur

6. **Évolutivité**: Facile d'ajouter de nouveaux agents (ex: prononciation, culture, etc.)

## Configuration

### Variables d'environnement requises

```bash
# OpenAI API pour les agents (détection, correction, validation)
OPENAI_API_KEY=sk-...

# Optionnel: personnaliser les modèles
OPENAI_MODEL=gpt-4o-mini  # ou gpt-3.5-turbo pour réduire les coûts
```

### Modes de contrainte

- **hard**: Blocage strict → Meilleur pour débutants absolus (N5)
- **soft**: Pénalisation légère → Bon équilibre pour N4-N3
- **adaptive**: Pénalisation contextuelle → Pour niveaux avancés N2-N1

## Évolutions futures

1. **Agents additionnels**:
   - PronunciationAgent: Évaluation de la prononciation
   - CulturalContextAgent: Ajout de contexte culturel
   - ConversationMemoryAgent: Gestion de l'historique long terme

2. **Optimisations**:
   - Cache des validations pour réponses similaires
   - Parallel processing des agents indépendants
   - Fine-tuning du générateur sur corpus JLPT

3. **Analytics**:
   - Tracking des intentions les plus fréquentes
   - Analyse des patterns d'erreurs
   - Recommandations personnalisées de vocabulaire

## Différences avec l'architecture simple

| Aspect | Architecture Simple | Architecture Agentique |
|--------|---------------------|------------------------|
| LLMs | 1 (Rinna GPT-2) | 3+ (GPT-4 Mini + Rinna) |
| Intelligence | Génération basique | Multi-agent orchestré |
| Adaptabilité | Statique | Dynamique (intent-based) |
| Qualité | Pas de validation | Validation + retry |
| Pédagogie | Limité | Corrections + explications |
| Coût | Faible (local) | Moyen (API calls) |
| Latence | ~1-2s | ~3-5s |

## Coûts estimés (OpenAI)

**Par requête** (avec GPT-4 Mini à $0.15/1M input tokens, $0.60/1M output tokens):
- IntentDetector: ~200 tokens → $0.0001
- ErrorCorrector: ~300 tokens → $0.0002
- ResponseValidator: ~400 tokens → $0.0003

**Total par requête**: ~$0.0006 (moins d'un centime)

Avec validation retry (2-3 itérations): ~$0.001-0.002 par requête

## Conclusion

L'architecture agentique transforme Glotta d'un simple générateur de texte contraint en un système pédagogique intelligent qui:
- Comprend l'intention de l'utilisateur
- S'adapte à son niveau
- Fournit du feedback constructif
- Garantit la qualité des réponses

Cette architecture pose les fondations pour un assistant d'apprentissage du japonais véritablement intelligent et personnalisé.
