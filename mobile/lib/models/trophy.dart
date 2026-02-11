// ── Trophy model & definitions ────────────────────────────────────────────────

class TrophyDefinition {
  final String   id;
  final String   emoji;
  final String   nameKey;        // AppLocalizations key — resolved at display time
  final String   descriptionKey;
  final TrophyTier tier;

  const TrophyDefinition({
    required this.id,
    required this.emoji,
    required this.nameKey,
    required this.descriptionKey,
    required this.tier,
  });
}

enum TrophyTier { bronze, silver, gold, platinum }

// ── All trophy definitions ────────────────────────────────────────────────────

const kTrophies = <TrophyDefinition>[
  // ── First steps ────────────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'first_word',
    emoji:           '🌱',
    nameKey:         'trophyFirstWordName',
    descriptionKey:  'trophyFirstWordDesc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'first_generation',
    emoji:           '✍️',
    nameKey:         'trophyFirstGenerationName',
    descriptionKey:  'trophyFirstGenerationDesc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'first_theme',
    emoji:           '🎓',
    nameKey:         'trophyFirstThemeName',
    descriptionKey:  'trophyFirstThemeDesc',
    tier:            TrophyTier.bronze,
  ),

  // ── Vocabulary count ────────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'vocab_10',
    emoji:           '📖',
    nameKey:         'trophyVocab10Name',
    descriptionKey:  'trophyVocab10Desc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'vocab_50',
    emoji:           '📚',
    nameKey:         'trophyVocab50Name',
    descriptionKey:  'trophyVocab50Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'vocab_100',
    emoji:           '🗂️',
    nameKey:         'trophyVocab100Name',
    descriptionKey:  'trophyVocab100Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'vocab_500',
    emoji:           '🏛️',
    nameKey:         'trophyVocab500Name',
    descriptionKey:  'trophyVocab500Desc',
    tier:            TrophyTier.gold,
  ),

  // ── Vocabulary mastery ──────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'accuracy_80',
    emoji:           '🎯',
    nameKey:         'trophyAccuracy80Name',
    descriptionKey:  'trophyAccuracy80Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'mastered_10',
    emoji:           '⭐',
    nameKey:         'trophyMastered10Name',
    descriptionKey:  'trophyMastered10Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'mastered_50',
    emoji:           '🌟',
    nameKey:         'trophyMastered50Name',
    descriptionKey:  'trophyMastered50Desc',
    tier:            TrophyTier.gold,
  ),

  // ── Generation ─────────────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'generations_10',
    emoji:           '📝',
    nameKey:         'trophyGenerations10Name',
    descriptionKey:  'trophyGenerations10Desc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'generations_50',
    emoji:           '🖊️',
    nameKey:         'trophyGenerations50Name',
    descriptionKey:  'trophyGenerations50Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'generations_200',
    emoji:           '✒️',
    nameKey:         'trophyGenerations200Name',
    descriptionKey:  'trophyGenerations200Desc',
    tier:            TrophyTier.gold,
  ),

  // ── Grammar mastery ─────────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'themes_5',
    emoji:           '🎭',
    nameKey:         'trophyThemes5Name',
    descriptionKey:  'trophyThemes5Desc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'themes_mastered_5',
    emoji:           '💫',
    nameKey:         'trophyThemesMastered5Name',
    descriptionKey:  'trophyThemesMastered5Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'n5_complete',
    emoji:           '🏅',
    nameKey:         'trophyN5CompleteName',
    descriptionKey:  'trophyN5CompleteDesc',
    tier:            TrophyTier.gold,
  ),

  // ── Streaks ─────────────────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'streak_3',
    emoji:           '🔥',
    nameKey:         'trophyStreak3Name',
    descriptionKey:  'trophyStreak3Desc',
    tier:            TrophyTier.bronze,
  ),
  TrophyDefinition(
    id:              'streak_7',
    emoji:           '🔥',
    nameKey:         'trophyStreak7Name',
    descriptionKey:  'trophyStreak7Desc',
    tier:            TrophyTier.silver,
  ),
  TrophyDefinition(
    id:              'streak_30',
    emoji:           '🏆',
    nameKey:         'trophyStreak30Name',
    descriptionKey:  'trophyStreak30Desc',
    tier:            TrophyTier.platinum,
  ),

  // ── Focus word practice ─────────────────────────────────────────────────────
  TrophyDefinition(
    id:              'focus_practice_10',
    emoji:           '🎪',
    nameKey:         'trophyFocusPractice10Name',
    descriptionKey:  'trophyFocusPractice10Desc',
    tier:            TrophyTier.bronze,
  ),
];
