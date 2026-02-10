/// A grammar theme from a classic textbook (e.g. Genki, Minna no Nihongo).
///
/// Used as a "constraint card" in the gamification screen:
/// the user selects a theme → the LLM is prompted to practise that pattern.
class GrammarTheme {
  final String id;
  final String category;
  final String jlptLevel;
  final String nameEn;
  final String nameJp;
  final String description;
  final String exampleJp;
  final String exampleEn;
  final String systemPromptHint;
  final List<String> prerequisiteIds;
  final String icon; // material icon name (used as a lookup key)

  const GrammarTheme({
    required this.id,
    required this.category,
    required this.jlptLevel,
    required this.nameEn,
    required this.nameJp,
    required this.description,
    required this.exampleJp,
    required this.exampleEn,
    required this.systemPromptHint,
    required this.prerequisiteIds,
    required this.icon,
  });

  factory GrammarTheme.fromJson(Map<String, dynamic> json) => GrammarTheme(
        id:               json['id'] as String,
        category:         json['category'] as String,
        jlptLevel:        json['jlpt_level'] as String,
        nameEn:           json['name_en'] as String,
        nameJp:           json['name_jp'] as String,
        description:      json['description'] as String,
        exampleJp:        json['example_jp'] as String,
        exampleEn:        json['example_en'] as String,
        systemPromptHint: json['system_prompt_hint'] as String,
        prerequisiteIds:  List<String>.from(json['prerequisite_ids'] as List),
        icon:             json['icon'] as String,
      );
}

/// Per-user progress on a single grammar theme.
class ThemeProgress {
  final String themeId;
  final int    masteryLevel;    // 0-3 stars
  final int    sessionsCount;
  final bool   isUnlocked;

  const ThemeProgress({
    required this.themeId,
    this.masteryLevel = 0,
    this.sessionsCount = 0,
    this.isUnlocked = false,
  });

  ThemeProgress copyWith({int? masteryLevel, int? sessionsCount, bool? isUnlocked}) =>
      ThemeProgress(
        themeId:       themeId,
        masteryLevel:  masteryLevel  ?? this.masteryLevel,
        sessionsCount: sessionsCount ?? this.sessionsCount,
        isUnlocked:    isUnlocked    ?? this.isUnlocked,
      );
}

/// Categories with display names and colors (as hex strings for JSON-safe storage).
const Map<String, ({String label, int colorValue})> kCategories = {
  'basics':           (label: 'Basics',             colorValue: 0xFF4CAF50),
  'verbs':            (label: 'Verbs',               colorValue: 0xFF2196F3),
  'adjectives':       (label: 'Adjectives',          colorValue: 0xFF00BCD4),
  'te_form':          (label: 'て-Form',              colorValue: 0xFF9C27B0),
  'conditionals':     (label: 'Conditionals',        colorValue: 0xFFFF9800),
  'reason':           (label: 'Reason & Cause',      colorValue: 0xFFFF5722),
  'giving_receiving': (label: 'Giving & Receiving',  colorValue: 0xFFE91E63),
  'voice':            (label: 'Passive & Causative', colorValue: 0xFF795548),
  'modality':         (label: 'Modality',            colorValue: 0xFF607D8B),
  'complex':          (label: 'Complex Patterns',    colorValue: 0xFF3F51B5),
  'keigo':            (label: 'Keigo (Politeness)',  colorValue: 0xFF009688),
};

/// JLPT level badge colors.
const Map<String, int> kLevelColors = {
  'N5': 0xFF4CAF50,
  'N4': 0xFF2196F3,
  'N3': 0xFFFF9800,
  'N2': 0xFFE91E63,
  'N1': 0xFF9C27B0,
};
