/// Model representing user settings
class UserSettings {
  final String apiBaseUrl;
  final String constraintMode;
  final String jlptLevel;
  final bool useConstraints;

  const UserSettings({
    required this.apiBaseUrl,
    required this.constraintMode,
    required this.jlptLevel,
    this.useConstraints = true,
  });

  factory UserSettings.defaultSettings() {
    return const UserSettings(
      apiBaseUrl: 'http://localhost:8000',
      constraintMode: 'hard',
      jlptLevel: 'N5',
      useConstraints: true,
    );
  }

  UserSettings copyWith({
    String? apiBaseUrl,
    String? constraintMode,
    String? jlptLevel,
    bool? useConstraints,
  }) {
    return UserSettings(
      apiBaseUrl: apiBaseUrl ?? this.apiBaseUrl,
      constraintMode: constraintMode ?? this.constraintMode,
      jlptLevel: jlptLevel ?? this.jlptLevel,
      useConstraints: useConstraints ?? this.useConstraints,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'api_base_url': apiBaseUrl,
      'constraint_mode': constraintMode,
      'jlpt_level': jlptLevel,
      'use_constraints': useConstraints,
    };
  }

  factory UserSettings.fromJson(Map<String, dynamic> json) {
    return UserSettings(
      apiBaseUrl: json['api_base_url'] as String? ?? 'http://localhost:8000',
      constraintMode: json['constraint_mode'] as String? ?? 'hard',
      jlptLevel: json['jlpt_level'] as String? ?? 'N5',
      useConstraints: json['use_constraints'] as bool? ?? true,
    );
  }
}
