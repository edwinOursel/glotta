/// Model representing a learning session
class LearningSession {
  final int? id;
  final DateTime date;
  final int duration; // in seconds
  final int wordsPracticed;
  final String? constraintMode;
  final String? notes;

  const LearningSession({
    this.id,
    required this.date,
    required this.duration,
    this.wordsPracticed = 0,
    this.constraintMode,
    this.notes,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'date': date.toIso8601String(),
      'duration': duration,
      'words_practiced': wordsPracticed,
      'constraint_mode': constraintMode,
      'notes': notes,
    };
  }

  factory LearningSession.fromMap(Map<String, dynamic> map) {
    return LearningSession(
      id: map['id'] as int?,
      date: DateTime.parse(map['date'] as String),
      duration: map['duration'] as int,
      wordsPracticed: map['words_practiced'] as int? ?? 0,
      constraintMode: map['constraint_mode'] as String?,
      notes: map['notes'] as String?,
    );
  }
}
