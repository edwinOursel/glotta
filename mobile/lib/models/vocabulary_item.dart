/// Vocabulary item as returned by the backend API (/api/vocabulary).
/// Contains SM-2 SRS metadata in addition to word data.
class VocabularyItem {
  final String   id;
  final String   word;
  final String?  reading;
  final String?  meaning;
  final String?  jlptLevel;
  final int      masteryLevel;   // 0-5
  final int      repetition;
  final int      interval;       // days
  final double   easeFactor;
  final DateTime nextReviewAt;
  final int      timesSeen;
  final int      timesCorrect;
  final DateTime dateAdded;

  const VocabularyItem({
    required this.id,
    required this.word,
    this.reading,
    this.meaning,
    this.jlptLevel,
    this.masteryLevel = 0,
    this.repetition   = 0,
    this.interval     = 1,
    this.easeFactor   = 2.5,
    required this.nextReviewAt,
    this.timesSeen    = 0,
    this.timesCorrect = 0,
    required this.dateAdded,
  });

  bool get isDueForReview => nextReviewAt.isBefore(DateTime.now());

  double get accuracy =>
      timesSeen == 0 ? 0 : timesCorrect / timesSeen;

  factory VocabularyItem.fromJson(Map<String, dynamic> json) {
    return VocabularyItem(
      id:           json['id'] as String,
      word:         json['word'] as String,
      reading:      json['reading'] as String?,
      meaning:      json['meaning'] as String?,
      jlptLevel:    json['jlpt_level'] as String?,
      masteryLevel: json['mastery_level'] as int? ?? 0,
      repetition:   json['repetition']   as int? ?? 0,
      interval:     json['interval']     as int? ?? 1,
      easeFactor:   (json['ease_factor'] as num?)?.toDouble() ?? 2.5,
      nextReviewAt: DateTime.parse(
          json['next_review_at'] as String? ?? DateTime.now().toIso8601String()),
      timesSeen:    json['times_seen']    as int? ?? 0,
      timesCorrect: json['times_correct'] as int? ?? 0,
      dateAdded:    DateTime.parse(
          json['date_added'] as String? ?? DateTime.now().toIso8601String()),
    );
  }
}
