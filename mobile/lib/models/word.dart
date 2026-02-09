/// Model representing a Japanese word/expression
class Word {
  final int? id;
  final String word;
  final String? reading;
  final String? meaning;
  final String? partOfSpeech;
  final String? jlptLevel;
  final DateTime dateAdded;
  final int timesSeen;
  final int timesCorrect;
  final int masteryLevel;
  final String? notes;

  const Word({
    this.id,
    required this.word,
    this.reading,
    this.meaning,
    this.partOfSpeech,
    this.jlptLevel,
    required this.dateAdded,
    this.timesSeen = 0,
    this.timesCorrect = 0,
    this.masteryLevel = 0,
    this.notes,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'word': word,
      'reading': reading,
      'meaning': meaning,
      'part_of_speech': partOfSpeech,
      'jlpt_level': jlptLevel,
      'date_added': dateAdded.toIso8601String(),
      'times_seen': timesSeen,
      'times_correct': timesCorrect,
      'mastery_level': masteryLevel,
      'notes': notes,
    };
  }

  factory Word.fromMap(Map<String, dynamic> map) {
    return Word(
      id: map['id'] as int?,
      word: map['word'] as String,
      reading: map['reading'] as String?,
      meaning: map['meaning'] as String?,
      partOfSpeech: map['part_of_speech'] as String?,
      jlptLevel: map['jlpt_level'] as String?,
      dateAdded: DateTime.parse(map['date_added'] as String),
      timesSeen: map['times_seen'] as int? ?? 0,
      timesCorrect: map['times_correct'] as int? ?? 0,
      masteryLevel: map['mastery_level'] as int? ?? 0,
      notes: map['notes'] as String?,
    );
  }

  Word copyWith({
    int? id,
    String? word,
    String? reading,
    String? meaning,
    String? partOfSpeech,
    String? jlptLevel,
    DateTime? dateAdded,
    int? timesSeen,
    int? timesCorrect,
    int? masteryLevel,
    String? notes,
  }) {
    return Word(
      id: id ?? this.id,
      word: word ?? this.word,
      reading: reading ?? this.reading,
      meaning: meaning ?? this.meaning,
      partOfSpeech: partOfSpeech ?? this.partOfSpeech,
      jlptLevel: jlptLevel ?? this.jlptLevel,
      dateAdded: dateAdded ?? this.dateAdded,
      timesSeen: timesSeen ?? this.timesSeen,
      timesCorrect: timesCorrect ?? this.timesCorrect,
      masteryLevel: masteryLevel ?? this.masteryLevel,
      notes: notes ?? this.notes,
    );
  }

  @override
  String toString() => 'Word(word: $word, reading: $reading, level: $jlptLevel)';
}
