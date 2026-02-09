/// Model representing a generated text response
class GeneratedText {
  final int? id;
  final String text;
  final String prompt;
  final String? constraintMode;
  final DateTime timestamp;

  const GeneratedText({
    this.id,
    required this.text,
    required this.prompt,
    this.constraintMode,
    required this.timestamp,
  });

  factory GeneratedText.fromJson(Map<String, dynamic> json) {
    return GeneratedText(
      text: json['text'] as String,
      prompt: json['prompt'] as String,
      constraintMode: json['constraint_mode'] as String?,
      timestamp: DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'prompt': prompt,
      'constraint_mode': constraintMode,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'text': text,
      'prompt': prompt,
      'constraint_mode': constraintMode,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  factory GeneratedText.fromMap(Map<String, dynamic> map) {
    return GeneratedText(
      id: map['id'] as int?,
      text: map['text'] as String,
      prompt: map['prompt'] as String,
      constraintMode: map['constraint_mode'] as String?,
      timestamp: DateTime.parse(map['timestamp'] as String),
    );
  }

  @override
  String toString() => 'GeneratedText(prompt: $prompt, mode: $constraintMode)';
}
