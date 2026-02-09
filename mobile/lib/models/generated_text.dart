/// Model representing a generated text response
class GeneratedText {
  final String text;
  final String prompt;
  final String? constraintMode;
  final DateTime timestamp;

  const GeneratedText({
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

  @override
  String toString() => 'GeneratedText(prompt: $prompt, mode: $constraintMode)';
}
