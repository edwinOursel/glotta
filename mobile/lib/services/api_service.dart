import 'dart:convert';
import 'package:http/http.dart' as http;

/// Service for communicating with the Glotta Python backend API
class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({
    this.baseUrl = 'http://localhost:8000',
    http.Client? client,
  }) : _client = client ?? http.Client();

  // =========================================================================
  // Text Generation
  // =========================================================================

  /// Generate Japanese text with vocabulary constraints
  Future<GenerateResponse> generateText({
    required String prompt,
    int maxLength = 50,
    double temperature = 0.8,
    bool useConstraints = true,
    String constraintMode = 'hard',
    int numSequences = 1,
  }) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'prompt': prompt,
          'max_length': maxLength,
          'temperature': temperature,
          'use_constraints': useConstraints,
          'constraint_mode': constraintMode,
          'num_sequences': numSequences,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return GenerateResponse.fromJson(data);
      } else {
        throw ApiException(
          'Failed to generate text: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  // =========================================================================
  // Vocabulary Management
  // =========================================================================

  /// Get current vocabulary statistics
  Future<VocabularyStats> getVocabulary() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/api/vocabulary'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return VocabularyStats.fromJson(data);
      } else {
        throw ApiException(
          'Failed to get vocabulary: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  /// Add words to user's vocabulary
  Future<AddWordsResponse> addWords(List<String> words) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/api/vocabulary/words'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'words': words,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return AddWordsResponse.fromJson(data);
      } else {
        throw ApiException(
          'Failed to add words: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  /// Remove a word from user's vocabulary
  Future<RemoveWordResponse> removeWord(String word) async {
    try {
      final response = await _client.delete(
        Uri.parse('$baseUrl/api/vocabulary/words/$word'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return RemoveWordResponse.fromJson(data);
      } else {
        throw ApiException(
          'Failed to remove word: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  /// Save vocabulary to file
  Future<void> saveVocabulary({String filename = 'user_vocabulary.json'}) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/api/vocabulary/save?filename=$filename'),
      );

      if (response.statusCode != 200) {
        throw ApiException(
          'Failed to save vocabulary: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  /// Load vocabulary from file
  Future<void> loadVocabulary({String filename = 'user_vocabulary.json'}) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/api/vocabulary/load?filename=$filename'),
      );

      if (response.statusCode != 200) {
        throw ApiException(
          'Failed to load vocabulary: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  // =========================================================================
  // Settings
  // =========================================================================

  /// Get current constraint mode
  Future<String> getConstraintMode() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/api/settings/constraint-mode'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['constraint_mode'] as String;
      } else {
        throw ApiException(
          'Failed to get constraint mode: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  /// Set constraint mode (hard/soft/adaptive)
  Future<void> setConstraintMode(String mode) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/api/settings/constraint-mode?mode=$mode'),
      );

      if (response.statusCode != 200) {
        throw ApiException(
          'Failed to set constraint mode: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  // =========================================================================
  // Statistics
  // =========================================================================

  /// Get general statistics
  Future<StatsResponse> getStats() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/api/stats'),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return StatsResponse.fromJson(data);
      } else {
        throw ApiException(
          'Failed to get stats: ${response.statusCode}',
          response.statusCode,
        );
      }
    } catch (e) {
      throw ApiException('Network error: $e', 0);
    }
  }

  // =========================================================================
  // Health Check
  // =========================================================================

  /// Check if the API server is healthy
  Future<bool> healthCheck() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/health'),
      );
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Dispose the HTTP client
  void dispose() {
    _client.close();
  }
}

// =============================================================================
// Response Models
// =============================================================================

class GenerateResponse {
  final List<String> texts;
  final String prompt;
  final String? constraintMode;

  GenerateResponse({
    required this.texts,
    required this.prompt,
    this.constraintMode,
  });

  factory GenerateResponse.fromJson(Map<String, dynamic> json) {
    return GenerateResponse(
      texts: List<String>.from(json['texts']),
      prompt: json['prompt'] as String,
      constraintMode: json['constraint_mode'] as String?,
    );
  }
}

class VocabularyStats {
  final int totalWords;
  final int totalExpressions;
  final int totalTokenIds;
  final List<String> sampleWords;

  VocabularyStats({
    required this.totalWords,
    required this.totalExpressions,
    required this.totalTokenIds,
    required this.sampleWords,
  });

  factory VocabularyStats.fromJson(Map<String, dynamic> json) {
    return VocabularyStats(
      totalWords: json['total_words'] as int,
      totalExpressions: json['total_expressions'] as int,
      totalTokenIds: json['total_token_ids'] as int,
      sampleWords: List<String>.from(json['sample_words']),
    );
  }
}

class AddWordsResponse {
  final String status;
  final int wordsAdded;
  final int totalWords;

  AddWordsResponse({
    required this.status,
    required this.wordsAdded,
    required this.totalWords,
  });

  factory AddWordsResponse.fromJson(Map<String, dynamic> json) {
    return AddWordsResponse(
      status: json['status'] as String,
      wordsAdded: json['words_added'] as int,
      totalWords: json['total_words'] as int,
    );
  }
}

class RemoveWordResponse {
  final String status;
  final String wordRemoved;
  final int totalWords;

  RemoveWordResponse({
    required this.status,
    required this.wordRemoved,
    required this.totalWords,
  });

  factory RemoveWordResponse.fromJson(Map<String, dynamic> json) {
    return RemoveWordResponse(
      status: json['status'] as String,
      wordRemoved: json['word_removed'] as String,
      totalWords: json['total_words'] as int,
    );
  }
}

class StatsResponse {
  final int vocabularySize;
  final String constraintMode;
  final String modelName;

  StatsResponse({
    required this.vocabularySize,
    required this.constraintMode,
    required this.modelName,
  });

  factory StatsResponse.fromJson(Map<String, dynamic> json) {
    return StatsResponse(
      vocabularySize: json['vocabulary_size'] as int,
      constraintMode: json['constraint_mode'] as String,
      modelName: json['model_name'] as String,
    );
  }
}

// =============================================================================
// Exceptions
// =============================================================================

class ApiException implements Exception {
  final String message;
  final int statusCode;

  ApiException(this.message, this.statusCode);

  @override
  String toString() => 'ApiException: $message (status: $statusCode)';
}
