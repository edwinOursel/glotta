import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/user_profile.dart';

// ── Service ───────────────────────────────────────────────────────────────────

/// HTTP client for the Glotta Python backend.
///
/// Auth endpoints are unauthenticated.
/// All other endpoints accept an optional [accessToken] — pass it so the
/// server can scope data to the current user.
class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({
    this.baseUrl = const String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'http://10.0.2.2:8000', // Android emulator → host; override with --dart-define
    ),
    http.Client? client,
  }) : _client = client ?? http.Client();

  // ── Internal helpers ───────────────────────────────────────────────────────

  static const _kTimeout = Duration(seconds: 30);

  Map<String, String> _headers({String? accessToken}) => {
        'Content-Type': 'application/json',
        if (accessToken != null) 'Authorization': 'Bearer $accessToken',
      };

  Future<http.Response> _get(Uri uri, {String? accessToken}) async {
    try {
      return await _client
          .get(uri, headers: _headers(accessToken: accessToken))
          .timeout(_kTimeout);
    } on TimeoutException {
      throw ApiException('Request timed out', 408);
    }
  }

  Future<http.Response> _post(Uri uri, {String? accessToken, String? body}) async {
    try {
      return await _client
          .post(uri, headers: _headers(accessToken: accessToken), body: body)
          .timeout(_kTimeout);
    } on TimeoutException {
      throw ApiException('Request timed out', 408);
    }
  }

  Future<http.Response> _patch(Uri uri, {String? accessToken, String? body}) async {
    try {
      return await _client
          .patch(uri, headers: _headers(accessToken: accessToken), body: body)
          .timeout(_kTimeout);
    } on TimeoutException {
      throw ApiException('Request timed out', 408);
    }
  }

  Future<http.Response> _delete(Uri uri, {String? accessToken}) async {
    try {
      return await _client
          .delete(uri, headers: _headers(accessToken: accessToken))
          .timeout(_kTimeout);
    } on TimeoutException {
      throw ApiException('Request timed out', 408);
    }
  }

  /// Throws [ApiException] for non-2xx responses.
  void _checkStatus(http.Response response, String context) {
    if (response.statusCode >= 200 && response.statusCode < 300) return;
    final body = _tryDecode(response.body);
    final detail = body is Map ? body['detail'] ?? response.body : response.body;
    throw ApiException('$context: $detail', response.statusCode);
  }

  dynamic _tryDecode(String body) {
    try {
      return jsonDecode(body);
    } catch (_) {
      return body;
    }
  }

  // ==========================================================================
  // Auth
  // ==========================================================================

  /// Register a new account. Returns {access_token, refresh_token}.
  Future<Map<String, String>> register({
    required String email,
    required String password,
    String? username,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/auth/register'),
      body: jsonEncode({
        'email':    email,
        'password': password,
        if (username != null) 'username': username,
      }),
    );
    _checkStatus(response, 'register');
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return {
      'access_token':  data['access_token']  as String,
      'refresh_token': data['refresh_token'] as String,
    };
  }

  /// Login. Returns {access_token, refresh_token}.
  Future<Map<String, String>> login({
    required String email,
    required String password,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/auth/login'),
      body: jsonEncode({'email': email, 'password': password}),
    );
    _checkStatus(response, 'login');
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return {
      'access_token':  data['access_token']  as String,
      'refresh_token': data['refresh_token'] as String,
    };
  }

  /// Exchange a refresh token for a new access token.
  Future<String> refreshToken({required String refreshToken}) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/auth/refresh'),
      body: jsonEncode({'refresh_token': refreshToken}),
    );
    _checkStatus(response, 'refresh');
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return data['access_token'] as String;
  }

  /// Fetch the authenticated user's profile.
  Future<UserProfile> getProfile({required String accessToken}) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/auth/me'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getProfile');
    return UserProfile.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  /// Update mutable profile fields.
  Future<UserProfile> updateProfile({
    required String accessToken,
    String? username,
    String? jlptLevel,
    String? constraintMode,
  }) async {
    final body = <String, dynamic>{
      if (username       != null) 'username':       username,
      if (jlptLevel      != null) 'jlpt_level':     jlptLevel,
      if (constraintMode != null) 'constraint_mode': constraintMode,
    };
    final response = await _patch(
      Uri.parse('$baseUrl/api/user/profile'),
      accessToken: accessToken,
      body: jsonEncode(body),
    );
    _checkStatus(response, 'updateProfile');
    return UserProfile.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  // ==========================================================================
  // Text Generation
  // ==========================================================================

  Future<GenerateResponse> generateText({
    required String prompt,
    String? accessToken,
    String? systemPrompt,
    int    maxLength       = 50,
    double temperature     = 0.8,
    bool   useConstraints  = true,
    String constraintMode  = 'hard',
    int    numSequences    = 1,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/generate'),
      accessToken: accessToken,
      body: jsonEncode({
        'prompt':          prompt,
        'max_length':      maxLength,
        'temperature':     temperature,
        'use_constraints': useConstraints,
        'constraint_mode': constraintMode,
        'num_sequences':   numSequences,
        if (systemPrompt != null) 'system_prompt': systemPrompt,
      }),
    );
    _checkStatus(response, 'generateText');
    return GenerateResponse.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  // ==========================================================================
  // User Vocabulary (authenticated)
  // ==========================================================================

  Future<List<Map<String, dynamic>>> getUserVocabulary({
    required String accessToken,
    String? level,
    String? search,
    int limit  = 50,
    int offset = 0,
  }) async {
    final query = {
      if (level  != null) 'level':  level,
      if (search != null) 'search': search,
      'limit':  '$limit',
      'offset': '$offset',
    };
    final uri = Uri.parse('$baseUrl/api/vocabulary').replace(queryParameters: query);
    final response = await _get(uri, accessToken: accessToken);
    _checkStatus(response, 'getUserVocabulary');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<Map<String, dynamic>> getVocabularyStats({required String accessToken}) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/vocabulary/stats'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getVocabularyStats');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> getDueWords({
    required String accessToken,
    int limit = 20,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/vocabulary/due?limit=$limit'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getDueWords');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<Map<String, dynamic>> addWord({
    required String accessToken,
    required String word,
    String? reading,
    String? meaning,
    String? partOfSpeech,
    String? jlptLevel,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/vocabulary'),
      accessToken: accessToken,
      body: jsonEncode({
        'word': word,
        if (reading      != null) 'reading':       reading,
        if (meaning      != null) 'meaning':        meaning,
        if (partOfSpeech != null) 'part_of_speech': partOfSpeech,
        if (jlptLevel    != null) 'jlpt_level':     jlptLevel,
      }),
    );
    _checkStatus(response, 'addWord');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> reviewWord({
    required String accessToken,
    required String wordId,
    required int    quality,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/vocabulary/$wordId/review'),
      accessToken: accessToken,
      body: jsonEncode({'quality': quality}),
    );
    _checkStatus(response, 'reviewWord');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> deleteWord({
    required String accessToken,
    required String wordId,
  }) async {
    final response = await _delete(
      Uri.parse('$baseUrl/api/vocabulary/$wordId'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'deleteWord');
  }

  Future<Map<String, dynamic>> seedJlptLevel({
    required String accessToken,
    required String jlptLevel,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/vocabulary/seed/$jlptLevel'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'seedJlptLevel');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // ==========================================================================
  // Sessions & Progress
  // ==========================================================================

  Future<Map<String, dynamic>> startSession({
    required String accessToken,
    String sessionType    = 'generate',
    String? constraintMode,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/sessions/start'),
      accessToken: accessToken,
      body: jsonEncode({
        'session_type':    sessionType,
        if (constraintMode != null) 'constraint_mode': constraintMode,
      }),
    );
    _checkStatus(response, 'startSession');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> endSession({
    required String accessToken,
    required String sessionId,
    int wordsPracticed = 0,
    int wordsCorrect   = 0,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/sessions/end'),
      accessToken: accessToken,
      body: jsonEncode({
        'session_id':      sessionId,
        'words_practiced': wordsPracticed,
        'words_correct':   wordsCorrect,
      }),
    );
    _checkStatus(response, 'endSession');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> getProgress({required String accessToken}) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/sessions/progress'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getProgress');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // ==========================================================================
  // Friends
  // ==========================================================================

  Future<List<Map<String, dynamic>>> getFriends({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/friends'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getFriends');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<List<Map<String, dynamic>>> getIncomingRequests({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/friends/requests'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getIncomingRequests');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<List<Map<String, dynamic>>> getSentRequests({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/friends/sent'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getSentRequests');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<Map<String, dynamic>> sendFriendRequest({
    required String accessToken,
    required String addresseeId,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/friends/request'),
      accessToken: accessToken,
      body: jsonEncode({'addressee_id': addresseeId}),
    );
    _checkStatus(response, 'sendFriendRequest');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> acceptFriendRequest({
    required String accessToken,
    required String friendshipId,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/friends/request/$friendshipId/accept'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'acceptFriendRequest');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> declineFriendRequest({
    required String accessToken,
    required String friendshipId,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/friends/request/$friendshipId/decline'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'declineFriendRequest');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> unfriend({
    required String accessToken,
    required String userId,
  }) async {
    final response = await _delete(
      Uri.parse('$baseUrl/api/friends/$userId'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'unfriend');
  }

  Future<List<Map<String, dynamic>>> searchUsers({
    required String accessToken,
    required String query,
  }) async {
    final uri = Uri.parse('$baseUrl/api/friends/search')
        .replace(queryParameters: {'q': query});
    final response = await _get(uri, accessToken: accessToken);
    _checkStatus(response, 'searchUsers');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<List<Map<String, dynamic>>> getLeaderboard({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/friends/leaderboard'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getLeaderboard');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  // ==========================================================================
  // Challenges
  // ==========================================================================

  Future<List<Map<String, dynamic>>> getChallenges({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/challenges'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getChallenges');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<List<Map<String, dynamic>>> getChallengeHistory({
    required String accessToken,
  }) async {
    final response = await _get(
      Uri.parse('$baseUrl/api/challenges/history'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'getChallengeHistory');
    return List<Map<String, dynamic>>.from(jsonDecode(response.body) as List);
  }

  Future<Map<String, dynamic>> createChallenge({
    required String accessToken,
    required String recipientId,
    required String type,
    int durationDays = 7,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/challenges'),
      accessToken: accessToken,
      body: jsonEncode({
        'recipient_id': recipientId,
        'type': type,
        'duration_days': durationDays,
      }),
    );
    _checkStatus(response, 'createChallenge');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> acceptChallenge({
    required String accessToken,
    required String challengeId,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/challenges/$challengeId/accept'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'acceptChallenge');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> declineChallenge({
    required String accessToken,
    required String challengeId,
  }) async {
    final response = await _post(
      Uri.parse('$baseUrl/api/challenges/$challengeId/decline'),
      accessToken: accessToken,
    );
    _checkStatus(response, 'declineChallenge');
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // ==========================================================================
  // Health
  // ==========================================================================

  Future<bool> healthCheck() async {
    try {
      final response = await _get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  void dispose() => _client.close();
}

// =============================================================================
// Lightweight response models (kept for backward compat)
// =============================================================================

class GenerateResponse {
  final List<String> texts;
  final String prompt;
  final String? constraintMode;

  const GenerateResponse({
    required this.texts,
    required this.prompt,
    this.constraintMode,
  });

  factory GenerateResponse.fromJson(Map<String, dynamic> json) => GenerateResponse(
        texts:          List<String>.from(json['texts'] as List),
        prompt:         json['prompt'] as String,
        constraintMode: json['constraint_mode'] as String?,
      );
}

// =============================================================================
// Exception
// =============================================================================

class ApiException implements Exception {
  final String message;
  final int statusCode;

  const ApiException(this.message, this.statusCode);

  @override
  String toString() => 'ApiException($statusCode): $message';
}
