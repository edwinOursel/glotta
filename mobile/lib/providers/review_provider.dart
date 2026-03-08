import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/vocabulary_item.dart';
import '../providers/auth_provider.dart';
import '../providers/gamification_provider.dart'
    show streakProvider, trophyProvider;
import '../providers/generation_provider.dart' show apiServiceProvider;
import '../providers/vocabulary_provider.dart' show vocabularyProvider;

// ── State ──────────────────────────────────────────────────────────────────────

enum ReviewPhase { loading, card, done }

class ReviewSessionState {
  final ReviewPhase          phase;
  final List<VocabularyItem> queue;        // words to review in order
  final int                  currentIndex;
  final Map<String, int>     results;      // wordId → quality given
  final String?              _sessionId;   // backend session id (internal)
  final String?              error;

  const ReviewSessionState({
    this.phase        = ReviewPhase.loading,
    this.queue        = const [],
    this.currentIndex = 0,
    this.results      = const {},
    String? sessionId,
    this.error,
  }) : _sessionId = sessionId;

  ReviewSessionState copyWith({
    ReviewPhase?          phase,
    List<VocabularyItem>? queue,
    int?                  currentIndex,
    Map<String, int>?     results,
    String?               sessionId,
    String?               error,
    bool                  clearError   = false,
    bool                  clearSession = false,
  }) =>
      ReviewSessionState(
        phase:        phase        ?? this.phase,
        queue:        queue        ?? this.queue,
        currentIndex: currentIndex ?? this.currentIndex,
        results:      results      ?? this.results,
        sessionId:    clearSession ? null : sessionId ?? _sessionId,
        error:        clearError   ? null : error     ?? this.error,
      );

  bool get isDone      => currentIndex >= queue.length;
  int  get total       => queue.length;
  int  get reviewed    => results.length;

  /// Cards answered correctly (quality >= 3).
  int get correctCount => results.values.where((q) => q >= 3).length;

  VocabularyItem? get current =>
      currentIndex < queue.length ? queue[currentIndex] : null;
}

// ── Notifier ───────────────────────────────────────────────────────────────────

class ReviewNotifier extends StateNotifier<ReviewSessionState> {
  final Ref _ref;

  ReviewNotifier(this._ref) : super(const ReviewSessionState()) {
    loadDueWords();
  }

  Future<String?> _token() =>
      _ref.read(authProvider.notifier).getAccessToken();

  // ── Load due words + open a backend session ────────────────────────────────

  Future<void> loadDueWords() async {
    state = state.copyWith(phase: ReviewPhase.loading, clearError: true);
    try {
      final token = await _token();
      if (token == null) {
        state = state.copyWith(phase: ReviewPhase.done, error: 'Not authenticated');
        return;
      }
      final api  = _ref.read(apiServiceProvider);
      final raw  = await api.getDueWords(accessToken: token, limit: 20);
      final words = raw.map(VocabularyItem.fromJson).toList();

      if (words.isEmpty) {
        state = state.copyWith(
          phase:        ReviewPhase.done,
          queue:        [],
          currentIndex: 0,
          results:      {},
          clearSession: true,
        );
        return;
      }

      // Open a backend session so it counts in stats
      String? sessionId;
      try {
        final session = await api.startSession(
          accessToken: token,
          sessionType: 'review',
        );
        sessionId = session['id'] as String?;
      } catch (e) {
        debugPrint('[ReviewNotifier] Failed to open backend session: $e');
      }

      state = state.copyWith(
        phase:        ReviewPhase.card,
        queue:        words,
        currentIndex: 0,
        results:      {},
        sessionId:    sessionId,
      );
    } catch (e) {
      state = state.copyWith(
        phase: ReviewPhase.done,
        error: 'Failed to load due words: $e',
      );
    }
  }

  // ── Submit a rating and advance to next card ────────────────────────────────

  /// [quality] 0-5 (Hard=1, OK=3, Easy=5).
  Future<void> submitRating(int quality) async {
    final word = state.current;
    if (word == null) return;

    final newResults = Map<String, int>.from(state.results)..[word.id] = quality;
    final nextIndex  = state.currentIndex + 1;
    final isLastCard = nextIndex >= state.queue.length;

    state = state.copyWith(
      results:      newResults,
      currentIndex: nextIndex,
      phase:        isLastCard ? ReviewPhase.done : ReviewPhase.card,
    );

    // Fire-and-forget SM-2 update
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.reviewWord(
        accessToken: token,
        wordId:      word.id,
        quality:     quality,
      );
    } catch (e) {
      debugPrint('[ReviewNotifier] Failed to submit SM-2 rating: $e');
    }

    // When the last card is rated: close session + refresh downstream state
    if (isLastCard) {
      await _finaliseSession(newResults);
    }
  }

  // ── Finalise session ────────────────────────────────────────────────────────

  Future<void> _finaliseSession(Map<String, int> results) async {
    final reviewed = results.length;
    final correct  = results.values.where((q) => q >= 3).length;

    // 1. Close the backend session (best-effort)
    try {
      final token = await _token();
      if (token != null && state._sessionId != null) {
        final api = _ref.read(apiServiceProvider);
        await api.endSession(
          accessToken:   token,
          sessionId:     state._sessionId!,
          wordsPracticed: reviewed,
          wordsCorrect:   correct,
        );
      }
    } catch (e) {
      debugPrint('[ReviewNotifier] Failed to end backend session: $e');
    }

    // 2. Refresh vocabulary so mastery levels + due counts are up-to-date
    try {
      await _ref.read(vocabularyProvider.notifier).load();
    } catch (e) {
      debugPrint('[ReviewNotifier] Failed to refresh vocabulary: $e');
    }

    // 3. Record activity for streak + re-evaluate trophies
    if (reviewed > 0) {
      try {
        _ref.read(streakProvider.notifier).recordActivity();
        await _ref.read(trophyProvider.notifier).evaluate();
      } catch (e) {
        debugPrint('[ReviewNotifier] Failed to update streak/trophies: $e');
      }
    }
  }

  // ── Restart (new session from scratch) ──────────────────────────────────────

  Future<void> restart() => loadDueWords();
}

// ── Provider ───────────────────────────────────────────────────────────────────

final reviewProvider =
    StateNotifierProvider.autoDispose<ReviewNotifier, ReviewSessionState>(
  (ref) => ReviewNotifier(ref),
);
