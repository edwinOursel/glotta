import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/vocabulary_item.dart';
import '../providers/auth_provider.dart';
import '../providers/generation_provider.dart' show apiServiceProvider;

// ── State ──────────────────────────────────────────────────────────────────────

enum ReviewPhase { loading, card, done }

class ReviewSessionState {
  final ReviewPhase phase;
  final List<VocabularyItem> queue;   // words to review in order
  final int currentIndex;            // index into queue
  final Map<String, int> results;    // wordId → quality given
  final String? error;

  const ReviewSessionState({
    this.phase        = ReviewPhase.loading,
    this.queue        = const [],
    this.currentIndex = 0,
    this.results      = const {},
    this.error,
  });

  ReviewSessionState copyWith({
    ReviewPhase?             phase,
    List<VocabularyItem>?    queue,
    int?                     currentIndex,
    Map<String, int>?        results,
    String?                  error,
    bool                     clearError = false,
  }) =>
      ReviewSessionState(
        phase:        phase        ?? this.phase,
        queue:        queue        ?? this.queue,
        currentIndex: currentIndex ?? this.currentIndex,
        results:      results      ?? this.results,
        error:        clearError ? null : error ?? this.error,
      );

  // Convenience getters
  bool get isDone     => currentIndex >= queue.length;
  int  get total      => queue.length;
  int  get reviewed   => results.length;

  /// Number of cards where quality >= 3 (answered correctly).
  int get correctCount =>
      results.values.where((q) => q >= 3).length;

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

  // ── Load due words ──────────────────────────────────────────────────────────

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

      state = state.copyWith(
        phase:        words.isEmpty ? ReviewPhase.done : ReviewPhase.card,
        queue:        words,
        currentIndex: 0,
        results:      {},
      );
    } catch (e) {
      state = state.copyWith(
        phase: ReviewPhase.done,
        error: 'Failed to load due words: $e',
      );
    }
  }

  // ── Submit a rating and advance to next card ────────────────────────────────

  /// [quality] must be 0-5 (Hard=1, OK=3, Easy=5).
  Future<void> submitRating(int quality) async {
    final word = state.current;
    if (word == null) return;

    // Optimistically advance the card so the UI is snappy
    final newResults = Map<String, int>.from(state.results)..[word.id] = quality;
    final nextIndex  = state.currentIndex + 1;
    final nextPhase  = nextIndex >= state.queue.length
        ? ReviewPhase.done
        : ReviewPhase.card;

    state = state.copyWith(
      results:      newResults,
      currentIndex: nextIndex,
      phase:        nextPhase,
    );

    // Fire-and-forget API call — failure is silent (word will come back next session)
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.reviewWord(
        accessToken: token,
        wordId:      word.id,
        quality:     quality,
      );
    } catch (_) {
      // Non-critical: SM-2 update failed but session continues
    }
  }

  // ── Reset (start a new session) ─────────────────────────────────────────────

  Future<void> restart() => loadDueWords();
}

// ── Provider ───────────────────────────────────────────────────────────────────

final reviewProvider =
    StateNotifierProvider.autoDispose<ReviewNotifier, ReviewSessionState>(
  (ref) => ReviewNotifier(ref),
);
