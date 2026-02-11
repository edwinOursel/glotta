import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/vocabulary_item.dart';
import '../providers/auth_provider.dart';
import '../providers/generation_provider.dart'
    show apiServiceProvider, focusWordProvider;

export '../providers/generation_provider.dart' show focusWordProvider;

// ── State ─────────────────────────────────────────────────────────────────────

class VocabularyState {
  final List<VocabularyItem> words;     // current filtered view (for display)
  final List<VocabularyItem> allWords;  // complete list regardless of filter
  final bool   isLoading;
  final String? error;
  final String? selectedLevel;  // null = all

  const VocabularyState({
    this.words         = const [],
    this.allWords      = const [],
    this.isLoading     = false,
    this.error,
    this.selectedLevel,
  });

  VocabularyState copyWith({
    List<VocabularyItem>? words,
    List<VocabularyItem>? allWords,
    bool?   isLoading,
    String? error,
    String? selectedLevel,
    bool    clearError         = false,
    bool    clearSelectedLevel = false,
  }) =>
      VocabularyState(
        words:         words         ?? this.words,
        allWords:      allWords      ?? this.allWords,
        isLoading:     isLoading     ?? this.isLoading,
        error:         clearError    ? null : error ?? this.error,
        selectedLevel: clearSelectedLevel ? null : selectedLevel ?? this.selectedLevel,
      );

  // Due words are those whose nextReviewAt is in the past.
  List<VocabularyItem> get dueWords =>
      words.where((w) => w.isDueForReview).toList();
}

// ── Notifier ──────────────────────────────────────────────────────────────────

class VocabularyNotifier extends StateNotifier<VocabularyState> {
  final Ref _ref;

  VocabularyNotifier(this._ref) : super(const VocabularyState()) {
    load();
  }

  Future<String?> _token() =>
      _ref.read(authProvider.notifier).getAccessToken();

  // ── Load ───────────────────────────────────────────────────────────────────

  Future<void> load() async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final token = await _token();
      if (token == null) {
        state = state.copyWith(isLoading: false);
        return;
      }
      final api      = _ref.read(apiServiceProvider);
      final rawFilt  = await api.getUserVocabulary(
        accessToken: token,
        level:       state.selectedLevel,
        limit:       500,
      );
      final filtered = rawFilt.map(VocabularyItem.fromJson).toList();

      // Also keep an unfiltered total for stats/trophies
      List<VocabularyItem> all = filtered;
      if (state.selectedLevel != null) {
        final rawAll = await api.getUserVocabulary(
          accessToken: token,
          level:       null,
          limit:       500,
        );
        all = rawAll.map(VocabularyItem.fromJson).toList();
      }

      state = state.copyWith(
        words:     filtered,
        allWords:  all,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to load vocabulary: $e',
      );
    }
  }

  // ── Filter by JLPT level ───────────────────────────────────────────────────

  Future<void> filterByLevel(String? level) async {
    state = state.copyWith(
      selectedLevel:      level,
      clearSelectedLevel: level == null,
    );
    await load();
  }

  // ── Add a single word ──────────────────────────────────────────────────────

  Future<VocabularyItem?> addWord({
    required String word,
    String? reading,
    String? meaning,
    String? jlptLevel,
  }) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final token = await _token();
      if (token == null) throw Exception('Not authenticated');

      final api  = _ref.read(apiServiceProvider);
      final raw  = await api.addWord(
        accessToken: token,
        word:        word,
        reading:     reading,
        meaning:     meaning,
        jlptLevel:   jlptLevel,
      );
      final item = VocabularyItem.fromJson(raw);
      state = state.copyWith(
        words:     [item, ...state.words],
        allWords:  [item, ...state.allWords],
        isLoading: false,
      );
      return item;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to add word: $e',
      );
      return null;
    }
  }

  // ── Delete ─────────────────────────────────────────────────────────────────

  Future<void> deleteWord(String id) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.deleteWord(accessToken: token, id: id);
      state = state.copyWith(
        words:    state.words.where((w) => w.id != id).toList(),
        allWords: state.allWords.where((w) => w.id != id).toList(),
      );
    } catch (e) {
      state = state.copyWith(error: 'Failed to delete word: $e');
    }
  }

  // ── Seed JLPT level ────────────────────────────────────────────────────────

  Future<void> seedLevel(String level) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final token = await _token();
      if (token == null) throw Exception('Not authenticated');
      final api = _ref.read(apiServiceProvider);
      await api.seedJlptLevel(accessToken: token, level: level);
      await load();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to seed $level vocabulary: $e',
      );
    }
  }

  void clearError() => state = state.copyWith(clearError: true);
}

// ── Providers ─────────────────────────────────────────────────────────────────

final vocabularyProvider =
    StateNotifierProvider<VocabularyNotifier, VocabularyState>(
  (ref) => VocabularyNotifier(ref),
);

final dueWordsCountProvider = Provider<int>((ref) {
  return ref.watch(vocabularyProvider).dueWords.length;
});
