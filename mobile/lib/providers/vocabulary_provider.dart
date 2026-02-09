import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/word.dart';
import '../services/database_service.dart';

// Database service provider
final databaseServiceProvider = Provider<DatabaseService>((ref) {
  return DatabaseService();
});

// Vocabulary state
class VocabularyState {
  final List<Word> words;
  final bool isLoading;
  final String? error;
  final String? selectedLevel;

  const VocabularyState({
    this.words = const [],
    this.isLoading = false,
    this.error,
    this.selectedLevel,
  });

  VocabularyState copyWith({
    List<Word>? words,
    bool? isLoading,
    String? error,
    String? selectedLevel,
  }) {
    return VocabularyState(
      words: words ?? this.words,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      selectedLevel: selectedLevel ?? this.selectedLevel,
    );
  }
}

// Vocabulary provider
final vocabularyProvider =
    StateNotifierProvider<VocabularyNotifier, VocabularyState>((ref) {
  final dbService = ref.watch(databaseServiceProvider);
  return VocabularyNotifier(dbService);
});

class VocabularyNotifier extends StateNotifier<VocabularyState> {
  final DatabaseService dbService;

  VocabularyNotifier(this.dbService) : super(const VocabularyState()) {
    loadVocabulary();
  }

  Future<void> loadVocabulary() async {
    state = state.copyWith(isLoading: true, error: null);

    try {
      final words = state.selectedLevel != null
          ? await dbService.getVocabularyByLevel(state.selectedLevel!)
          : await dbService.getVocabulary();

      state = state.copyWith(
        words: words,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to load vocabulary: ${e.toString()}',
      );
    }
  }

  Future<void> addWord(Word word) async {
    try {
      await dbService.insertWord(word);
      await loadVocabulary(); // Reload to get updated list
    } catch (e) {
      state = state.copyWith(
        error: 'Failed to add word: ${e.toString()}',
      );
    }
  }

  Future<void> addWords(List<Word> words) async {
    state = state.copyWith(isLoading: true, error: null);

    try {
      await dbService.insertWords(words);
      await loadVocabulary();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to add words: ${e.toString()}',
      );
    }
  }

  Future<void> deleteWord(int id) async {
    try {
      await dbService.deleteWord(id);
      await loadVocabulary();
    } catch (e) {
      state = state.copyWith(
        error: 'Failed to delete word: ${e.toString()}',
      );
    }
  }

  Future<void> updateWord(Word word) async {
    try {
      await dbService.updateWord(word);
      await loadVocabulary();
    } catch (e) {
      state = state.copyWith(
        error: 'Failed to update word: ${e.toString()}',
      );
    }
  }

  Future<void> searchVocabulary(String query) async {
    if (query.isEmpty) {
      await loadVocabulary();
      return;
    }

    state = state.copyWith(isLoading: true, error: null);

    try {
      final words = await dbService.searchVocabulary(query);
      state = state.copyWith(
        words: words,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Search failed: ${e.toString()}',
      );
    }
  }

  Future<void> filterByLevel(String? level) async {
    state = state.copyWith(selectedLevel: level);
    await loadVocabulary();
  }

  void clearError() {
    state = state.copyWith(error: null);
  }
}

// Vocabulary stats provider
final vocabularyStatsProvider = FutureProvider<VocabularyStats>((ref) async {
  final dbService = ref.watch(databaseServiceProvider);

  final total = await dbService.getVocabularyCount();
  final n5 = await dbService.getVocabularyCountByLevel('N5');
  final n4 = await dbService.getVocabularyCountByLevel('N4');
  final n3 = await dbService.getVocabularyCountByLevel('N3');
  final n2 = await dbService.getVocabularyCountByLevel('N2');
  final n1 = await dbService.getVocabularyCountByLevel('N1');

  return VocabularyStats(
    total: total,
    n5: n5,
    n4: n4,
    n3: n3,
    n2: n2,
    n1: n1,
  );
});

class VocabularyStats {
  final int total;
  final int n5;
  final int n4;
  final int n3;
  final int n2;
  final int n1;

  const VocabularyStats({
    required this.total,
    required this.n5,
    required this.n4,
    required this.n3,
    required this.n2,
    required this.n1,
  });
}
