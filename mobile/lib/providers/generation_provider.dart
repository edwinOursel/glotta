import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/user_settings.dart';
import '../models/generated_text.dart';
import '../models/vocabulary_item.dart';
import '../services/api_service.dart';
import 'grammar_provider.dart';

// Settings provider
final settingsProvider = StateNotifierProvider<SettingsNotifier, UserSettings>((ref) {
  return SettingsNotifier();
});

class SettingsNotifier extends StateNotifier<UserSettings> {
  SettingsNotifier() : super(UserSettings.defaultSettings());

  void updateApiUrl(String url) {
    state = state.copyWith(apiBaseUrl: url);
  }

  void updateConstraintMode(String mode) {
    state = state.copyWith(constraintMode: mode);
  }

  void updateJlptLevel(String level) {
    state = state.copyWith(jlptLevel: level);
  }

  void toggleConstraints() {
    state = state.copyWith(useConstraints: !state.useConstraints);
  }
}

// API service provider (singleton)
final apiServiceProvider = Provider<ApiService>((ref) {
  final settings = ref.watch(settingsProvider);
  return ApiService(baseUrl: settings.apiBaseUrl);
});

// Focus word provider — set by VocabularyScreen before switching to LearnScreen.
// The LearnScreen pre-fills the prompt and GenerationNotifier builds a
// word-specific system prompt hint from it.
final focusWordProvider = StateProvider<VocabularyItem?>((ref) => null);

// Generation state
class GenerationState {
  final List<GeneratedText> history;
  final bool isLoading;
  final String? error;

  const GenerationState({
    this.history = const [],
    this.isLoading = false,
    this.error,
  });

  GenerationState copyWith({
    List<GeneratedText>? history,
    bool? isLoading,
    String? error,
  }) {
    return GenerationState(
      history: history ?? this.history,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

// Generation provider
final generationProvider = StateNotifierProvider<GenerationNotifier, GenerationState>((ref) {
  final apiService   = ref.watch(apiServiceProvider);
  final settings     = ref.watch(settingsProvider);
  final grammarState = ref.watch(grammarProvider);
  return GenerationNotifier(apiService, settings, ref, grammarState);
});

class GenerationNotifier extends StateNotifier<GenerationState> {
  final ApiService   apiService;
  final UserSettings settings;
  final Ref          _ref;
  final GrammarState _grammarState;

  GenerationNotifier(this.apiService, this.settings, this._ref, this._grammarState)
      : super(const GenerationState());

  /// Build the system_prompt string from active grammar theme and/or focus word.
  String? _buildSystemPrompt() {
    final focusWord    = _ref.read(focusWordProvider);
    final grammarHint  = _grammarState.selected?.systemPromptHint;

    if (focusWord != null) {
      final reading = focusWord.reading != null ? '（${focusWord.reading}）' : '';
      final meaning = focusWord.meaning != null ? ' — ${focusWord.meaning}' : '';
      final wordHint =
          'この文章に単語「${focusWord.word}」$reading$meaningを自然に使ってください。';
      return grammarHint != null ? '$wordHint $grammarHint' : wordHint;
    }
    return grammarHint;
  }

  Future<void> generateText({
    required String prompt,
    int maxLength = 50,
    double temperature = 0.8,
  }) async {
    if (prompt.trim().isEmpty) {
      state = state.copyWith(error: 'Please enter a prompt');
      return;
    }

    state = state.copyWith(isLoading: true, error: null);

    final selectedTheme  = _grammarState.selected;
    final systemPrompt   = _buildSystemPrompt();

    try {
      final response = await apiService.generateText(
        prompt:         prompt,
        maxLength:      maxLength,
        temperature:    temperature,
        useConstraints: settings.useConstraints,
        constraintMode: settings.constraintMode,
        numSequences:   1,
        systemPrompt:   systemPrompt,
      );

      final generated = GeneratedText(
        text:          response.texts.first,
        prompt:        prompt,
        constraintMode: settings.useConstraints ? settings.constraintMode : null,
        timestamp:     DateTime.now(),
      );

      state = state.copyWith(
        history:   [generated, ...state.history],
        isLoading: false,
      );

      // Record session progress for the active grammar theme
      if (selectedTheme != null) {
        await _ref.read(grammarProvider.notifier).recordSession(selectedTheme.id);
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to generate text: ${e.toString()}',
      );
    }
  }

  void clearHistory() {
    state = const GenerationState();
  }

  void clearError() {
    state = state.copyWith(error: null);
  }
}
