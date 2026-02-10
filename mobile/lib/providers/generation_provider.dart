import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/user_settings.dart';
import '../models/generated_text.dart';
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

    final selectedTheme = _grammarState.selected;

    try {
      final response = await apiService.generateText(
        prompt:         prompt,
        maxLength:      maxLength,
        temperature:    temperature,
        useConstraints: settings.useConstraints,
        constraintMode: settings.constraintMode,
        numSequences:   1,
        systemPrompt:   selectedTheme?.systemPromptHint,
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
