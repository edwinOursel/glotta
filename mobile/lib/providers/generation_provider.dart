import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/user_settings.dart';
import '../models/generated_text.dart';
import '../services/api_service.dart';

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
  final apiService = ref.watch(apiServiceProvider);
  final settings = ref.watch(settingsProvider);
  return GenerationNotifier(apiService, settings);
});

class GenerationNotifier extends StateNotifier<GenerationState> {
  final ApiService apiService;
  final UserSettings settings;

  GenerationNotifier(this.apiService, this.settings) : super(const GenerationState());

  Future<void> generateText({
    required String prompt,
    int maxLength = 50,
    double temperature = 0.8,
  }) async {
    if (prompt.trim().isEmpty) {
      state = state.copyWith(error: 'Please enter a prompt');
      return;
    }

    // Start loading
    state = state.copyWith(isLoading: true, error: null);

    try {
      final response = await apiService.generateText(
        prompt: prompt,
        maxLength: maxLength,
        temperature: temperature,
        useConstraints: settings.useConstraints,
        constraintMode: settings.constraintMode,
        numSequences: 1,
      );

      // Create GeneratedText object
      final generated = GeneratedText(
        text: response.texts.first,
        prompt: prompt,
        constraintMode: settings.useConstraints ? settings.constraintMode : null,
        timestamp: DateTime.now(),
      );

      // Add to history
      state = state.copyWith(
        history: [generated, ...state.history],
        isLoading: false,
      );
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
