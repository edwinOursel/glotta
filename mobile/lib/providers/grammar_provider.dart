import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/grammar_theme.dart';

// ── State ─────────────────────────────────────────────────────────────────────

class GrammarState {
  final List<GrammarTheme>         themes;
  final Map<String, ThemeProgress> progress; // themeId → ThemeProgress
  final GrammarTheme?              selected;
  final bool                       isLoading;

  const GrammarState({
    this.themes    = const [],
    this.progress  = const {},
    this.selected  = null,
    this.isLoading = true,
  });

  GrammarState copyWith({
    List<GrammarTheme>?         themes,
    Map<String, ThemeProgress>? progress,
    GrammarTheme?               selected,
    bool?                       isLoading,
    bool                        clearSelected = false,
  }) =>
      GrammarState(
        themes:    themes    ?? this.themes,
        progress:  progress  ?? this.progress,
        selected:  clearSelected ? null : selected ?? this.selected,
        isLoading: isLoading ?? this.isLoading,
      );

  /// Whether a theme is unlocked given current progress.
  bool isUnlocked(GrammarTheme theme) {
    if (theme.prerequisiteIds.isEmpty) return true;
    return theme.prerequisiteIds.every(
      (id) => progress[id]?.isUnlocked ?? false,
    );
  }

  /// Themes grouped by category, in insertion order.
  Map<String, List<GrammarTheme>> get byCategory {
    final map = <String, List<GrammarTheme>>{};
    for (final t in themes) {
      map.putIfAbsent(t.category, () => []).add(t);
    }
    return map;
  }
}

// ── Notifier ──────────────────────────────────────────────────────────────────

class GrammarNotifier extends StateNotifier<GrammarState> {
  GrammarNotifier() : super(const GrammarState()) {
    _load();
  }

  static const _prefsKey = 'grammar_progress_v1';

  Future<void> _load() async {
    // Load themes from bundled asset
    final raw = await rootBundle.loadString('assets/data/grammar_themes.json');
    final list = (jsonDecode(raw) as List)
        .map((e) => GrammarTheme.fromJson(e as Map<String, dynamic>))
        .toList();

    // Load saved progress from SharedPreferences
    final prefs   = await SharedPreferences.getInstance();
    final saved   = prefs.getString(_prefsKey);
    final progress = <String, ThemeProgress>{};

    if (saved != null) {
      final decoded = jsonDecode(saved) as Map<String, dynamic>;
      for (final entry in decoded.entries) {
        final v = entry.value as Map<String, dynamic>;
        progress[entry.key] = ThemeProgress(
          themeId:      entry.key,
          masteryLevel: v['mastery'] as int? ?? 0,
          sessionsCount: v['sessions'] as int? ?? 0,
          isUnlocked:   v['unlocked'] as bool? ?? false,
        );
      }
    }

    // N5 themes are unlocked by default
    for (final theme in list) {
      if (theme.jlptLevel == 'N5' && theme.prerequisiteIds.isEmpty) {
        progress.putIfAbsent(
          theme.id,
          () => ThemeProgress(themeId: theme.id, isUnlocked: true),
        );
      }
    }

    state = GrammarState(
      themes:    list,
      progress:  progress,
      isLoading: false,
    );
  }

  Future<void> _save() async {
    final prefs = await SharedPreferences.getInstance();
    final data  = state.progress.map((k, v) => MapEntry(k, {
      'mastery':  v.masteryLevel,
      'sessions': v.sessionsCount,
      'unlocked': v.isUnlocked,
    }));
    await prefs.setString(_prefsKey, jsonEncode(data));
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  void selectTheme(GrammarTheme? theme) {
    state = theme == null
        ? state.copyWith(clearSelected: true)
        : state.copyWith(selected: theme);
  }

  /// Call after a generation session with this theme completes.
  Future<void> recordSession(String themeId) async {
    final current = state.progress[themeId] ??
        ThemeProgress(themeId: themeId, isUnlocked: true);

    final sessions  = current.sessionsCount + 1;
    final mastery   = _calculateMastery(sessions);
    final unlocked  = true;

    final updated = current.copyWith(
      sessionsCount: sessions,
      masteryLevel:  mastery,
      isUnlocked:    unlocked,
    );

    final newProgress = Map<String, ThemeProgress>.from(state.progress)
      ..[themeId] = updated;

    // Unlock themes whose prerequisites are now satisfied
    for (final theme in state.themes) {
      if (!newProgress.containsKey(theme.id) &&
          theme.prerequisiteIds.every((id) => newProgress[id]?.isUnlocked ?? false)) {
        newProgress[theme.id] = ThemeProgress(themeId: theme.id, isUnlocked: true);
      }
    }

    state = state.copyWith(progress: newProgress);
    await _save();
  }

  int _calculateMastery(int sessions) {
    if (sessions >= 10) return 3;
    if (sessions >= 5)  return 2;
    if (sessions >= 2)  return 1;
    return 0;
  }
}

// ── Providers ─────────────────────────────────────────────────────────────────

final grammarProvider = StateNotifierProvider<GrammarNotifier, GrammarState>(
  (_) => GrammarNotifier(),
);

/// Currently selected theme's system prompt hint, or null.
final activeSystemPromptProvider = Provider<String?>((ref) {
  return ref.watch(grammarProvider).selected?.systemPromptHint;
});
