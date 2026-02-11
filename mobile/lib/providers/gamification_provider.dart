import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/trophy.dart';
import '../providers/generation_provider.dart' show generationProvider;
import '../providers/grammar_provider.dart';
import '../providers/vocabulary_provider.dart';

// ── Streak ────────────────────────────────────────────────────────────────────

class StreakState {
  final int  currentStreak;  // days
  final int  longestStreak;  // days
  final bool activityToday;

  const StreakState({
    required this.currentStreak,
    required this.longestStreak,
    required this.activityToday,
  });

  StreakState copyWith({int? currentStreak, int? longestStreak, bool? activityToday}) =>
      StreakState(
        currentStreak: currentStreak ?? this.currentStreak,
        longestStreak: longestStreak ?? this.longestStreak,
        activityToday: activityToday ?? this.activityToday,
      );
}

const _kLastActivityKey = 'streak_last_activity'; // ISO date string
const _kCurrentStreakKey = 'streak_current';
const _kLongestStreakKey = 'streak_longest';

class StreakNotifier extends StateNotifier<StreakState> {
  StreakNotifier()
      : super(const StreakState(
            currentStreak: 0, longestStreak: 0, activityToday: false)) {
    _load();
  }

  Future<void> _load() async {
    final prefs       = await SharedPreferences.getInstance();
    final lastRaw     = prefs.getString(_kLastActivityKey);
    final current     = prefs.getInt(_kCurrentStreakKey) ?? 0;
    final longest     = prefs.getInt(_kLongestStreakKey) ?? 0;
    final today       = _today();

    int newCurrent = current;
    bool active    = false;

    if (lastRaw != null) {
      final last = DateTime.parse(lastRaw);
      final diff = today.difference(last).inDays;
      if (diff == 0) {
        active = true;                // already active today
      } else if (diff == 1) {
        // yesterday — streak still valid, not yet active today
      } else {
        // gap > 1 day: streak broken
        newCurrent = 0;
        await prefs.setInt(_kCurrentStreakKey, 0);
      }
    }

    state = StreakState(
      currentStreak: newCurrent,
      longestStreak: longest,
      activityToday: active,
    );
  }

  /// Call whenever the user does something meaningful (generate, review, add word).
  Future<void> recordActivity() async {
    if (state.activityToday) return; // already counted today

    final prefs   = await SharedPreferences.getInstance();
    final lastRaw = prefs.getString(_kLastActivityKey);
    final today   = _today();

    int newCurrent = state.currentStreak;

    if (lastRaw != null) {
      final diff = today.difference(DateTime.parse(lastRaw)).inDays;
      if (diff == 1) {
        newCurrent += 1; // consecutive day
      } else if (diff > 1) {
        newCurrent = 1;  // streak broken, restart
      }
      // diff == 0: shouldn't reach here (guarded by activityToday)
    } else {
      newCurrent = 1; // first ever activity
    }

    final newLongest = newCurrent > state.longestStreak
        ? newCurrent
        : state.longestStreak;

    await prefs.setString(_kLastActivityKey, today.toIso8601String());
    await prefs.setInt(_kCurrentStreakKey,    newCurrent);
    await prefs.setInt(_kLongestStreakKey,    newLongest);

    state = state.copyWith(
      currentStreak: newCurrent,
      longestStreak: newLongest,
      activityToday: true,
    );
  }

  static DateTime _today() {
    final n = DateTime.now();
    return DateTime(n.year, n.month, n.day);
  }
}

final streakProvider =
    StateNotifierProvider<StreakNotifier, StreakState>((_) => StreakNotifier());

// ── Persistent counters ───────────────────────────────────────────────────────
// Lightweight counters written to SharedPreferences and exposed as providers.

const _kGenerationsKey    = 'stat_generations';
const _kFocusPracticesKey = 'stat_focus_practices';

class _CounterNotifier extends StateNotifier<int> {
  _CounterNotifier(this._key) : super(0) {
    _load();
  }
  final String _key;

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    state = prefs.getInt(_key) ?? 0;
  }

  Future<void> increment() async {
    state += 1;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_key, state);
  }
}

final generationsCountProvider =
    StateNotifierProvider<_CounterNotifier, int>(
        (_) => _CounterNotifier(_kGenerationsKey));

final focusPracticesCountProvider =
    StateNotifierProvider<_CounterNotifier, int>(
        (_) => _CounterNotifier(_kFocusPracticesKey));

// ── Trophy state ──────────────────────────────────────────────────────────────

class TrophyState {
  /// IDs of unlocked trophies, in the order they were unlocked.
  final List<String> unlocked;

  /// IDs newly unlocked this session (not yet shown to the user).
  final List<String> newlyUnlocked;

  const TrophyState({required this.unlocked, required this.newlyUnlocked});

  TrophyState copyWith({List<String>? unlocked, List<String>? newlyUnlocked}) =>
      TrophyState(
        unlocked:       unlocked       ?? this.unlocked,
        newlyUnlocked:  newlyUnlocked  ?? this.newlyUnlocked,
      );
}

const _kUnlockedTrophiesKey = 'trophies_unlocked';

class TrophyNotifier extends StateNotifier<TrophyState> {
  TrophyNotifier(this._ref)
      : super(const TrophyState(unlocked: [], newlyUnlocked: [])) {
    _load();
  }

  final Ref _ref;

  Future<void> _load() async {
    final prefs   = await SharedPreferences.getInstance();
    final stored  = prefs.getStringList(_kUnlockedTrophiesKey) ?? [];
    state = state.copyWith(unlocked: stored);
  }

  Future<void> _persist(List<String> unlocked) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_kUnlockedTrophiesKey, unlocked);
  }

  /// Evaluate all trophies against current app state; unlock new ones.
  Future<void> evaluate() async {
    final vocabState   = _ref.read(vocabularyProvider);
    final grammarState = _ref.read(grammarProvider);
    final genState     = _ref.read(generationProvider);
    final streak       = _ref.read(streakProvider);
    final genCount     = _ref.read(generationsCountProvider);
    final focusCount   = _ref.read(focusPracticesCountProvider);

    final words      = vocabState.allWords;
    final masteredWs = words.where((w) => w.masteryLevel >= 5).length;
    final reviewedWs = words.where((w) => w.timesSeen >= 5).toList();
    final avgAccuracy = reviewedWs.isEmpty
        ? 0.0
        : reviewedWs.fold<double>(0, (s, w) => s + w.accuracy) /
            reviewedWs.length;

    final themesUsed     = grammarState.progress.length;
    final themesMastered = grammarState.progress.values
        .where((p) => p.masteryLevel >= 3)
        .length;
    final n5Themes       = grammarState.themes
        .where((t) => t.jlptLevel == 'N5')
        .toList();
    final n5AllMastered  = n5Themes.isNotEmpty &&
        n5Themes.every((t) =>
            (grammarState.progress[t.id]?.masteryLevel ?? 0) >= 3);

    final conditions = <String, bool>{
      'first_word':          words.isNotEmpty,
      'first_generation':    genState.history.isNotEmpty || genCount > 0,
      'first_theme':         grammarState.selected != null || themesUsed > 0,
      'vocab_10':            words.length >= 10,
      'vocab_50':            words.length >= 50,
      'vocab_100':           words.length >= 100,
      'vocab_500':           words.length >= 500,
      'accuracy_80':         reviewedWs.length >= 20 && avgAccuracy >= 0.80,
      'mastered_10':         masteredWs >= 10,
      'mastered_50':         masteredWs >= 50,
      'generations_10':      genCount >= 10,
      'generations_50':      genCount >= 50,
      'generations_200':     genCount >= 200,
      'themes_5':            themesUsed >= 5,
      'themes_mastered_5':   themesMastered >= 5,
      'n5_complete':         n5AllMastered,
      'streak_3':            streak.currentStreak >= 3 || streak.longestStreak >= 3,
      'streak_7':            streak.currentStreak >= 7 || streak.longestStreak >= 7,
      'streak_30':           streak.currentStreak >= 30 || streak.longestStreak >= 30,
      'focus_practice_10':   focusCount >= 10,
    };

    final previously  = List<String>.from(state.unlocked);
    final justUnlocked = <String>[];

    for (final entry in conditions.entries) {
      if (entry.value && !previously.contains(entry.key)) {
        previously.add(entry.key);
        justUnlocked.add(entry.key);
      }
    }

    if (justUnlocked.isNotEmpty) {
      await _persist(previously);
      state = state.copyWith(
        unlocked:      previously,
        newlyUnlocked: justUnlocked,
      );
    }
  }

  /// Mark newly-unlocked trophies as seen (clears the notification queue).
  void clearNewlyUnlocked() {
    state = state.copyWith(newlyUnlocked: []);
  }
}

final trophyProvider =
    StateNotifierProvider<TrophyNotifier, TrophyState>(
        (ref) => TrophyNotifier(ref));
