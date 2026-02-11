import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _kLocaleKey = 'app_locale';

class LocaleNotifier extends StateNotifier<Locale> {
  LocaleNotifier() : super(const Locale('en')) {
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final code  = prefs.getString(_kLocaleKey);
    if (code != null && ['en', 'fr'].contains(code)) {
      state = Locale(code);
    }
  }

  Future<void> setLocale(Locale locale) async {
    state = locale;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kLocaleKey, locale.languageCode);
  }
}

final localeProvider = StateNotifierProvider<LocaleNotifier, Locale>(
  (_) => LocaleNotifier(),
);

// ── Nav label style ───────────────────────────────────────────────────────────
// true  = kanji  (学習 / 単語 / 進捗 / 設定)
// false = hiragana (がくしゅう / たんご / しんちょく / せってい)

const _kNavKanjiKey = 'nav_kanji';

class NavStyleNotifier extends StateNotifier<bool> {
  NavStyleNotifier() : super(true) {
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    state = prefs.getBool(_kNavKanjiKey) ?? true;
  }

  Future<void> setKanji(bool value) async {
    state = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kNavKanjiKey, value);
  }
}

final navStyleProvider = StateNotifierProvider<NavStyleNotifier, bool>(
  (_) => NavStyleNotifier(),
);
