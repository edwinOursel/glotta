import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'l10n/app_localizations.dart';
import 'models/trophy.dart';
import 'providers/auth_provider.dart';
import 'providers/gamification_provider.dart';
import 'providers/generation_provider.dart' show generationProvider;
import 'providers/locale_provider.dart'; // also exports navStyleProvider
import 'providers/vocabulary_provider.dart'
    show focusWordProvider, vocabularyProvider;
import 'screens/auth/login_screen.dart';
import 'screens/learn/learn_screen.dart';
import 'screens/progress/progress_screen.dart';
import 'screens/vocabulary/vocabulary_screen.dart';

void main() {
  runApp(const ProviderScope(child: GlottaApp()));
}

class GlottaApp extends ConsumerWidget {
  const GlottaApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final locale = ref.watch(localeProvider);

    return MaterialApp(
      title: 'Glotta',
      debugShowCheckedModeBanner: false,
      // ── i18n ────────────────────────────────────────────────────────────
      locale:             locale,
      supportedLocales:   AppLocalizations.supportedLocales,
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      // ── Themes ──────────────────────────────────────────────────────────
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const AuthGate(),
    );
  }
}

// ============================================================================
// Auth Gate — restores session on startup, routes to Login or main app
// ============================================================================

class AuthGate extends ConsumerStatefulWidget {
  const AuthGate({super.key});

  @override
  ConsumerState<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends ConsumerState<AuthGate> {
  bool _booting = true;

  @override
  void initState() {
    super.initState();
    _restore();
  }

  Future<void> _restore() async {
    await ref.read(authProvider.notifier).restoreSession();
    if (mounted) setState(() => _booting = false);
  }

  @override
  Widget build(BuildContext context) {
    if (_booting) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final isAuthenticated = ref.watch(authProvider).isAuthenticated;
    return isAuthenticated ? const HomePage() : const LoginScreen();
  }
}

// ============================================================================
// Main app shell with bottom navigation
// ============================================================================

class HomePage extends ConsumerStatefulWidget {
  const HomePage({super.key});

  @override
  ConsumerState<HomePage> createState() => _HomePageState();
}

class _HomePageState extends ConsumerState<HomePage> {
  int _selectedIndex = 0;

  void _switchToLearn() => setState(() => _selectedIndex = 0);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Switch to learn tab when a focus word is set from vocabulary
      ref.listenManual(focusWordProvider, (_, next) {
        if (next != null) {
          _switchToLearn();
          ref.read(focusPracticesCountProvider.notifier).increment();
        }
      });

      // Track generation count, streak, and trophy unlocks
      ref.listenManual(generationProvider, (prev, next) {
        if (prev == null) return;
        if (next.history.length > (prev.history.length)) {
          ref.read(generationsCountProvider.notifier).increment();
          ref.read(streakProvider.notifier).recordActivity();
          ref.read(trophyProvider.notifier).evaluate().then((_) {
            _showNewTrophies();
          });
        }
      });

      // Also evaluate trophies when vocabulary changes (word added/deleted/seeded)
      ref.listenManual(vocabularyProvider, (prev, next) {
        if (prev == null) return;
        if (next.allWords.length != prev.allWords.length) {
          ref.read(streakProvider.notifier).recordActivity();
          ref.read(trophyProvider.notifier).evaluate().then((_) {
            _showNewTrophies();
          });
        }
      });
    });
  }

  void _showNewTrophies() {
    final newly = ref.read(trophyProvider).newlyUnlocked;
    if (newly.isEmpty || !mounted) return;
    ref.read(trophyProvider.notifier).clearNewlyUnlocked();

    final l = AppLocalizations.of(context);
    for (final id in newly) {
      final def = kTrophies.firstWhere((t) => t.id == id,
          orElse: () => kTrophies.first);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${def.emoji}  ${l.trophyNewUnlocked}'),
          duration: const Duration(seconds: 3),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final l        = AppLocalizations.of(context);
    final useKanji = ref.watch(navStyleProvider);

    // Nav labels: kanji (漢字) or hiragana (ひらがな), never the UI locale
    final navLearn    = useKanji ? l.navLearn      : 'がくしゅう';
    final navVocab    = useKanji ? l.navVocabulary  : 'たんご';
    final navProgress = useKanji ? l.navProgress    : 'しんちょく';
    final navSettings = useKanji ? l.navSettings    : 'せってい';

    final pages = <Widget>[
      const LearnScreen(),
      VocabularyScreen(onPractiseWord: _switchToLearn),
      const ProgressScreen(),
      const SettingsPage(),
    ];

    return Scaffold(
      body: pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) =>
            setState(() => _selectedIndex = index),
        destinations: [
          NavigationDestination(
            icon:         const Icon(Icons.auto_stories_outlined),
            selectedIcon: const Icon(Icons.auto_stories),
            label:        navLearn,
          ),
          NavigationDestination(
            icon:         const Icon(Icons.book_outlined),
            selectedIcon: const Icon(Icons.book),
            label:        navVocab,
          ),
          NavigationDestination(
            icon:         const Icon(Icons.insights_outlined),
            selectedIcon: const Icon(Icons.insights),
            label:        navProgress,
          ),
          NavigationDestination(
            icon:         const Icon(Icons.settings_outlined),
            selectedIcon: const Icon(Icons.settings),
            label:        navSettings,
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// Settings Page
// ============================================================================

class SettingsPage extends ConsumerWidget {
  const SettingsPage({super.key});

  static const _levels = ['N5', 'N4', 'N3', 'N2', 'N1'];
  static const _modes  = ['hard', 'soft', 'adaptive'];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l         = AppLocalizations.of(context);
    final auth      = ref.watch(authProvider);
    final user      = auth.user;
    final locale    = ref.watch(localeProvider);
    final useKanji  = ref.watch(navStyleProvider);
    final colors    = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: Text(l.settingsTitle)),
      body: ListView(
        children: [
          // ── Profile info ──────────────────────────────────────────────
          if (user != null) ...[
            ListTile(
              leading: CircleAvatar(
                backgroundColor: colors.primaryContainer,
                child: Text(
                  (user.username ?? user.email)[0].toUpperCase(),
                  style: TextStyle(color: colors.onPrimaryContainer),
                ),
              ),
              title:    Text(user.username ?? user.email),
              subtitle: Text(user.email),
            ),
            const Divider(),
          ],

          // ── Language ──────────────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.language_outlined),
            title:    Text(l.settingsLanguage),
            subtitle: Text(locale.languageCode == 'fr' ? 'Français' : 'English'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickLocale(context, ref, locale),
          ),

          // ── Nav label style ───────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.text_fields_outlined),
            title:    Text(l.settingsNavStyle),
            subtitle: Text(useKanji ? l.settingsNavKanji : l.settingsNavHiragana),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickNavStyle(context, ref, useKanji),
          ),

          // ── JLPT Level ────────────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.school_outlined),
            title:    Text(l.settingsJlptLevel),
            subtitle: Text(user?.jlptLevel ?? 'N5'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickOption(
              context, ref,
              title:    l.settingsJlptLevel,
              options:  _levels,
              current:  user?.jlptLevel ?? 'N5',
              onSelect: (v) => ref.read(authProvider.notifier)
                  .updateProfile(jlptLevel: v),
            ),
          ),

          // ── Constraint Mode ───────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.tune_outlined),
            title:    Text(l.settingsConstraintMode),
            subtitle: Text(user?.constraintMode ?? 'soft'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickOption(
              context, ref,
              title:    l.settingsConstraintMode,
              options:  _modes,
              current:  user?.constraintMode ?? 'soft',
              onSelect: (v) => ref.read(authProvider.notifier)
                  .updateProfile(constraintMode: v),
            ),
          ),
          const Divider(),

          // ── About ─────────────────────────────────────────────────────
          ListTile(
            leading: const Icon(Icons.info_outline),
            title:   Text(l.settingsAbout),
            onTap: () => showAboutDialog(
              context:            context,
              applicationName:    'Glotta',
              applicationVersion: '0.2.0',
              applicationIcon:    const Icon(Icons.auto_stories, size: 48),
              children: [Text(l.settingsAboutText)],
            ),
          ),
          const Divider(),

          // ── Logout ────────────────────────────────────────────────────
          ListTile(
            leading: Icon(Icons.logout, color: colors.error),
            title: Text(l.settingsSignOut,
                style: TextStyle(color: colors.error)),
            onTap: () async {
              await ref.read(authProvider.notifier).logout();
            },
          ),
        ],
      ),
    );
  }

  void _pickNavStyle(BuildContext context, WidgetRef ref, bool useKanji) {
    final l = AppLocalizations.of(context);
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(l.settingsNavStyle,
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          ),
          ListTile(
            leading:  const Text('漢', style: TextStyle(fontSize: 20)),
            title:    Text(l.settingsNavKanji),
            subtitle: const Text('学習  単語  進捗  設定'),
            trailing: useKanji
                ? const Icon(Icons.check, color: Colors.indigo)
                : null,
            onTap: () {
              Navigator.pop(context);
              ref.read(navStyleProvider.notifier).setKanji(true);
            },
          ),
          ListTile(
            leading:  const Text('あ', style: TextStyle(fontSize: 20)),
            title:    Text(l.settingsNavHiragana),
            subtitle: const Text('がくしゅう  たんご  しんちょく  せってい'),
            trailing: !useKanji
                ? const Icon(Icons.check, color: Colors.indigo)
                : null,
            onTap: () {
              Navigator.pop(context);
              ref.read(navStyleProvider.notifier).setKanji(false);
            },
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }

  void _pickLocale(BuildContext context, WidgetRef ref, Locale current) {
    final l = AppLocalizations.of(context);
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(l.settingsLanguage,
                style:
                    const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          ),
          ListTile(
            leading:  const Text('🇬🇧', style: TextStyle(fontSize: 20)),
            title:    const Text('English'),
            trailing: current.languageCode == 'en'
                ? const Icon(Icons.check, color: Colors.indigo)
                : null,
            onTap: () {
              Navigator.pop(context);
              ref
                  .read(localeProvider.notifier)
                  .setLocale(const Locale('en'));
            },
          ),
          ListTile(
            leading:  const Text('🇫🇷', style: TextStyle(fontSize: 20)),
            title:    const Text('Français'),
            trailing: current.languageCode == 'fr'
                ? const Icon(Icons.check, color: Colors.indigo)
                : null,
            onTap: () {
              Navigator.pop(context);
              ref
                  .read(localeProvider.notifier)
                  .setLocale(const Locale('fr'));
            },
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }

  void _pickOption(
    BuildContext context,
    WidgetRef ref, {
    required String title,
    required List<String> options,
    required String current,
    required Future<void> Function(String) onSelect,
  }) {
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(title,
                style:
                    const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          ),
          ...options.map((opt) => ListTile(
                title: Text(opt),
                trailing: opt == current
                    ? const Icon(Icons.check, color: Colors.indigo)
                    : null,
                onTap: () {
                  Navigator.pop(context);
                  onSelect(opt);
                },
              )),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
