import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/auth_provider.dart';
import 'screens/auth/login_screen.dart';
import 'screens/learn/learn_screen.dart';

void main() {
  runApp(const ProviderScope(child: GlottaApp()));
}

class GlottaApp extends StatelessWidget {
  const GlottaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Glotta',
      debugShowCheckedModeBanner: false,
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

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  static const List<Widget> _pages = <Widget>[
    LearnScreen(),
    VocabularyPage(),
    ProgressPage(),
    SettingsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) =>
            setState(() => _selectedIndex = index),
        destinations: const [
          NavigationDestination(
            icon:         Icon(Icons.auto_stories_outlined),
            selectedIcon: Icon(Icons.auto_stories),
            label:        '学習',
          ),
          NavigationDestination(
            icon:         Icon(Icons.book_outlined),
            selectedIcon: Icon(Icons.book),
            label:        '単語',
          ),
          NavigationDestination(
            icon:         Icon(Icons.insights_outlined),
            selectedIcon: Icon(Icons.insights),
            label:        '進捗',
          ),
          NavigationDestination(
            icon:         Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label:        '設定',
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// Vocabulary Page — placeholder
// ============================================================================

class VocabularyPage extends StatelessWidget {
  const VocabularyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('単語帳 - Vocabulary')),
      body: const Center(child: Text('Vocabulary management coming soon…')),
    );
  }
}

// ============================================================================
// Progress Page — placeholder
// ============================================================================

class ProgressPage extends StatelessWidget {
  const ProgressPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('進捗 - Progress')),
      body: const Center(child: Text('Progress tracking coming soon…')),
    );
  }
}

// ============================================================================
// Settings Page — shows user profile, allows JLPT level / mode change, logout
// ============================================================================

class SettingsPage extends ConsumerWidget {
  const SettingsPage({super.key});

  static const _levels = ['N5', 'N4', 'N3', 'N2', 'N1'];
  static const _modes  = ['hard', 'soft', 'adaptive'];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final auth   = ref.watch(authProvider);
    final user   = auth.user;
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('設定 - Settings')),
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

          // ── JLPT Level ────────────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.school_outlined),
            title:    const Text('JLPT Level'),
            subtitle: Text(user?.jlptLevel ?? 'N5'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickOption(
              context, ref,
              title:    'JLPT Level',
              options:  _levels,
              current:  user?.jlptLevel ?? 'N5',
              onSelect: (v) => ref.read(authProvider.notifier)
                  .updateProfile(jlptLevel: v),
            ),
          ),

          // ── Constraint Mode ───────────────────────────────────────────
          ListTile(
            leading:  const Icon(Icons.tune_outlined),
            title:    const Text('Constraint Mode'),
            subtitle: Text(user?.constraintMode ?? 'soft'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _pickOption(
              context, ref,
              title:    'Constraint Mode',
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
            title:   const Text('About'),
            onTap:   () => showAboutDialog(
              context:            context,
              applicationName:    'Glotta',
              applicationVersion: '0.2.0',
              applicationIcon:    const Icon(Icons.auto_stories, size: 48),
              children: const [
                Text(
                  'Language learning with constrained LLM generation.\n\n'
                  'Learn Japanese by reading AI-generated text built '
                  'exclusively from your personal vocabulary.',
                ),
              ],
            ),
          ),
          const Divider(),

          // ── Logout ────────────────────────────────────────────────────
          ListTile(
            leading: Icon(Icons.logout, color: colors.error),
            title:   Text('Sign out', style: TextStyle(color: colors.error)),
            onTap:   () async {
              await ref.read(authProvider.notifier).logout();
            },
          ),
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
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
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
