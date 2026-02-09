import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'screens/learn/learn_screen.dart';

void main() {
  runApp(
    const ProviderScope(
      child: GlottaApp(),
    ),
  );
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
      home: const HomePage(),
    );
  }
}

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
        onDestinationSelected: (int index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.auto_stories_outlined),
            selectedIcon: Icon(Icons.auto_stories),
            label: '学習',
          ),
          NavigationDestination(
            icon: Icon(Icons.book_outlined),
            selectedIcon: Icon(Icons.book),
            label: '単語',
          ),
          NavigationDestination(
            icon: Icon(Icons.insights_outlined),
            selectedIcon: Icon(Icons.insights),
            label: '進捗',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: '設定',
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// VOCABULARY PAGE - Placeholder
// ============================================================================

class VocabularyPage extends StatelessWidget {
  const VocabularyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('単語帳 - Vocabulary'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () {
              // TODO: Add new word
            },
          ),
        ],
      ),
      body: const Center(
        child: Text('Vocabulary management coming soon...'),
      ),
    );
  }
}

// ============================================================================
// PROGRESS PAGE - Placeholder
// ============================================================================

class ProgressPage extends StatelessWidget {
  const ProgressPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('進捗 - Progress'),
      ),
      body: const Center(
        child: Text('Progress tracking coming soon...'),
      ),
    );
  }
}

// ============================================================================
// SETTINGS PAGE - Placeholder
// ============================================================================

class SettingsPage extends ConsumerWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('設定 - Settings'),
      ),
      body: ListView(
        children: [
          const ListTile(
            leading: Icon(Icons.language),
            title: Text('Target Language'),
            subtitle: Text('Japanese'),
          ),
          const ListTile(
            leading: Icon(Icons.school),
            title: Text('JLPT Level'),
            subtitle: Text('N5'),
          ),
          const Divider(),
          const ListTile(
            leading: Icon(Icons.tune),
            title: Text('Constraint Mode'),
            subtitle: Text('Hard'),
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.info),
            title: const Text('About'),
            onTap: () {
              showAboutDialog(
                context: context,
                applicationName: 'Glotta',
                applicationVersion: '0.1.0',
                applicationIcon: const Icon(Icons.auto_stories, size: 48),
                children: const [
                  Text(
                    'Language learning with constrained LLM generation.\n\n'
                    'Learn Japanese by reading text generated specifically '
                    'for your vocabulary level.',
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }
}
