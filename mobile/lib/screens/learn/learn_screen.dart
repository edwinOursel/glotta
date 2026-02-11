import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../models/grammar_theme.dart';
import '../../providers/generation_provider.dart';
import '../../providers/grammar_provider.dart';
import '../../providers/vocabulary_provider.dart';
import '../grammar/grammar_selection_screen.dart';

class LearnScreen extends ConsumerStatefulWidget {
  const LearnScreen({super.key});

  @override
  ConsumerState<LearnScreen> createState() => _LearnScreenState();
}

class _LearnScreenState extends ConsumerState<LearnScreen> {
  final _promptController = TextEditingController();
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // Called once per frame; detects an incoming focus word from VocabularyScreen
  // and pre-fills the prompt with it.
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final focusWord = ref.read(focusWordProvider);
    if (focusWord != null && _promptController.text.isEmpty) {
      _promptController.text = focusWord.word;
    }
  }

  void _handleGenerate() {
    final prompt = _promptController.text;
    ref.read(generationProvider.notifier).generateText(prompt: prompt);
    // Clear focus word after first generation so it doesn't re-prime next time
    ref.read(focusWordProvider.notifier).state = null;
  }

  @override
  Widget build(BuildContext context) {
    final generationState = ref.watch(generationProvider);
    final settings        = ref.watch(settingsProvider);
    final grammarState    = ref.watch(grammarProvider);
    final selectedTheme   = grammarState.selected;
    final focusWord       = ref.watch(focusWordProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('学習 - Learn'),
        actions: [
          // Grammar theme selector chip
          _GrammarThemeChip(
            selectedTheme: selectedTheme,
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute<void>(
                builder: (_) => const GrammarSelectionScreen(),
              ),
            ),
          ),
          // Constraint mode badge
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: Chip(
              avatar: Icon(
                settings.useConstraints ? Icons.lock : Icons.lock_open,
                size: 16,
              ),
              label: Text(
                settings.useConstraints ? settings.constraintMode.toUpperCase() : 'FREE',
                style: const TextStyle(fontSize: 12),
              ),
            ),
          ),
          // Toggle constraints
          IconButton(
            icon: Icon(
              settings.useConstraints ? Icons.toggle_on : Icons.toggle_off,
            ),
            onPressed: () {
              ref.read(settingsProvider.notifier).toggleConstraints();
            },
            tooltip: 'Toggle vocabulary constraints',
          ),
          // Clear history
          if (generationState.history.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: () {
                ref.read(generationProvider.notifier).clearHistory();
              },
              tooltip: 'Clear history',
            ),
        ],
      ),
      body: Column(
        children: [
          // Error banner
          if (generationState.error != null)
            MaterialBanner(
              content: Text(generationState.error!),
              actions: [
                TextButton(
                  onPressed: () {
                    ref.read(generationProvider.notifier).clearError();
                  },
                  child: const Text('Dismiss'),
                ),
              ],
              backgroundColor: Theme.of(context).colorScheme.errorContainer,
            ),

          // Active grammar theme banner
          if (selectedTheme != null)
            _ActiveGrammarBanner(theme: selectedTheme),

          // Focus word banner (set from VocabularyScreen)
          if (focusWord != null)
            _FocusWordBanner(
              word:    focusWord.word,
              reading: focusWord.reading,
              meaning: focusWord.meaning,
              onDismiss: () =>
                  ref.read(focusWordProvider.notifier).state = null,
            ),

          // History list
          Expanded(
            child: generationState.history.isEmpty
                ? _buildEmptyState()
                : _buildHistoryList(generationState),
          ),

          // Input section
          _buildInputSection(generationState.isLoading),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.auto_stories_outlined,
            size: 100,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 16),
          Text(
            'Start learning!',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Enter a Japanese prompt below to generate text',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Colors.grey[600],
                ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryList(GenerationState state) {
    return ListView.builder(
      controller: _scrollController,
      reverse: true,
      padding: const EdgeInsets.all(16),
      itemCount: state.history.length,
      itemBuilder: (context, index) {
        final generated = state.history[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 16),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header with prompt and mode
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Prompt: ${generated.prompt}',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                              color: Colors.grey[700],
                            ),
                      ),
                    ),
                    if (generated.constraintMode != null)
                      Chip(
                        label: Text(
                          generated.constraintMode!.toUpperCase(),
                          style: const TextStyle(fontSize: 10),
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                  ],
                ),
                const Divider(),
                const SizedBox(height: 8),
                // Generated text
                SelectableText(
                  generated.text,
                  style: const TextStyle(
                    fontSize: 18,
                    height: 1.8,
                  ),
                ),
                const SizedBox(height: 8),
                // Timestamp
                Text(
                  _formatTimestamp(generated.timestamp),
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[500],
                      ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildInputSection(bool isLoading) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(16),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _promptController,
                enabled: !isLoading,
                decoration: const InputDecoration(
                  hintText: 'Enter a Japanese prompt... (e.g., 私は)',
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                ),
                onSubmitted: (_) => _handleGenerate(),
                textInputAction: TextInputAction.send,
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: isLoading ? null : _handleGenerate,
              icon: isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.send),
              label: const Text('Generate'),
            ),
          ],
        ),
      ),
    );
  }

  String _formatTimestamp(DateTime timestamp) {
    final now = DateTime.now();
    final difference = now.difference(timestamp);

    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inHours < 1) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inDays < 1) {
      return '${difference.inHours}h ago';
    } else {
      return '${timestamp.day}/${timestamp.month} ${timestamp.hour}:${timestamp.minute.toString().padLeft(2, '0')}';
    }
  }
}

// ── Focus word banner ─────────────────────────────────────────────────────────

class _FocusWordBanner extends StatelessWidget {
  const _FocusWordBanner({
    required this.word,
    this.reading,
    this.meaning,
    required this.onDismiss,
  });

  final String       word;
  final String?      reading;
  final String?      meaning;
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Material(
      color: colors.tertiaryContainer.withOpacity(0.6),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            Icon(Icons.local_library_outlined,
                size: 16, color: colors.onTertiaryContainer),
            const SizedBox(width: 8),
            Expanded(
              child: RichText(
                text: TextSpan(
                  style: TextStyle(
                      fontSize: 13, color: colors.onTertiaryContainer),
                  children: [
                    const TextSpan(text: 'Practising  '),
                    TextSpan(
                        text: word,
                        style: const TextStyle(fontWeight: FontWeight.bold)),
                    if (reading != null)
                      TextSpan(
                          text: '（$reading）',
                          style: const TextStyle(fontStyle: FontStyle.italic)),
                    if (meaning != null)
                      TextSpan(
                          text: '  $meaning',
                          style: TextStyle(
                              color: colors.onTertiaryContainer
                                  .withOpacity(0.7))),
                  ],
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            GestureDetector(
              onTap: onDismiss,
              child: Icon(Icons.close,
                  size: 16, color: colors.onTertiaryContainer),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Grammar theme chip shown in the AppBar ────────────────────────────────────

class _GrammarThemeChip extends StatelessWidget {
  const _GrammarThemeChip({
    required this.selectedTheme,
    required this.onTap,
  });

  final GrammarTheme? selectedTheme;
  final VoidCallback  onTap;

  @override
  Widget build(BuildContext context) {
    final catColor = selectedTheme != null
        ? Color(kCategories[selectedTheme!.category]?.colorValue ?? 0xFF9E9E9E)
        : null;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      child: ActionChip(
        avatar: Icon(
          Icons.menu_book_outlined,
          size: 16,
          color: catColor ?? Theme.of(context).colorScheme.onSurfaceVariant,
        ),
        label: Text(
          selectedTheme != null ? selectedTheme!.nameJp : '文法',
          style: TextStyle(
            fontSize:   12,
            color:      catColor,
            fontWeight: selectedTheme != null ? FontWeight.bold : null,
          ),
        ),
        side: catColor != null
            ? BorderSide(color: catColor.withOpacity(0.5))
            : null,
        onPressed: onTap,
        tooltip: selectedTheme != null
            ? '${selectedTheme!.nameEn} — tap to change'
            : 'Select grammar theme',
      ),
    );
  }
}

// ── Active grammar theme contextual banner ────────────────────────────────────

class _ActiveGrammarBanner extends ConsumerWidget {
  const _ActiveGrammarBanner({required this.theme});

  final GrammarTheme theme;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final catColor = Color(kCategories[theme.category]?.colorValue ?? 0xFF9E9E9E);

    return Material(
      color: catColor.withOpacity(0.08),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            Icon(Icons.menu_book_outlined, size: 16, color: catColor),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                '${theme.nameJp}  ·  ${theme.nameEn}',
                style: TextStyle(
                  fontSize:   13,
                  color:      catColor,
                  fontWeight: FontWeight.w600,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            GestureDetector(
              onTap: () =>
                  ref.read(grammarProvider.notifier).selectTheme(null),
              child: Icon(Icons.close, size: 16, color: catColor),
            ),
          ],
        ),
      ),
    );
  }
}
