import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/generation_provider.dart';

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

  void _handleGenerate() {
    final prompt = _promptController.text;
    ref.read(generationProvider.notifier).generateText(prompt: prompt);
  }

  @override
  Widget build(BuildContext context) {
    final generationState = ref.watch(generationProvider);
    final settings = ref.watch(settingsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('学習 - Learn'),
        actions: [
          // Constraint mode badge
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
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
