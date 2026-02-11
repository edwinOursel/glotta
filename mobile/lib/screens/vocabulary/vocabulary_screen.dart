import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../models/vocabulary_item.dart';
import '../../providers/vocabulary_provider.dart';

// ── Level colours (match grammar theme palette) ───────────────────────────────

const _levelColors = <String, Color>{
  'N5': Color(0xFF4CAF50),
  'N4': Color(0xFF2196F3),
  'N3': Color(0xFFFF9800),
  'N2': Color(0xFFE91E63),
  'N1': Color(0xFF9C27B0),
};

Color _levelColor(String? level) =>
    _levelColors[level] ?? const Color(0xFF9E9E9E);

// ── Screen ────────────────────────────────────────────────────────────────────

class VocabularyScreen extends ConsumerStatefulWidget {
  /// Callback called when user wants to practise a word (switches to LearnTab).
  final VoidCallback? onPractiseWord;

  const VocabularyScreen({super.key, this.onPractiseWord});

  @override
  ConsumerState<VocabularyScreen> createState() => _VocabularyScreenState();
}

class _VocabularyScreenState extends ConsumerState<VocabularyScreen> {
  final _searchController = TextEditingController();
  bool  _searching        = false;
  String _query           = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  List<VocabularyItem> _filtered(List<VocabularyItem> words) {
    if (_query.isEmpty) return words;
    final q = _query.toLowerCase();
    return words.where((w) =>
        w.word.contains(_query) ||
        (w.reading?.toLowerCase().contains(q) ?? false) ||
        (w.meaning?.toLowerCase().contains(q)  ?? false)).toList();
  }

  @override
  Widget build(BuildContext context) {
    final l          = AppLocalizations.of(context);
    final vocabState = ref.watch(vocabularyProvider);
    final filtered   = _filtered(vocabState.words);
    final due        = vocabState.dueWords.length;

    return Scaffold(
      appBar: AppBar(
        title: _searching
            ? TextField(
                controller:  _searchController,
                autofocus:   true,
                decoration:  InputDecoration(
                  hintText: l.vocabSearch,
                  border:   InputBorder.none,
                ),
                onChanged: (v) => setState(() => _query = v),
              )
            : Text(l.vocabTitle),
        actions: [
          // Search toggle
          IconButton(
            icon: Icon(_searching ? Icons.close : Icons.search),
            onPressed: () => setState(() {
              _searching = !_searching;
              if (!_searching) {
                _searchController.clear();
                _query = '';
              }
            }),
          ),
          // Seed JLPT menu
          PopupMenuButton<String>(
            icon: const Icon(Icons.download_outlined),
            tooltip: l.vocabImportTooltip,
            onSelected: (level) => _confirmSeed(level),
            itemBuilder: (_) => ['N5', 'N4', 'N3', 'N2', 'N1']
                .map((level) => PopupMenuItem(
                      value: level,
                      child: Text(l.vocabImportLevel(level)),
                    ))
                .toList(),
          ),
        ],
      ),

      body: Column(
        children: [
          // ── Error banner ────────────────────────────────────────────────
          if (vocabState.error != null)
            MaterialBanner(
              content: Text(vocabState.error!),
              actions: [
                TextButton(
                  onPressed: () =>
                      ref.read(vocabularyProvider.notifier).clearError(),
                  child: Text(l.learnDismiss),
                ),
              ],
              backgroundColor:
                  Theme.of(context).colorScheme.errorContainer,
            ),

          // ── Level filter chips ──────────────────────────────────────────
          _LevelFilter(
            selected: vocabState.selectedLevel,
            onSelect: (lvl) =>
                ref.read(vocabularyProvider.notifier).filterByLevel(lvl),
          ),

          // ── Summary bar ─────────────────────────────────────────────────
          _SummaryBar(
            total: vocabState.words.length,
            due:   due,
          ),

          // ── Word list ───────────────────────────────────────────────────
          Expanded(
            child: vocabState.isLoading
                ? const Center(child: CircularProgressIndicator())
                : filtered.isEmpty
                    ? _buildEmpty(vocabState.words.isEmpty, l)
                    : RefreshIndicator(
                        onRefresh: () =>
                            ref.read(vocabularyProvider.notifier).load(),
                        child: ListView.builder(
                          padding: const EdgeInsets.only(
                              left: 16, right: 16, top: 4, bottom: 96),
                          itemCount: filtered.length,
                          itemBuilder: (_, i) => _WordCard(
                            word:         filtered[i],
                            onDelete:     () => _confirmDelete(filtered[i]),
                            onPractise:   () => _practise(filtered[i]),
                          ),
                        ),
                      ),
          ),
        ],
      ),

      floatingActionButton: FloatingActionButton.extended(
        icon:    const Icon(Icons.add),
        label:   Text(l.vocabAddFab),
        onPressed: () => _showAddSheet(),
      ),
    );
  }

  Widget _buildEmpty(bool noWords, AppLocalizations l) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.book_outlined,
              size: 72, color: Theme.of(context).colorScheme.outlineVariant),
          const SizedBox(height: 16),
          Text(
            noWords ? l.vocabNoWords : l.vocabNoResults,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          if (noWords)
            Text(
              l.vocabEmptyHint,
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: Colors.grey),
              textAlign: TextAlign.center,
            ),
        ],
      ),
    );
  }

  // ── Add word bottom sheet ──────────────────────────────────────────────────

  void _showAddSheet() {
    showModalBottomSheet<void>(
      context:            context,
      isScrollControlled: true,
      useSafeArea:        true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => _AddWordSheet(
        onSaved: (item) {
          final l = AppLocalizations.of(context);
          // After saving, offer to practise
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(l.vocabWordAdded(item.word)),
              duration: const Duration(seconds: 4),
              action: SnackBarAction(
                label:     l.vocabPractiseNow,
                onPressed: () => _practise(item),
              ),
            ),
          );
        },
      ),
    );
  }

  // ── Practise a word ────────────────────────────────────────────────────────

  void _practise(VocabularyItem item) {
    ref.read(focusWordProvider.notifier).state = item;
    widget.onPractiseWord?.call();
  }

  // ── Confirm delete ─────────────────────────────────────────────────────────

  Future<void> _confirmDelete(VocabularyItem item) async {
    final l  = AppLocalizations.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title:   Text(l.vocabDeleteTitle),
        content: Text(l.vocabDeleteContent(item.word)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(l.vocabCancel)),
          FilledButton(
              style: FilledButton.styleFrom(
                  backgroundColor:
                      Theme.of(context).colorScheme.error),
              onPressed: () => Navigator.pop(context, true),
              child: Text(l.vocabDeleteButton)),
        ],
      ),
    );
    if (ok == true) {
      await ref.read(vocabularyProvider.notifier).deleteWord(item.id);
    }
  }

  // ── Confirm seed ───────────────────────────────────────────────────────────

  Future<void> _confirmSeed(String level) async {
    final l  = AppLocalizations.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title:   Text(l.vocabImportConfirmTitle(level)),
        content: Text(l.vocabImportConfirmContent(level)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(l.vocabCancel)),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: Text(l.vocabImportButton)),
        ],
      ),
    );
    if (ok == true) {
      await ref.read(vocabularyProvider.notifier).seedLevel(level);
    }
  }
}

// ── Level filter row ──────────────────────────────────────────────────────────

class _LevelFilter extends StatelessWidget {
  const _LevelFilter({required this.selected, required this.onSelect});

  final String? selected;
  final void Function(String?) onSelect;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    const levels = ['N5', 'N4', 'N3', 'N2', 'N1'];
    return SizedBox(
      height: 48,
      child: ListView(
        padding:         const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        children: [
          Padding(
            padding: const EdgeInsets.only(right: 8, top: 8, bottom: 8),
            child: FilterChip(
              label:      Text(l.vocabLevelAll),
              selected:   selected == null,
              onSelected: (_) => onSelect(null),
            ),
          ),
          ...levels.map((lvl) => Padding(
                padding: const EdgeInsets.only(right: 8, top: 8, bottom: 8),
                child: FilterChip(
                  label:    Text(lvl),
                  selected: selected == lvl,
                  selectedColor: _levelColor(lvl).withOpacity(0.25),
                  onSelected: (_) => onSelect(selected == lvl ? null : lvl),
                ),
              )),
        ],
      ),
    );
  }
}

// ── Summary bar ───────────────────────────────────────────────────────────────

class _SummaryBar extends StatelessWidget {
  const _SummaryBar({required this.total, required this.due});

  final int total;
  final int due;

  @override
  Widget build(BuildContext context) {
    final l      = AppLocalizations.of(context);
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color:   colors.surfaceContainerLow,
      child: Row(children: [
        _Stat(label: l.vocabStatsWords, value: '$total', color: colors.primary),
        const SizedBox(width: 20),
        _Stat(
          label: l.vocabStatsDue,
          value: '$due',
          color: due > 0 ? const Color(0xFFFF9800) : colors.outline,
        ),
      ]),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value, required this.color});

  final String label;
  final String value;
  final Color  color;

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Text(value,
              style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize:   18,
                  color:      color)),
          const SizedBox(width: 4),
          Text(label,
              style: TextStyle(
                  fontSize: 12,
                  color:    Theme.of(context).colorScheme.onSurfaceVariant)),
        ],
      );
}

// ── Word card ─────────────────────────────────────────────────────────────────

class _WordCard extends StatelessWidget {
  const _WordCard({
    required this.word,
    required this.onDelete,
    required this.onPractise,
  });

  final VocabularyItem word;
  final VoidCallback   onDelete;
  final VoidCallback   onPractise;

  @override
  Widget build(BuildContext context) {
    final theme   = Theme.of(context);
    final lcolor  = _levelColor(word.jlptLevel);
    final isDue   = word.isDueForReview;

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onLongPress:  () => _showOptions(context),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Word + reading + meaning ──────────────────────────────
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Text(
                        word.word,
                        style: const TextStyle(
                          fontSize:   20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (word.reading != null) ...[
                        const SizedBox(width: 8),
                        Text(
                          word.reading!,
                          style: TextStyle(
                            fontSize: 13,
                            color:    Colors.grey.shade600,
                          ),
                        ),
                      ],
                    ]),
                    if (word.meaning != null)
                      Text(
                        word.meaning!,
                        style: TextStyle(
                          fontSize: 13,
                          color:    theme.colorScheme.onSurfaceVariant,
                        ),
                      ),

                    const SizedBox(height: 8),

                    // ── Mastery + stats ─────────────────────────────────
                    Row(children: [
                      // Mastery dots (0-5)
                      ...List.generate(5, (i) => Container(
                            width:  8, height: 8,
                            margin: const EdgeInsets.only(right: 3),
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: i < word.masteryLevel
                                  ? Colors.amber.shade600
                                  : Colors.grey.shade300,
                            ),
                          )),
                      const SizedBox(width: 8),
                      if (word.timesSeen > 0)
                        Text(
                          '${(word.accuracy * 100).round()}%',
                          style: TextStyle(
                            fontSize: 11,
                            color:    Colors.grey.shade500,
                          ),
                        ),
                    ]),
                  ],
                ),
              ),

              // ── Right column ──────────────────────────────────────────
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  // JLPT level badge
                  if (word.jlptLevel != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 7, vertical: 2),
                      decoration: BoxDecoration(
                        color:        lcolor.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        word.jlptLevel!,
                        style: TextStyle(
                          color:      lcolor,
                          fontSize:   10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  const SizedBox(height: 6),
                  // Due badge
                  if (isDue)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 7, vertical: 2),
                      decoration: BoxDecoration(
                        color:        const Color(0xFFFF9800).withOpacity(0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        AppLocalizations.of(context).vocabBadgeDue,
                        style: const TextStyle(
                          color:      Color(0xFFE65100),
                          fontSize:   10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  const SizedBox(height: 8),
                  // Practice button
                  GestureDetector(
                    onTap: onPractise,
                    child: Icon(
                      Icons.play_circle_outline,
                      color: theme.colorScheme.primary,
                      size:  24,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showOptions(BuildContext context) {
    final l = AppLocalizations.of(context);
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            leading:  const Icon(Icons.play_circle_outline),
            title:    Text(l.vocabPractiseWord(word.word)),
            onTap: () { Navigator.pop(context); onPractise(); },
          ),
          ListTile(
            leading:  const Icon(Icons.delete_outline, color: Colors.red),
            title:    Text(l.vocabDeleteButton,
                style: const TextStyle(color: Colors.red)),
            onTap: () { Navigator.pop(context); onDelete(); },
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

// ── Add word bottom sheet ─────────────────────────────────────────────────────

class _AddWordSheet extends ConsumerStatefulWidget {
  const _AddWordSheet({required this.onSaved});

  final void Function(VocabularyItem) onSaved;

  @override
  ConsumerState<_AddWordSheet> createState() => _AddWordSheetState();
}

class _AddWordSheetState extends ConsumerState<_AddWordSheet> {
  final _formKey    = GlobalKey<FormState>();
  final _wordCtrl   = TextEditingController();
  final _readCtrl   = TextEditingController();
  final _meaningCtrl = TextEditingController();
  String? _level;
  bool    _saving   = false;

  @override
  void dispose() {
    _wordCtrl.dispose();
    _readCtrl.dispose();
    _meaningCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!(_formKey.currentState?.validate() ?? false)) return;
    setState(() => _saving = true);

    final item = await ref.read(vocabularyProvider.notifier).addWord(
          word:      _wordCtrl.text.trim(),
          reading:   _readCtrl.text.trim().isEmpty
              ? null
              : _readCtrl.text.trim(),
          meaning:   _meaningCtrl.text.trim().isEmpty
              ? null
              : _meaningCtrl.text.trim(),
          jlptLevel: _level,
        );

    setState(() => _saving = false);

    if (!mounted) return;
    Navigator.pop(context);
    if (item != null) widget.onSaved(item);
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Padding(
      padding: EdgeInsets.only(
        left:   24,
        right:  24,
        top:    20,
        bottom: MediaQuery.of(context).viewInsets.bottom + 24,
      ),
      child: Form(
        key: _formKey,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle
            Center(
              child: Container(
                width: 40, height: 4,
                margin: const EdgeInsets.only(bottom: 20),
                decoration: BoxDecoration(
                  color:        Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),

            Text(l.vocabAddSheetTitle,
                style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 20),

            // Word (required)
            TextFormField(
              controller:    _wordCtrl,
              decoration:    InputDecoration(
                labelText: l.vocabWordField,
                hintText:  l.vocabWordHint,
                border:    const OutlineInputBorder(),
              ),
              validator: (v) =>
                  v == null || v.trim().isEmpty ? l.validRequired : null,
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 14),

            // Reading (optional)
            TextFormField(
              controller:  _readCtrl,
              decoration:  InputDecoration(
                labelText: l.vocabReadingField,
                hintText:  l.vocabReadingHint,
                border:    const OutlineInputBorder(),
              ),
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 14),

            // Meaning (optional)
            TextFormField(
              controller:  _meaningCtrl,
              decoration:  InputDecoration(
                labelText: l.vocabMeaningField,
                hintText:  l.vocabMeaningHint,
                border:    const OutlineInputBorder(),
              ),
              textInputAction: TextInputAction.done,
              onFieldSubmitted: (_) => _save(),
            ),
            const SizedBox(height: 14),

            // JLPT Level picker
            DropdownButtonFormField<String>(
              value:       _level,
              decoration:  InputDecoration(
                labelText: l.vocabLevelField,
                border:    const OutlineInputBorder(),
              ),
              hint:        Text(l.vocabLevelOptional),
              items: ['N5', 'N4', 'N3', 'N2', 'N1']
                  .map((lvl) => DropdownMenuItem(value: lvl, child: Text(lvl)))
                  .toList(),
              onChanged: (v) => setState(() => _level = v),
            ),
            const SizedBox(height: 20),

            // Save button
            FilledButton.icon(
              onPressed: _saving ? null : _save,
              icon: _saving
                  ? const SizedBox(
                      width: 18, height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.check),
              label: Text(l.vocabSaveButton),
              style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(50)),
            ),
          ],
        ),
      ),
    );
  }
}
