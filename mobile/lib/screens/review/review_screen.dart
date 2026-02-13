import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../models/vocabulary_item.dart';
import '../../providers/review_provider.dart';

// ── Entry-point ────────────────────────────────────────────────────────────────

/// Push this screen onto the navigator to start an SRS review session.
class ReviewScreen extends ConsumerWidget {
  const ReviewScreen({super.key});

  static Route<void> route() =>
      MaterialPageRoute<void>(builder: (_) => const ReviewScreen());

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(reviewProvider);
    final l     = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.reviewTitle),
        centerTitle: true,
      ),
      body: switch (state.phase) {
        ReviewPhase.loading => const Center(child: CircularProgressIndicator()),
        ReviewPhase.card    => _CardView(state: state),
        ReviewPhase.done    => _SummaryView(state: state),
      },
    );
  }
}

// ── Card view ──────────────────────────────────────────────────────────────────

class _CardView extends ConsumerStatefulWidget {
  final ReviewSessionState state;
  const _CardView({required this.state});

  @override
  ConsumerState<_CardView> createState() => _CardViewState();
}

class _CardViewState extends ConsumerState<_CardView>
    with SingleTickerProviderStateMixin {
  late AnimationController _flipCtrl;
  late Animation<double>   _flipAnim;
  bool _showAnswer = false;

  @override
  void initState() {
    super.initState();
    _flipCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _flipAnim = Tween<double>(begin: 0, end: math.pi)
        .animate(CurvedAnimation(parent: _flipCtrl, curve: Curves.easeInOut));
  }

  @override
  void didUpdateWidget(_CardView old) {
    super.didUpdateWidget(old);
    // Reset flip animation when a new card appears
    if (old.state.currentIndex != widget.state.currentIndex) {
      _flipCtrl.reset();
      _showAnswer = false;
    }
  }

  @override
  void dispose() {
    _flipCtrl.dispose();
    super.dispose();
  }

  void _flip() {
    if (_showAnswer) return; // can't un-flip
    _flipCtrl.forward();
    setState(() => _showAnswer = true);
  }

  void _rate(int quality) {
    ref.read(reviewProvider.notifier).submitRating(quality);
  }

  @override
  Widget build(BuildContext context) {
    final l     = AppLocalizations.of(context);
    final state = widget.state;
    final word  = state.current;
    if (word == null) return const SizedBox();

    final theme     = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Column(
      children: [
        // ── Progress bar ──────────────────────────────────────────────────
        _ProgressBar(
          current: state.currentIndex,
          total:   state.total,
          label:   l.reviewProgress(state.currentIndex, state.total),
        ),

        // ── Flashcard ─────────────────────────────────────────────────────
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: GestureDetector(
              onTap: _flip,
              child: AnimatedBuilder(
                animation: _flipAnim,
                builder: (context, child) {
                  final angle = _flipAnim.value;
                  final isFront = angle <= math.pi / 2;
                  // When past 90° we render the back face (mirrored)
                  final displayAngle = isFront ? angle : angle - math.pi;
                  return Transform(
                    alignment: Alignment.center,
                    transform: Matrix4.identity()
                      ..setEntry(3, 2, 0.001)
                      ..rotateY(displayAngle),
                    child: isFront
                        ? _CardFace(word: word, showAnswer: false, l: l)
                        : Transform(
                            alignment: Alignment.center,
                            transform: Matrix4.identity()..rotateY(math.pi),
                            child: _CardFace(
                              word: word,
                              showAnswer: true,
                              l: l,
                            ),
                          ),
                  );
                },
              ),
            ),
          ),
        ),

        // ── Rating buttons (only visible after flip) ──────────────────────
        AnimatedOpacity(
          opacity: _showAnswer ? 1.0 : 0.0,
          duration: const Duration(milliseconds: 200),
          child: IgnorePointer(
            ignoring: !_showAnswer,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
              child: Row(
                children: [
                  _RatingButton(
                    label: l.reviewHard,
                    color: colorScheme.error,
                    onPressed: () => _rate(1),
                  ),
                  const SizedBox(width: 12),
                  _RatingButton(
                    label: l.reviewOk,
                    color: colorScheme.tertiary,
                    onPressed: () => _rate(3),
                  ),
                  const SizedBox(width: 12),
                  _RatingButton(
                    label: l.reviewEasy,
                    color: Colors.green,
                    onPressed: () => _rate(5),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ── Card faces ─────────────────────────────────────────────────────────────────

class _CardFace extends StatelessWidget {
  final VocabularyItem word;
  final bool showAnswer;
  final AppLocalizations l;

  const _CardFace({
    required this.word,
    required this.showAnswer,
    required this.l,
  });

  @override
  Widget build(BuildContext context) {
    final theme       = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      color: colorScheme.surfaceContainerHigh,
      child: SizedBox.expand(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: showAnswer ? _backContent(context) : _frontContent(context),
        ),
      ),
    );
  }

  Widget _frontContent(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          word.word,
          style: theme.textTheme.displayMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
          textAlign: TextAlign.center,
        ),
        if (word.reading != null && word.reading!.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(
            word.reading!,
            style: theme.textTheme.headlineSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
        ],
        const Spacer(),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.touch_app_outlined,
                size: 18, color: theme.colorScheme.onSurfaceVariant),
            const SizedBox(width: 6),
            Text(
              l.reviewFlipHint,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _backContent(BuildContext context) {
    final theme       = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Word + reading recap at top
        Center(
          child: Text(
            word.word,
            style: theme.textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        if (word.reading != null && word.reading!.isNotEmpty) ...[
          const SizedBox(height: 4),
          Center(
            child: Text(
              word.reading!,
              style: theme.textTheme.titleMedium?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
        Divider(height: 40, color: colorScheme.outlineVariant),
        // Meaning
        if (word.meaning != null && word.meaning!.isNotEmpty)
          Text(
            word.meaning!,
            style: theme.textTheme.headlineSmall,
          ),
        const SizedBox(height: 20),
        // JLPT + mastery chips
        Wrap(
          spacing: 8,
          children: [
            if (word.jlptLevel != null)
              _Chip(
                label: word.jlptLevel!,
                color: colorScheme.primaryContainer,
                textColor: colorScheme.onPrimaryContainer,
              ),
            _Chip(
              label: '${'★' * word.masteryLevel}${'☆' * (5 - word.masteryLevel)}',
              color: colorScheme.secondaryContainer,
              textColor: colorScheme.onSecondaryContainer,
            ),
          ],
        ),
      ],
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color  color;
  final Color  textColor;
  const _Chip({required this.label, required this.color, required this.textColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color:        color,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: TextStyle(color: textColor, fontSize: 13),
      ),
    );
  }
}

// ── Rating button ──────────────────────────────────────────────────────────────

class _RatingButton extends StatelessWidget {
  final String   label;
  final Color    color;
  final VoidCallback onPressed;

  const _RatingButton({
    required this.label,
    required this.color,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: FilledButton(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          backgroundColor: color,
          minimumSize: const Size.fromHeight(52),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        child: Text(label, style: const TextStyle(fontSize: 16)),
      ),
    );
  }
}

// ── Progress bar ───────────────────────────────────────────────────────────────

class _ProgressBar extends StatelessWidget {
  final int    current;
  final int    total;
  final String label;
  const _ProgressBar({required this.current, required this.total, required this.label});

  @override
  Widget build(BuildContext context) {
    final fraction = total == 0 ? 0.0 : current / total;
    return Column(
      children: [
        LinearProgressIndicator(
          value:            fraction,
          minHeight:        6,
          borderRadius:     BorderRadius.zero,
          backgroundColor:  Theme.of(context).colorScheme.surfaceContainerHighest,
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Text(
                label,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ── Session summary ────────────────────────────────────────────────────────────

class _SummaryView extends ConsumerWidget {
  final ReviewSessionState state;
  const _SummaryView({required this.state});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final theme       = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final reviewed    = state.reviewed;
    final correct     = state.correctCount;
    final pct         = reviewed == 0 ? 0 : (correct * 100 ~/ reviewed);

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Trophy / check icon
            Container(
              width:  96,
              height: 96,
              decoration: BoxDecoration(
                color:  colorScheme.primaryContainer,
                shape:  BoxShape.circle,
              ),
              child: reviewed == 0
                  ? Icon(Icons.check_circle_outline,
                      size: 56, color: colorScheme.onPrimaryContainer)
                  : Center(
                      child: Text(
                        '$pct%',
                        style: theme.textTheme.headlineMedium?.copyWith(
                          color:      colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
            ),
            const SizedBox(height: 24),
            Text(
              reviewed == 0 ? l.reviewNothingDue : l.reviewSessionDone,
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              reviewed == 0
                  ? l.reviewNothingDueHint
                  : l.reviewSessionSummary(reviewed, correct),
              style: theme.textTheme.bodyLarge?.copyWith(
                color: colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 40),
            if (reviewed > 0) ...[
              FilledButton.icon(
                onPressed: () =>
                    ref.read(reviewProvider.notifier).restart(),
                icon:  const Icon(Icons.replay),
                label: Text(l.reviewAgain),
                style: FilledButton.styleFrom(
                  minimumSize: const Size(220, 48),
                ),
              ),
              const SizedBox(height: 12),
            ],
            OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon:  const Icon(Icons.arrow_back),
              label: Text(l.reviewBackToVocab),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size(220, 48),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
