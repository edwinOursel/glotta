import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../models/trophy.dart';
import '../../providers/gamification_provider.dart';
import '../../providers/grammar_provider.dart';
import '../../providers/vocabulary_provider.dart';
import '../review/review_screen.dart';

// ── Localized trophy label resolver ──────────────────────────────────────────

String _trophyName(AppLocalizations l, String nameKey) {
  switch (nameKey) {
    case 'trophyFirstWordName':       return l.trophyFirstWordName;
    case 'trophyFirstGenerationName': return l.trophyFirstGenerationName;
    case 'trophyFirstThemeName':      return l.trophyFirstThemeName;
    case 'trophyVocab10Name':         return l.trophyVocab10Name;
    case 'trophyVocab50Name':         return l.trophyVocab50Name;
    case 'trophyVocab100Name':        return l.trophyVocab100Name;
    case 'trophyVocab500Name':        return l.trophyVocab500Name;
    case 'trophyAccuracy80Name':      return l.trophyAccuracy80Name;
    case 'trophyMastered10Name':      return l.trophyMastered10Name;
    case 'trophyMastered50Name':      return l.trophyMastered50Name;
    case 'trophyGenerations10Name':   return l.trophyGenerations10Name;
    case 'trophyGenerations50Name':   return l.trophyGenerations50Name;
    case 'trophyGenerations200Name':  return l.trophyGenerations200Name;
    case 'trophyThemes5Name':         return l.trophyThemes5Name;
    case 'trophyThemesMastered5Name': return l.trophyThemesMastered5Name;
    case 'trophyN5CompleteName':      return l.trophyN5CompleteName;
    case 'trophyStreak3Name':         return l.trophyStreak3Name;
    case 'trophyStreak7Name':         return l.trophyStreak7Name;
    case 'trophyStreak30Name':        return l.trophyStreak30Name;
    case 'trophyFocusPractice10Name': return l.trophyFocusPractice10Name;
    default:                          return nameKey;
  }
}

String _trophyDesc(AppLocalizations l, String descKey) {
  switch (descKey) {
    case 'trophyFirstWordDesc':       return l.trophyFirstWordDesc;
    case 'trophyFirstGenerationDesc': return l.trophyFirstGenerationDesc;
    case 'trophyFirstThemeDesc':      return l.trophyFirstThemeDesc;
    case 'trophyVocab10Desc':         return l.trophyVocab10Desc;
    case 'trophyVocab50Desc':         return l.trophyVocab50Desc;
    case 'trophyVocab100Desc':        return l.trophyVocab100Desc;
    case 'trophyVocab500Desc':        return l.trophyVocab500Desc;
    case 'trophyAccuracy80Desc':      return l.trophyAccuracy80Desc;
    case 'trophyMastered10Desc':      return l.trophyMastered10Desc;
    case 'trophyMastered50Desc':      return l.trophyMastered50Desc;
    case 'trophyGenerations10Desc':   return l.trophyGenerations10Desc;
    case 'trophyGenerations50Desc':   return l.trophyGenerations50Desc;
    case 'trophyGenerations200Desc':  return l.trophyGenerations200Desc;
    case 'trophyThemes5Desc':         return l.trophyThemes5Desc;
    case 'trophyThemesMastered5Desc': return l.trophyThemesMastered5Desc;
    case 'trophyN5CompleteDesc':      return l.trophyN5CompleteDesc;
    case 'trophyStreak3Desc':         return l.trophyStreak3Desc;
    case 'trophyStreak7Desc':         return l.trophyStreak7Desc;
    case 'trophyStreak30Desc':        return l.trophyStreak30Desc;
    case 'trophyFocusPractice10Desc': return l.trophyFocusPractice10Desc;
    default:                          return descKey;
  }
}

// ── Tier colours / labels ─────────────────────────────────────────────────────

const _tierColors = {
  TrophyTier.bronze:   Color(0xFFCD7F32),
  TrophyTier.silver:   Color(0xFFAAAAAA),
  TrophyTier.gold:     Color(0xFFFFD700),
  TrophyTier.platinum: Color(0xFF00E5FF),
};

Color _tierColor(TrophyTier t) => _tierColors[t]!;

// ── Screen ────────────────────────────────────────────────────────────────────

class ProgressScreen extends ConsumerWidget {
  const ProgressScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final vocabState  = ref.watch(vocabularyProvider);
    final grammarState = ref.watch(grammarProvider);
    final streak      = ref.watch(streakProvider);
    final genCount    = ref.watch(generationsCountProvider);
    final trophyState = ref.watch(trophyProvider);

    final totalWords   = vocabState.allWords.length;
    final dueWords     = vocabState.allWords.where((w) => w.isDueForReview).length;
    final sessionsTotal = grammarState.progress.values
        .fold<int>(0, (s, p) => s + p.sessionsCount);

    final unlockedIds = trophyState.unlocked.toSet();

    return Scaffold(
      appBar: AppBar(title: Text(l.progressTitle)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Stats cards ──────────────────────────────────────────────
          _StatsRow(
            stats: [
              _StatCard(
                label: l.progressStatsWords,
                value: '$totalWords',
                icon: Icons.book_outlined,
                color: Theme.of(context).colorScheme.primary,
              ),
              _StatCard(
                label: l.progressStatsDue,
                value: '$dueWords',
                icon: Icons.schedule_outlined,
                color: dueWords > 0
                    ? const Color(0xFFFF9800)
                    : Theme.of(context).colorScheme.outline,
              ),
              _StatCard(
                label: l.progressStatsSessions,
                value: '$sessionsTotal',
                icon: Icons.auto_stories_outlined,
                color: const Color(0xFF9C27B0),
              ),
              _StatCard(
                label: l.progressStatsStreak,
                value: '${streak.currentStreak}',
                icon: Icons.local_fire_department_outlined,
                color: const Color(0xFFFF5722),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // ── Review button (when words are due) ────────────────────────
          if (dueWords > 0) ...[
            _ReviewCard(dueCount: dueWords),
            const SizedBox(height: 20),
          ],

          // ── Streak card ───────────────────────────────────────────────
          _StreakCard(streak: streak),
          const SizedBox(height: 20),

          // ── JLPT vocabulary progress ──────────────────────────────────
          _SectionTitle(l.progressJlptProgress),
          const SizedBox(height: 8),
          _JlptProgressBars(words: vocabState.allWords),
          const SizedBox(height: 20),

          // ── Grammar mastery ───────────────────────────────────────────
          if (grammarState.progress.isNotEmpty) ...[
            _SectionTitle(l.progressGrammarProgress),
            const SizedBox(height: 8),
            _GrammarMasteryBars(grammarState: grammarState),
            const SizedBox(height: 20),
          ],

          // ── Trophies ──────────────────────────────────────────────────
          Row(children: [
            Expanded(child: _SectionTitle(l.progressTrophies)),
            Text(
              l.progressTrophyCount(
                  unlockedIds.length, kTrophies.length),
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant),
            ),
          ]),
          const SizedBox(height: 8),
          _TrophyGrid(
            trophies:    kTrophies,
            unlockedIds: unlockedIds,
          ),
        ],
      ),
    );
  }
}

// ── Stats row ─────────────────────────────────────────────────────────────────

class _StatsRow extends StatelessWidget {
  const _StatsRow({required this.stats});
  final List<_StatCard> stats;

  @override
  Widget build(BuildContext context) => Row(
        children: stats
            .map((s) => Expanded(child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: s,
                )))
            .toList(),
      );
}

class _StatCard extends StatelessWidget {
  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  final String   label;
  final String   value;
  final IconData icon;
  final Color    color;

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
          child: Column(
            children: [
              Icon(icon, color: color, size: 22),
              const SizedBox(height: 6),
              Text(value,
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize:   20,
                      color:      color)),
              const SizedBox(height: 2),
              Text(label,
                  style: Theme.of(context).textTheme.bodySmall,
                  textAlign: TextAlign.center,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis),
            ],
          ),
        ),
      );
}

// ── Streak card ───────────────────────────────────────────────────────────────

class _StreakCard extends StatelessWidget {
  const _StreakCard({required this.streak});
  final StreakState streak;

  @override
  Widget build(BuildContext context) {
    final l      = AppLocalizations.of(context);
    final colors = Theme.of(context).colorScheme;
    const fireColor = Color(0xFFFF5722);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color:        fireColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.local_fire_department,
                color: fireColor, size: 28),
          ),
          const SizedBox(width: 16),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text('${streak.currentStreak}',
                  style: const TextStyle(
                      fontSize:   28,
                      fontWeight: FontWeight.bold,
                      color:      fireColor)),
              const SizedBox(width: 4),
              Text(l.progressStreakDays,
                  style: TextStyle(
                      fontSize: 14,
                      color:    colors.onSurfaceVariant)),
            ]),
            Text('${l.progressLongestStreak}: ${streak.longestStreak}',
                style: TextStyle(
                    fontSize: 12,
                    color:    colors.onSurfaceVariant)),
          ]),
          const Spacer(),
          if (streak.activityToday)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color:        fireColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text('🔥',
                  style: TextStyle(fontSize: 18)),
            ),
        ]),
      ),
    );
  }
}

// ── JLPT progress bars ────────────────────────────────────────────────────────

const _jlptLevelColors = {
  'N5': Color(0xFF4CAF50),
  'N4': Color(0xFF2196F3),
  'N3': Color(0xFFFF9800),
  'N2': Color(0xFFE91E63),
  'N1': Color(0xFF9C27B0),
};

// Approximate total JLPT word counts (for progress bar denominator)
const _jlptTotals = {'N5': 800, 'N4': 1500, 'N3': 3750, 'N2': 6000, 'N1': 10000};

class _JlptProgressBars extends StatelessWidget {
  const _JlptProgressBars({required this.words});
  final List<dynamic> words;

  @override
  Widget build(BuildContext context) {
    final counts = <String, int>{};
    for (final w in words) {
      final level = w.jlptLevel as String?;
      if (level != null) counts[level] = (counts[level] ?? 0) + 1;
    }

    return Column(
      children: ['N5', 'N4', 'N3', 'N2', 'N1'].map((level) {
        final count = counts[level] ?? 0;
        final total = _jlptTotals[level]!;
        final pct   = (count / total).clamp(0.0, 1.0);
        final color = _jlptLevelColors[level]!;

        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Column(
            children: [
              Row(children: [
                Container(
                  width: 28, height: 20,
                  decoration: BoxDecoration(
                    color:        color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Center(
                    child: Text(level,
                        style: TextStyle(
                            color:      color,
                            fontSize:   10,
                            fontWeight: FontWeight.bold)),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value:            pct,
                      minHeight:        8,
                      backgroundColor:  color.withOpacity(0.12),
                      valueColor:       AlwaysStoppedAnimation(color),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text('$count',
                    style: TextStyle(
                        fontSize:   12,
                        fontWeight: FontWeight.bold,
                        color:      color)),
              ]),
            ],
          ),
        );
      }).toList(),
    );
  }
}

// ── Grammar mastery bars ──────────────────────────────────────────────────────

class _GrammarMasteryBars extends StatelessWidget {
  const _GrammarMasteryBars({required this.grammarState});
  final GrammarState grammarState;

  @override
  Widget build(BuildContext context) {
    // Group by JLPT level
    final byLevel = <String, _LevelMasteryStats>{};
    for (final theme in grammarState.themes) {
      final lvl      = theme.jlptLevel;
      final mastery  = grammarState.progress[theme.id]?.masteryLevel ?? 0;
      final existing = byLevel[lvl] ?? _LevelMasteryStats(level: lvl);
      byLevel[lvl] = existing.add(mastery);
    }

    return Column(
      children: ['N5', 'N4', 'N3', 'N2', 'N1']
          .where((lvl) => byLevel.containsKey(lvl))
          .map((lvl) {
        final stats  = byLevel[lvl]!;
        final color  = _jlptLevelColors[lvl]!;
        final pct    = stats.total == 0 ? 0.0 : stats.masterySum / (stats.total * 3);

        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Row(children: [
            Container(
              width: 28, height: 20,
              decoration: BoxDecoration(
                color:        color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Center(
                child: Text(lvl,
                    style: TextStyle(
                        color:      color,
                        fontSize:   10,
                        fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value:           pct.clamp(0.0, 1.0),
                  minHeight:       8,
                  backgroundColor: color.withOpacity(0.12),
                  valueColor:      AlwaysStoppedAnimation(color),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text('${stats.mastered}/${stats.total}',
                style: TextStyle(
                    fontSize:   12,
                    fontWeight: FontWeight.bold,
                    color:      color)),
          ]),
        );
      }).toList(),
    );
  }
}

class _LevelMasteryStats {
  final String level;
  final int total;
  final int masterySum;
  final int mastered; // themes with mastery == 3

  const _LevelMasteryStats({
    required this.level,
    this.total = 0,
    this.masterySum = 0,
    this.mastered = 0,
  });

  _LevelMasteryStats add(int mastery) => _LevelMasteryStats(
        level:      level,
        total:      total + 1,
        masterySum: masterySum + mastery,
        mastered:   mastered + (mastery >= 3 ? 1 : 0),
      );
}

// ── Trophy grid ───────────────────────────────────────────────────────────────

class _TrophyGrid extends StatelessWidget {
  const _TrophyGrid({required this.trophies, required this.unlockedIds});

  final List<TrophyDefinition> trophies;
  final Set<String>            unlockedIds;

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      shrinkWrap: true,
      physics:    const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount:   3,
        childAspectRatio: 0.78,
        crossAxisSpacing: 10,
        mainAxisSpacing:  10,
      ),
      itemCount:   trophies.length,
      itemBuilder: (_, i) => _TrophyCard(
        trophy:    trophies[i],
        unlocked:  unlockedIds.contains(trophies[i].id),
      ),
    );
  }
}

class _TrophyCard extends StatelessWidget {
  const _TrophyCard({required this.trophy, required this.unlocked});

  final TrophyDefinition trophy;
  final bool             unlocked;

  @override
  Widget build(BuildContext context) {
    final l      = AppLocalizations.of(context);
    final color  = unlocked ? _tierColor(trophy.tier) : Colors.grey.shade300;
    final theme  = Theme.of(context);

    return GestureDetector(
      onTap: () => _showDetail(context, l),
      child: AnimatedOpacity(
        opacity:  unlocked ? 1.0 : 0.45,
        duration: const Duration(milliseconds: 300),
        child: Container(
          padding:     const EdgeInsets.all(10),
          decoration:  BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            color:        color.withOpacity(0.1),
            border:       Border.all(
              color: unlocked ? color.withOpacity(0.5) : Colors.grey.shade200,
              width: 1.5,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Medal ring
              Container(
                width:  52, height: 52,
                decoration: BoxDecoration(
                  shape:  BoxShape.circle,
                  color:  color.withOpacity(0.15),
                  border: Border.all(color: color, width: 2),
                ),
                child: Center(
                  child: Text(trophy.emoji,
                      style: const TextStyle(fontSize: 24)),
                ),
              ),
              const SizedBox(height: 6),
              Text(
                _trophyName(l, trophy.nameKey),
                style: theme.textTheme.bodySmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color:      unlocked
                      ? theme.colorScheme.onSurface
                      : Colors.grey.shade400,
                ),
                textAlign: TextAlign.center,
                maxLines:  2,
                overflow:  TextOverflow.ellipsis,
              ),
              if (unlocked)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color:        color.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      _tierLabel(trophy.tier),
                      style: TextStyle(
                          fontSize:   9,
                          fontWeight: FontWeight.bold,
                          color:      color),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  void _showDetail(BuildContext context, AppLocalizations l) {
    showModalBottomSheet<void>(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color:        Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Text(trophy.emoji, style: const TextStyle(fontSize: 48)),
            const SizedBox(height: 12),
            Text(
              _trophyName(l, trophy.nameKey),
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
              decoration: BoxDecoration(
                color:        _tierColor(trophy.tier).withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _tierLabel(trophy.tier),
                style: TextStyle(
                    color:      _tierColor(trophy.tier),
                    fontWeight: FontWeight.bold,
                    fontSize:   12),
              ),
            ),
            const SizedBox(height: 14),
            Text(
              _trophyDesc(l, trophy.descriptionKey),
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey.shade600),
            ),
            const SizedBox(height: 16),
            if (!unlocked)
              Text('🔒  ${_trophyDesc(l, trophy.descriptionKey)}',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey.shade400, fontSize: 13)),
          ],
        ),
      ),
    );
  }

  static String _tierLabel(TrophyTier t) {
    switch (t) {
      case TrophyTier.bronze:   return 'Bronze';
      case TrophyTier.silver:   return 'Silver';
      case TrophyTier.gold:     return 'Gold';
      case TrophyTier.platinum: return 'Platinum';
    }
  }
}

// ── Review card ───────────────────────────────────────────────────────────────

class _ReviewCard extends StatelessWidget {
  final int dueCount;
  const _ReviewCard({required this.dueCount});

  @override
  Widget build(BuildContext context) {
    final l           = AppLocalizations.of(context);
    final colorScheme = Theme.of(context).colorScheme;
    final theme       = Theme.of(context);

    return Material(
      color:        colorScheme.primaryContainer,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => Navigator.of(context).push(ReviewScreen.route()),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              Container(
                width:  52,
                height: 52,
                decoration: BoxDecoration(
                  color:  colorScheme.primary.withOpacity(0.15),
                  shape:  BoxShape.circle,
                ),
                child: Icon(Icons.flash_on_rounded,
                    size: 28, color: colorScheme.primary),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      l.reviewStart,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color:      colorScheme.onPrimaryContainer,
                      ),
                    ),
                    Text(
                      '${l.progressStatsDue}: $dueCount',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: colorScheme.onPrimaryContainer.withOpacity(0.8),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.arrow_forward_ios_rounded,
                  size: 18, color: colorScheme.onPrimaryContainer),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Section title ─────────────────────────────────────────────────────────────

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Text(
        text,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
      );
}
