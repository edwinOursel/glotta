import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../models/grammar_theme.dart';
import '../../providers/grammar_provider.dart';

// ── Icon map (material icon name → IconData) ──────────────────────────────────

const _icons = <String, IconData>{
  'school':              Icons.school_outlined,
  'tag':                 Icons.label_outlined,
  'place':               Icons.place_outlined,
  'arrow_forward':       Icons.arrow_forward,
  'play_arrow':          Icons.play_arrow_outlined,
  'history':             Icons.history,
  'favorite_border':     Icons.favorite_border,
  'fitness_center':      Icons.fitness_center,
  'warning':             Icons.warning_amber_outlined,
  'link':                Icons.link,
  'loop':                Icons.loop,
  'pan_tool':            Icons.pan_tool_outlined,
  'access_time':         Icons.access_time,
  'done_all':            Icons.done_all,
  'palette':             Icons.palette_outlined,
  'star_border':         Icons.star_border,
  'compare_arrows':      Icons.compare_arrows,
  'device_hub':          Icons.device_hub,
  'trending_flat':       Icons.trending_flat,
  'account_tree':        Icons.account_tree_outlined,
  'lightbulb_outline':   Icons.lightbulb_outline,
  'help_outline':        Icons.help_outline,
  'description':         Icons.description_outlined,
  'sentiment_dissatisfied': Icons.sentiment_dissatisfied_outlined,
  'card_giftcard':       Icons.card_giftcard_outlined,
  'inbox':               Icons.inbox_outlined,
  'volunteer_activism':  Icons.volunteer_activism,
  'swap_vert':           Icons.swap_vert,
  'supervisor_account':  Icons.supervisor_account_outlined,
  'block':               Icons.block,
  'cloud_queue':         Icons.cloud_queue,
  'psychology':          Icons.psychology_outlined,
  'record_voice_over':   Icons.record_voice_over_outlined,
  'visibility':          Icons.visibility_outlined,
  'notes':               Icons.notes,
  'flag':                Icons.flag_outlined,
  'security':            Icons.security_outlined,
  'not_interested':      Icons.not_interested,
  'emoji_people':        Icons.emoji_people_outlined,
};

IconData _iconFor(String name) => _icons[name] ?? Icons.auto_stories_outlined;

// ── Screen ────────────────────────────────────────────────────────────────────

class GrammarSelectionScreen extends ConsumerStatefulWidget {
  const GrammarSelectionScreen({super.key});

  @override
  ConsumerState<GrammarSelectionScreen> createState() =>
      _GrammarSelectionScreenState();
}

class _GrammarSelectionScreenState
    extends ConsumerState<GrammarSelectionScreen> {
  String? _selectedCategory;

  @override
  Widget build(BuildContext context) {
    final grammarState = ref.watch(grammarProvider);
    final theme        = Theme.of(context);
    final colors       = theme.colorScheme;

    if (grammarState.isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final categories   = grammarState.byCategory;
    final categoryKeys = categories.keys.toList();

    return Scaffold(
      appBar: AppBar(
        title: Text(AppLocalizations.of(context).grammarTitle),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(48),
          child: _CategoryFilter(
            categories:       categoryKeys,
            selected:         _selectedCategory,
            onSelect:         (c) =>
                setState(() => _selectedCategory = _selectedCategory == c ? null : c),
          ),
        ),
        actions: [
          if (grammarState.selected != null)
            TextButton.icon(
              icon:  const Icon(Icons.clear),
              label: Text(AppLocalizations.of(context).grammarClear),
              onPressed: () =>
                  ref.read(grammarProvider.notifier).selectTheme(null),
            ),
        ],
      ),

      body: Column(
        children: [
          // ── Active theme banner ──────────────────────────────────────
          if (grammarState.selected != null)
            _ActiveBanner(theme: grammarState.selected!),

          // ── Theme grid ───────────────────────────────────────────────
          Expanded(
            child: ListView.builder(
              padding:     const EdgeInsets.all(16),
              itemCount:   _selectedCategory != null ? 1 : categoryKeys.length,
              itemBuilder: (_, i) {
                final cat    = _selectedCategory ?? categoryKeys[i];
                final themes = categories[cat] ?? [];
                return _CategorySection(
                  category: cat,
                  themes:   themes,
                  state:    grammarState,
                  onSelect: (t) => _onSelect(t, grammarState),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  void _onSelect(GrammarTheme t, GrammarState gs) {
    if (!gs.isUnlocked(t)) {
      _showLockedDialog(t, gs);
      return;
    }
    if (gs.selected?.id == t.id) {
      ref.read(grammarProvider.notifier).selectTheme(null);
    } else {
      ref.read(grammarProvider.notifier).selectTheme(t);
      _showThemeSheet(t, gs);
    }
  }

  void _showLockedDialog(GrammarTheme t, GrammarState gs) {
    final prereqs = t.prerequisiteIds
        .map((id) => gs.themes.firstWhere((x) => x.id == id,
            orElse: () => GrammarTheme(
              id: id, category: '', jlptLevel: '',
              nameEn: id, nameJp: id, description: '',
              exampleJp: '', exampleEn: '', systemPromptHint: '',
              prerequisiteIds: [], icon: '',
            )))
        .toList();

    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        icon: const Icon(Icons.lock_outline, size: 40),
        title: Text(t.nameEn),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(AppLocalizations.of(context).grammarCompleteFirst),
            const SizedBox(height: 12),
            ...prereqs.map((p) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Row(children: [
                    const Icon(Icons.check_box_outline_blank, size: 18),
                    const SizedBox(width: 8),
                    Text('${p.nameEn} (${p.nameJp})'),
                  ]),
                )),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(AppLocalizations.of(context).grammarOk),
          ),
        ],
      ),
    );
  }

  void _showThemeSheet(GrammarTheme t, GrammarState gs) {
    final progress = gs.progress[t.id];

    showModalBottomSheet<void>(
      context:     context,
      isScrollControlled: true,
      useSafeArea: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => _ThemeDetailSheet(
        theme:    t,
        progress: progress,
        onStart:  () {
          Navigator.pop(context);
          Navigator.pop(context); // back to LearnScreen
        },
      ),
    );
  }
}

// ── Category filter chips ─────────────────────────────────────────────────────

class _CategoryFilter extends StatelessWidget {
  const _CategoryFilter({
    required this.categories,
    required this.selected,
    required this.onSelect,
  });

  final List<String> categories;
  final String?      selected;
  final void Function(String) onSelect;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 48,
      child: ListView.separated(
        padding:       const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemCount:     categories.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final cat   = categories[i];
          final meta  = kCategories[cat];
          final isSelected = selected == cat;
          return FilterChip(
            label:    Text(meta?.label ?? cat),
            selected: isSelected,
            onSelected: (_) => onSelect(cat),
          );
        },
      ),
    );
  }
}

// ── Category section ──────────────────────────────────────────────────────────

class _CategorySection extends StatelessWidget {
  const _CategorySection({
    required this.category,
    required this.themes,
    required this.state,
    required this.onSelect,
  });

  final String                category;
  final List<GrammarTheme>    themes;
  final GrammarState          state;
  final void Function(GrammarTheme) onSelect;

  @override
  Widget build(BuildContext context) {
    final meta  = kCategories[category];
    final color = meta != null ? Color(meta.colorValue) : Colors.grey;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 12, top: 8),
          child: Row(children: [
            Container(
              width: 4, height: 20,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(width: 10),
            Text(
              meta?.label ?? category,
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 15,
              ),
            ),
          ]),
        ),
        GridView.builder(
          shrinkWrap: true,
          physics:    const NeverScrollableScrollPhysics(),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount:    2,
            childAspectRatio:  0.85,
            crossAxisSpacing:  12,
            mainAxisSpacing:   12,
          ),
          itemCount:   themes.length,
          itemBuilder: (_, i) => _ThemeCard(
            theme:    themes[i],
            state:    state,
            onSelect: onSelect,
          ),
        ),
        const SizedBox(height: 24),
      ],
    );
  }
}

// ── Theme card ────────────────────────────────────────────────────────────────

class _ThemeCard extends StatelessWidget {
  const _ThemeCard({
    required this.theme,
    required this.state,
    required this.onSelect,
  });

  final GrammarTheme  theme;
  final GrammarState  state;
  final void Function(GrammarTheme) onSelect;

  @override
  Widget build(BuildContext context) {
    final unlocked = state.isUnlocked(theme);
    final progress = state.progress[theme.id];
    final mastery  = progress?.masteryLevel ?? 0;
    final selected = state.selected?.id == theme.id;

    final levelColor = Color(kLevelColors[theme.jlptLevel] ?? 0xFF9E9E9E);
    final catColor   = Color(
        kCategories[theme.category]?.colorValue ?? 0xFF9E9E9E);

    return GestureDetector(
      onTap: () => onSelect(theme),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: selected
              ? Border.all(color: catColor, width: 2.5)
              : Border.all(color: Colors.transparent, width: 2.5),
          boxShadow: [
            BoxShadow(
              color:       catColor.withOpacity(selected ? 0.35 : 0.12),
              blurRadius:  selected ? 12 : 6,
              offset:      const Offset(0, 3),
            ),
          ],
          gradient: LinearGradient(
            begin:  Alignment.topLeft,
            end:    Alignment.bottomRight,
            colors: unlocked
                ? [
                    catColor.withOpacity(0.15),
                    catColor.withOpacity(0.05),
                  ]
                : [Colors.grey.shade100, Colors.grey.shade50],
          ),
        ),
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Icon + JLPT badge ───────────────────────────────
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color:        catColor.withOpacity(unlocked ? 0.2 : 0.08),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Icon(
                          _iconFor(theme.icon),
                          size:  22,
                          color: unlocked ? catColor : Colors.grey.shade400,
                        ),
                      ),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color:        levelColor.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          theme.jlptLevel,
                          style: TextStyle(
                            color:      levelColor,
                            fontSize:   10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),

                  // ── Names ───────────────────────────────────────────
                  Text(
                    theme.nameJp,
                    style: TextStyle(
                      fontSize:   13,
                      fontWeight: FontWeight.bold,
                      color:      unlocked ? null : Colors.grey,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    theme.nameEn,
                    style: TextStyle(
                      fontSize: 11,
                      color:    unlocked
                          ? Colors.grey.shade600
                          : Colors.grey.shade400,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),

                  const Spacer(),

                  // ── Mastery stars ───────────────────────────────────
                  if (unlocked)
                    Row(
                      children: List.generate(3, (i) => Icon(
                        i < mastery ? Icons.star : Icons.star_border,
                        size:  14,
                        color: i < mastery
                            ? Colors.amber.shade600
                            : Colors.grey.shade300,
                      )),
                    ),
                ],
              ),
            ),

            // ── Lock overlay ────────────────────────────────────────
            if (!unlocked)
              Positioned.fill(
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    color:        Colors.white.withOpacity(0.45),
                  ),
                  child: Center(
                    child: Icon(
                      Icons.lock_outline,
                      color: Colors.grey.shade400,
                      size:  32,
                    ),
                  ),
                ),
              ),

            // ── Selected checkmark ──────────────────────────────────
            if (selected)
              Positioned(
                bottom: 10,
                right:  10,
                child:  Container(
                  padding: const EdgeInsets.all(3),
                  decoration: BoxDecoration(
                    color:  catColor,
                    shape:  BoxShape.circle,
                  ),
                  child: const Icon(Icons.check, color: Colors.white, size: 14),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ── Theme detail bottom sheet ─────────────────────────────────────────────────

class _ThemeDetailSheet extends StatelessWidget {
  const _ThemeDetailSheet({
    required this.theme,
    required this.progress,
    required this.onStart,
  });

  final GrammarTheme   theme;
  final ThemeProgress? progress;
  final VoidCallback   onStart;

  @override
  Widget build(BuildContext context) {
    final catColor   = Color(
        kCategories[theme.category]?.colorValue ?? 0xFF9E9E9E);
    final levelColor = Color(kLevelColors[theme.jlptLevel] ?? 0xFF9E9E9E);
    final mastery    = progress?.masteryLevel ?? 0;
    final sessions   = progress?.sessionsCount ?? 0;

    final l = AppLocalizations.of(context);
    return DraggableScrollableSheet(
      expand:          false,
      initialChildSize: 0.6,
      minChildSize:    0.4,
      maxChildSize:    0.95,
      builder: (_, controller) => SingleChildScrollView(
        controller: controller,
        padding:    const EdgeInsets.fromLTRB(24, 8, 24, 32),
        child: Column(
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

            // ── Header ─────────────────────────────────────────────────
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color:        catColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(_iconFor(theme.icon), size: 28, color: catColor),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(theme.nameJp,
                          style: const TextStyle(
                              fontSize: 20, fontWeight: FontWeight.bold)),
                      Text(theme.nameEn,
                          style: TextStyle(
                              fontSize: 14, color: Colors.grey.shade600)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color:        levelColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(theme.jlptLevel,
                      style: TextStyle(
                          color:      levelColor,
                          fontWeight: FontWeight.bold)),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // ── Stats row ───────────────────────────────────────────────
            Row(
              children: [
                _StatChip(
                  label: l.grammarSessions,
                  value: '$sessions',
                  icon:  Icons.play_circle_outline,
                  color: catColor,
                ),
                const SizedBox(width: 12),
                _StatChip(
                  label: l.grammarMastery,
                  value: '★' * mastery + '☆' * (3 - mastery),
                  icon:  Icons.grade_outlined,
                  color: Colors.amber.shade700,
                ),
              ],
            ),
            const SizedBox(height: 20),

            // ── Description ─────────────────────────────────────────────
            Text(l.grammarDescription,
                style: TextStyle(
                    fontWeight: FontWeight.w700, color: Colors.grey.shade700)),
            const SizedBox(height: 6),
            Text(theme.description),
            const SizedBox(height: 20),

            // ── Example ─────────────────────────────────────────────────
            Text(l.grammarExample,
                style: TextStyle(
                    fontWeight: FontWeight.w700, color: Colors.grey.shade700)),
            const SizedBox(height: 8),
            Container(
              width:   double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color:        catColor.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
                border:       Border.all(color: catColor.withOpacity(0.2)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(theme.exampleJp,
                      style: const TextStyle(
                          fontSize: 17, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 6),
                  Text(theme.exampleEn,
                      style: TextStyle(
                          fontSize: 13,
                          color:    Colors.grey.shade600,
                          fontStyle: FontStyle.italic)),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── System prompt hint ───────────────────────────────────────
            Text(l.grammarPracticeFocus,
                style: TextStyle(
                    fontWeight: FontWeight.w700, color: Colors.grey.shade700)),
            const SizedBox(height: 6),
            Text(theme.systemPromptHint,
                style: TextStyle(color: Colors.grey.shade700)),
            const SizedBox(height: 28),

            // ── CTA ─────────────────────────────────────────────────────
            FilledButton.icon(
              onPressed: onStart,
              icon:      const Icon(Icons.auto_stories),
              label:     Text(l.grammarPractiseButton),
              style: FilledButton.styleFrom(
                backgroundColor: catColor,
                minimumSize:     const Size.fromHeight(52),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatChip extends StatelessWidget {
  const _StatChip({
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
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color:        color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(label,
                style: TextStyle(
                    fontSize: 10, color: color.withOpacity(0.8))),
            Text(value,
                style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                    color: color)),
          ]),
        ]),
      );
}
