import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../providers/auth_provider.dart';
import '../../providers/friends_provider.dart';

// ── Main screen ────────────────────────────────────────────────────────────────

class FriendsScreen extends ConsumerStatefulWidget {
  const FriendsScreen({super.key});

  @override
  ConsumerState<FriendsScreen> createState() => _FriendsScreenState();
}

class _FriendsScreenState extends ConsumerState<FriendsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabs;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l     = AppLocalizations.of(context);
    final state = ref.watch(friendsProvider);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.friendsTitle),
        centerTitle: true,
        bottom: TabBar(
          controller: _tabs,
          isScrollable: true,
          tabAlignment: TabAlignment.start,
          tabs: [
            Tab(text: l.friendsFriends),
            Tab(
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Text(l.friendsRequests),
                if (state.pendingIncoming > 0) ...[
                  const SizedBox(width: 6),
                  _Badge(state.pendingIncoming),
                ],
              ]),
            ),
            Tab(text: l.challengesTitle),
            Tab(text: l.friendsLeaderboard),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add_outlined),
            tooltip: l.friendsAddFriend,
            onPressed: () => _openSearch(context),
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(friendsProvider.notifier).load(),
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                if (state.error != null)
                  MaterialBanner(
                    content: Text(state.error!),
                    actions: [
                      TextButton(
                        onPressed: () =>
                            ref.read(friendsProvider.notifier).clearError(),
                        child: const Text('Dismiss'),
                      ),
                    ],
                    backgroundColor:
                        Theme.of(context).colorScheme.errorContainer,
                  ),
                Expanded(
                  child: TabBarView(
                    controller: _tabs,
                    children: [
                      _FriendsTab(onSearch: () => _openSearch(context)),
                      _RequestsTab(),
                      _ChallengesTab(),
                      _LeaderboardTab(),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  void _openSearch(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (_) => const _SearchSheet(),
    );
  }
}

// ── Badge ──────────────────────────────────────────────────────────────────────

class _Badge extends StatelessWidget {
  final int count;
  const _Badge(this.count);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color:        Theme.of(context).colorScheme.error,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        '$count',
        style: TextStyle(
          color:    Theme.of(context).colorScheme.onError,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

// ── Friends tab ────────────────────────────────────────────────────────────────

class _FriendsTab extends ConsumerWidget {
  final VoidCallback onSearch;
  const _FriendsTab({required this.onSearch});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l       = AppLocalizations.of(context);
    final friends = ref.watch(friendsProvider).friends;

    if (friends.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.people_outline,
                size: 64,
                color: Theme.of(context).colorScheme.outlineVariant),
            const SizedBox(height: 16),
            Text(l.friendsNoFriends,
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Text(
                l.friendsNoFriendsHint,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(context).colorScheme.onSurfaceVariant),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: onSearch,
              icon:  const Icon(Icons.person_search),
              label: Text(l.friendsAddFriend),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: () => ref.read(friendsProvider.notifier).load(),
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: friends.length,
        itemBuilder: (_, i) => _FriendCard(friendship: friends[i]),
      ),
    );
  }
}

// ── Friend card ────────────────────────────────────────────────────────────────

class _FriendCard extends ConsumerWidget {
  final FriendshipModel friendship;
  const _FriendCard({required this.friendship});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final u           = friendship.otherUser;
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            // Avatar
            CircleAvatar(
              radius: 24,
              backgroundColor: colorScheme.primaryContainer,
              child: Text(
                (u.username ?? '?').substring(0, 1).toUpperCase(),
                style: TextStyle(
                  color: colorScheme.onPrimaryContainer,
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                ),
              ),
            ),
            const SizedBox(width: 14),
            // Info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    u.displayName,
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  _UserStats(user: u),
                ],
              ),
            ),
            // Challenge + remove menu
            PopupMenuButton<String>(
              onSelected: (v) {
                if (v == 'challenge') {
                  _showChallengeSheet(context, ref, u);
                } else if (v == 'remove') {
                  _confirmRemove(context, ref, u.id, u.displayName);
                }
              },
              itemBuilder: (_) => [
                PopupMenuItem(
                  value: 'challenge',
                  child: Row(children: [
                    const Icon(Icons.emoji_events_outlined),
                    const SizedBox(width: 8),
                    Text(l.challengesSend),
                  ]),
                ),
                PopupMenuItem(
                  value: 'remove',
                  child: Row(children: [
                    Icon(Icons.person_remove_outlined,
                        color: Theme.of(context).colorScheme.error),
                    const SizedBox(width: 8),
                    Text(l.friendsRemove,
                        style: TextStyle(
                            color: Theme.of(context).colorScheme.error)),
                  ]),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _showChallengeSheet(
      BuildContext context, WidgetRef ref, PublicUserProfile friend) {
    showModalBottomSheet<void>(
      context: context,
      builder: (_) => _SendChallengeSheet(friend: friend),
    );
  }

  void _confirmRemove(
      BuildContext context, WidgetRef ref, String userId, String name) {
    final l = AppLocalizations.of(context);
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title:   Text(l.friendsRemoveTitle),
        content: Text(l.friendsRemoveContent(name)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(l.vocabCancel),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              ref.read(friendsProvider.notifier).removeFriend(userId);
            },
            child: Text(l.friendsRemove,
                style: TextStyle(
                    color: Theme.of(context).colorScheme.error)),
          ),
        ],
      ),
    );
  }
}

// ── User stats inline widget ───────────────────────────────────────────────────

class _UserStats extends StatelessWidget {
  final PublicUserProfile user;
  const _UserStats({required this.user});

  @override
  Widget build(BuildContext context) {
    final l     = AppLocalizations.of(context);
    final color = Theme.of(context).colorScheme.onSurfaceVariant;
    return Row(
      children: [
        _MiniStat(label: l.friendsVocabSize, value: '${user.vocabularySize}', color: color),
        const SizedBox(width: 12),
        _MiniStat(label: l.friendsMastered,  value: '${user.wordsMastered}',  color: color),
        const SizedBox(width: 12),
        _MiniStat(label: l.friendsStreak,
            value: '${user.currentStreakDays}🔥', color: color),
        const Spacer(),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
          decoration: BoxDecoration(
            color:        Theme.of(context).colorScheme.secondaryContainer,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Text(
            user.jlptLevel,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onSecondaryContainer,
            ),
          ),
        ),
      ],
    );
  }
}

class _MiniStat extends StatelessWidget {
  final String label;
  final String value;
  final Color  color;
  const _MiniStat({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
          Text(label, style: TextStyle(fontSize: 10, color: color)),
        ],
      );
}

// ── Requests tab ───────────────────────────────────────────────────────────────

class _RequestsTab extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l        = AppLocalizations.of(context);
    final state    = ref.watch(friendsProvider);
    final incoming = state.incomingRequests;
    final sent     = state.sentRequests;

    if (incoming.isEmpty && sent.isEmpty) {
      return Center(
        child: Text(l.friendsNoPending,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant)),
      );
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (incoming.isNotEmpty) ...[
          _SectionHeader(l.friendsRequests),
          ...incoming.map((f) => _RequestCard(friendship: f, isIncoming: true)),
          const SizedBox(height: 16),
        ],
        if (sent.isNotEmpty) ...[
          _SectionHeader(l.friendsSent),
          ...sent.map((f) => _RequestCard(friendship: f, isIncoming: false)),
        ],
      ],
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String text;
  const _SectionHeader(this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Text(
          text,
          style: Theme.of(context).textTheme.labelLarge?.copyWith(
              color: Theme.of(context).colorScheme.onSurfaceVariant),
        ),
      );
}

class _RequestCard extends ConsumerWidget {
  final FriendshipModel friendship;
  final bool            isIncoming;
  const _RequestCard({required this.friendship, required this.isIncoming});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final u           = friendship.otherUser;
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            CircleAvatar(
              radius: 20,
              backgroundColor: colorScheme.primaryContainer,
              child: Text(
                (u.username ?? '?').substring(0, 1).toUpperCase(),
                style: TextStyle(color: colorScheme.onPrimaryContainer),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(u.displayName,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(fontWeight: FontWeight.w600)),
            ),
            if (isIncoming) ...[
              TextButton(
                onPressed: () =>
                    ref.read(friendsProvider.notifier).acceptRequest(friendship.id),
                child: Text(l.friendsAccept),
              ),
              TextButton(
                onPressed: () =>
                    ref.read(friendsProvider.notifier).declineRequest(friendship.id),
                child: Text(l.friendsDecline,
                    style: TextStyle(color: colorScheme.error)),
              ),
            ] else
              Chip(
                label: Text(l.friendsRequestPending,
                    style: const TextStyle(fontSize: 12)),
                backgroundColor: colorScheme.surfaceContainerHighest,
              ),
          ],
        ),
      ),
    );
  }
}

// ── Challenges tab ─────────────────────────────────────────────────────────────

class _ChallengesTab extends ConsumerStatefulWidget {
  @override
  ConsumerState<_ChallengesTab> createState() => _ChallengesTabState();
}

class _ChallengesTabState extends ConsumerState<_ChallengesTab>
    with SingleTickerProviderStateMixin {
  late TabController _subtabs;

  @override
  void initState() {
    super.initState();
    _subtabs = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _subtabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l     = AppLocalizations.of(context);
    final state = ref.watch(friendsProvider);

    return Column(
      children: [
        TabBar(
          controller: _subtabs,
          tabs: [Tab(text: l.challengesActive), Tab(text: l.challengesHistory)],
        ),
        Expanded(
          child: TabBarView(
            controller: _subtabs,
            children: [
              _ChallengeList(
                challenges: state.activeChallenges,
                isHistory: false,
              ),
              _ChallengeList(
                challenges: state.challengeHistory,
                isHistory: true,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ChallengeList extends ConsumerWidget {
  final List<ChallengeModel> challenges;
  final bool isHistory;
  const _ChallengeList({required this.challenges, required this.isHistory});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l = AppLocalizations.of(context);

    if (challenges.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.emoji_events_outlined,
                size: 56,
                color: Theme.of(context).colorScheme.outlineVariant),
            const SizedBox(height: 12),
            Text(l.challengesNone,
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 6),
            Text(
              l.challengesNoneHint,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: challenges.length,
      itemBuilder: (_, i) =>
          _ChallengeCard(challenge: challenges[i]),
    );
  }
}

class _ChallengeCard extends ConsumerWidget {
  final ChallengeModel challenge;
  const _ChallengeCard({required this.challenge});

  String _typeName(AppLocalizations l) => switch (challenge.type) {
        'vocab_sprint'  => l.challengesTypeVocabSprint,
        'mastery_race'  => l.challengesTypeMasteryRace,
        'accuracy_duel' => l.challengesTypeAccuracyDuel,
        _               => challenge.type,
      };

  String _statusLabel(AppLocalizations l) => switch (challenge.status) {
        'pending'   => l.challengesStatusPending,
        'active'    => l.challengesStatusActive,
        'completed' => l.challengesStatusCompleted,
        'declined'  => l.challengesStatusDeclined,
        _           => challenge.status,
      };

  Color _statusColor(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return switch (challenge.status) {
      'active'    => Colors.green,
      'pending'   => cs.tertiary,
      'completed' => cs.primary,
      _           => cs.outline,
    };
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final colorScheme = Theme.of(context).colorScheme;
    final theme       = Theme.of(context);

    // Determine my role
    final authState = ref.watch(authProvider);
    final myId      = authState.user?.id ?? '';
    final isSender  = challenge.sender.id == myId;
    final me        = isSender ? challenge.sender    : challenge.recipient;
    final opponent  = isSender ? challenge.recipient : challenge.sender;
    final myScore   = isSender ? challenge.senderScore : challenge.recipientScore;
    final opScore   = isSender ? challenge.recipientScore : challenge.senderScore;

    return Card(
      margin: const EdgeInsets.only(bottom: 14),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header: type + status chip
            Row(children: [
              Expanded(
                child: Text(
                  _typeName(l),
                  style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color:        _statusColor(context).withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                  border:       Border.all(color: _statusColor(context).withOpacity(0.3)),
                ),
                child: Text(
                  _statusLabel(l),
                  style: TextStyle(
                    fontSize:   11,
                    color:      _statusColor(context),
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ]),

            const SizedBox(height: 12),

            // Scoreboard
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _ScoreColumn(
                  name:    l.challengesYou,
                  score:   myScore,
                  type:    challenge.type,
                  isWinner: challenge.winnerId == me.id,
                  l: l,
                ),
                Text(l.challengesVs,
                    style: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.onSurfaceVariant)),
                _ScoreColumn(
                  name:     opponent.displayName,
                  score:    opScore,
                  type:     challenge.type,
                  isWinner: challenge.winnerId == opponent.id,
                  l: l,
                ),
              ],
            ),

            // Days left / duration
            if (challenge.status == 'active' &&
                challenge.daysLeft != null) ...[
              const SizedBox(height: 10),
              Text(
                l.challengesEndsInDays(challenge.daysLeft!),
                style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant),
              ),
            ],

            // Action buttons for pending incoming
            if (challenge.status == 'pending' &&
                challenge.recipient.id == myId) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => ref
                          .read(friendsProvider.notifier)
                          .declineChallenge(challenge.id),
                      child: Text(l.challengesDecline),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: FilledButton(
                      onPressed: () => ref
                          .read(friendsProvider.notifier)
                          .acceptChallenge(challenge.id),
                      child: Text(l.challengesAccept),
                    ),
                  ),
                ],
              ),
            ],

            // Tie/winner for completed
            if (challenge.status == 'completed') ...[
              const SizedBox(height: 8),
              Row(children: [
                const Icon(Icons.emoji_events, size: 16, color: Color(0xFFFFD700)),
                const SizedBox(width: 4),
                Text(
                  challenge.winnerId == null
                      ? l.challengesTie
                      : '${l.challengesWinner}: ${challenge.winnerId == myId ? l.challengesYou : opponent.displayName}',
                  style: theme.textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.bold),
                ),
              ]),
            ],
          ],
        ),
      ),
    );
  }
}

class _ScoreColumn extends StatelessWidget {
  final String  name;
  final double? score;
  final String  type;
  final bool    isWinner;
  final AppLocalizations l;

  const _ScoreColumn({
    required this.name,
    required this.score,
    required this.type,
    required this.isWinner,
    required this.l,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      children: [
        if (isWinner)
          const Icon(Icons.emoji_events, size: 18, color: Color(0xFFFFD700)),
        Text(
          score != null ? l.challengesScore(score!, type) : '—',
          style: theme.textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.bold,
            color: isWinner ? const Color(0xFFFFD700) : null,
          ),
        ),
        Text(name,
            style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant)),
      ],
    );
  }
}

// ── Leaderboard tab ────────────────────────────────────────────────────────────

class _LeaderboardTab extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final leaderboard = ref.watch(friendsProvider).leaderboard;
    final colorScheme = Theme.of(context).colorScheme;
    final theme       = Theme.of(context);

    if (leaderboard.isEmpty) {
      return Center(
        child: Text(l.friendsNoFriends,
            style: theme.textTheme.bodyLarge?.copyWith(
                color: colorScheme.onSurfaceVariant)),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: leaderboard.length,
      itemBuilder: (_, i) {
        final entry = leaderboard[i];
        final u     = entry.user;
        final medal = switch (entry.rank) {
          1 => '🥇',
          2 => '🥈',
          3 => '🥉',
          _ => '${entry.rank}.',
        };

        return Card(
          margin: const EdgeInsets.only(bottom: 10),
          color: entry.isSelf
              ? colorScheme.primaryContainer.withOpacity(0.35)
              : null,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Row(
              children: [
                SizedBox(
                  width: 36,
                  child: Text(medal,
                      style: const TextStyle(fontSize: 20),
                      textAlign: TextAlign.center),
                ),
                const SizedBox(width: 10),
                CircleAvatar(
                  radius: 18,
                  backgroundColor: colorScheme.primaryContainer,
                  child: Text(
                    (u.username ?? '?').substring(0, 1).toUpperCase(),
                    style: TextStyle(color: colorScheme.onPrimaryContainer),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${u.displayName}${entry.isSelf ? ' (${l.challengesYou})' : ''}',
                        style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.bold),
                      ),
                      _UserStats(user: u),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ── Search sheet ───────────────────────────────────────────────────────────────

class _SearchSheet extends ConsumerStatefulWidget {
  const _SearchSheet();

  @override
  ConsumerState<_SearchSheet> createState() => _SearchSheetState();
}

class _SearchSheetState extends ConsumerState<_SearchSheet> {
  final _ctrl = TextEditingController();

  @override
  void dispose() {
    _ctrl.dispose();
    ref.read(friendsProvider.notifier).clearSearch();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l           = AppLocalizations.of(context);
    final friendState = ref.watch(friendsProvider);

    return DraggableScrollableSheet(
      initialChildSize: 0.9,
      minChildSize:     0.5,
      maxChildSize:     0.95,
      expand:           false,
      builder: (_, controller) => Padding(
        padding: EdgeInsets.only(
          left: 16, right: 16, top: 16,
          bottom: MediaQuery.of(context).viewInsets.bottom + 16,
        ),
        child: Column(
          children: [
            // Drag handle
            Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color:        Theme.of(context).colorScheme.outlineVariant,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Text(l.friendsAddFriend,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            SearchBar(
              controller: _ctrl,
              hintText:   l.friendsSearchHint,
              leading:    const Icon(Icons.search),
              onChanged:  (v) =>
                  ref.read(friendsProvider.notifier).searchUsers(v),
            ),
            const SizedBox(height: 16),
            if (friendState.isSearching)
              const Center(child: CircularProgressIndicator())
            else if (friendState.searchResults.isEmpty &&
                _ctrl.text.length >= 2)
              Text(l.friendsSearchNoResults,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant))
            else
              Expanded(
                child: ListView.builder(
                  controller: controller,
                  itemCount:  friendState.searchResults.length,
                  itemBuilder: (_, i) =>
                      _SearchResultTile(user: friendState.searchResults[i]),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ── Search result tile ─────────────────────────────────────────────────────────

class _SearchResultTile extends ConsumerWidget {
  final PublicUserProfile user;
  const _SearchResultTile({required this.user});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l           = AppLocalizations.of(context);
    final state       = ref.watch(friendsProvider);
    final colorScheme = Theme.of(context).colorScheme;

    final isAlreadyFriend = state.friends
        .any((f) => f.otherUser.id == user.id);
    final isPending = state.sentRequests
        .any((f) => f.otherUser.id == user.id);

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: colorScheme.primaryContainer,
        child: Text(
          (user.username ?? '?').substring(0, 1).toUpperCase(),
          style: TextStyle(color: colorScheme.onPrimaryContainer),
        ),
      ),
      title:    Text(user.displayName),
      subtitle: Text(user.jlptLevel),
      trailing: isAlreadyFriend
          ? Chip(
              label: Text(l.friendsAlreadyFriends,
                  style: const TextStyle(fontSize: 12)))
          : isPending
              ? Chip(
                  label: Text(l.friendsRequestPending,
                      style: const TextStyle(fontSize: 12)))
              : FilledButton.tonal(
                  onPressed: () {
                    ref
                        .read(friendsProvider.notifier)
                        .sendRequest(user.id);
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(l.friendsRequestSent)),
                    );
                  },
                  child: Text(l.friendsAddFriend),
                ),
    );
  }
}

// ── Send challenge sheet ───────────────────────────────────────────────────────

class _SendChallengeSheet extends ConsumerStatefulWidget {
  final PublicUserProfile friend;
  const _SendChallengeSheet({required this.friend});

  @override
  ConsumerState<_SendChallengeSheet> createState() =>
      _SendChallengeSheetState();
}

class _SendChallengeSheetState extends ConsumerState<_SendChallengeSheet> {
  String _type         = 'vocab_sprint';
  int    _durationDays = 7;

  @override
  Widget build(BuildContext context) {
    final l     = AppLocalizations.of(context);
    final theme = Theme.of(context);

    final types = [
      (id: 'vocab_sprint',  name: l.challengesTypeVocabSprint,  desc: l.challengesTypeVocabSprintDesc,  icon: Icons.add_box_outlined),
      (id: 'mastery_race',  name: l.challengesTypeMasteryRace,  desc: l.challengesTypeMasteryRaceDesc,  icon: Icons.star_outline),
      (id: 'accuracy_duel', name: l.challengesTypeAccuracyDuel, desc: l.challengesTypeAccuracyDuelDesc, icon: Icons.gps_fixed_outlined),
    ];

    return Padding(
      padding: EdgeInsets.only(
        left: 20, right: 20, top: 20,
        bottom: MediaQuery.of(context).viewInsets.bottom + 24,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color:        theme.colorScheme.outlineVariant,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          Text(
            '${l.challengesSend} — ${widget.friend.displayName}',
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),

          // Challenge type selection
          ...types.map((t) => RadioListTile<String>(
                value:    t.id,
                groupValue: _type,
                onChanged: (v) => setState(() => _type = v!),
                title:    Text(t.name, style: theme.textTheme.bodyMedium),
                subtitle: Text(t.desc, style: theme.textTheme.bodySmall),
                secondary: Icon(t.icon),
                contentPadding: EdgeInsets.zero,
              )),

          const SizedBox(height: 12),

          // Duration picker
          Row(children: [
            Text('${l.challengesDuration}: ', style: theme.textTheme.bodyMedium),
            ...([3, 7, 14, 30]).map((d) => Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label:     Text(l.challengesDurationDays(d)),
                    selected:  _durationDays == d,
                    onSelected: (_) => setState(() => _durationDays = d),
                  ),
                )),
          ]),

          const SizedBox(height: 20),

          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              icon:  const Icon(Icons.send_outlined),
              label: Text(l.challengesSend),
              onPressed: () {
                ref.read(friendsProvider.notifier).sendChallenge(
                  recipientId:  widget.friend.id,
                  type:         _type,
                  durationDays: _durationDays,
                );
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      '${l.challengesSend} → ${widget.friend.displayName} ✓'),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
