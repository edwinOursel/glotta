import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/auth_provider.dart';
import '../providers/generation_provider.dart' show apiServiceProvider;

// ── Data models ────────────────────────────────────────────────────────────────

class PublicUserProfile {
  final String  id;
  final String? username;
  final String  jlptLevel;
  final int     vocabularySize;
  final int     wordsMastered;
  final int     currentStreakDays;

  const PublicUserProfile({
    required this.id,
    this.username,
    required this.jlptLevel,
    required this.vocabularySize,
    required this.wordsMastered,
    required this.currentStreakDays,
  });

  String get displayName => username ?? id.substring(0, 8);

  factory PublicUserProfile.fromJson(Map<String, dynamic> j) =>
      PublicUserProfile(
        id:               j['id'] as String,
        username:         j['username'] as String?,
        jlptLevel:        j['jlpt_level'] as String? ?? 'N5',
        vocabularySize:   j['vocabulary_size'] as int? ?? 0,
        wordsMastered:    j['words_mastered'] as int? ?? 0,
        currentStreakDays: j['current_streak_days'] as int? ?? 0,
      );
}

class FriendshipModel {
  final String          id;
  final String          status;   // pending | accepted | declined
  final DateTime        createdAt;
  final DateTime        updatedAt;
  final PublicUserProfile otherUser;

  const FriendshipModel({
    required this.id,
    required this.status,
    required this.createdAt,
    required this.updatedAt,
    required this.otherUser,
  });

  factory FriendshipModel.fromJson(Map<String, dynamic> j) =>
      FriendshipModel(
        id:        j['id'] as String,
        status:    j['status'] as String,
        createdAt: DateTime.parse(j['created_at'] as String),
        updatedAt: DateTime.parse(j['updated_at'] as String),
        otherUser: PublicUserProfile.fromJson(
            j['other_user'] as Map<String, dynamic>),
      );
}

class LeaderboardEntry {
  final int             rank;
  final PublicUserProfile user;
  final bool            isSelf;

  const LeaderboardEntry({
    required this.rank,
    required this.user,
    required this.isSelf,
  });

  factory LeaderboardEntry.fromJson(Map<String, dynamic> j) =>
      LeaderboardEntry(
        rank:   j['rank'] as int,
        user:   PublicUserProfile.fromJson(j['user'] as Map<String, dynamic>),
        isSelf: j['is_self'] as bool? ?? false,
      );
}

class ChallengeModel {
  final String  id;
  final String  type;   // vocab_sprint | mastery_race | accuracy_duel
  final String  status; // pending | active | completed | declined | expired
  final int     durationDays;
  final DateTime? startsAt;
  final DateTime? endsAt;
  final DateTime  createdAt;
  final PublicUserProfile sender;
  final PublicUserProfile recipient;
  final double? senderScore;
  final double? recipientScore;
  final String? winnerId;

  const ChallengeModel({
    required this.id,
    required this.type,
    required this.status,
    required this.durationDays,
    this.startsAt,
    this.endsAt,
    required this.createdAt,
    required this.sender,
    required this.recipient,
    this.senderScore,
    this.recipientScore,
    this.winnerId,
  });

  factory ChallengeModel.fromJson(Map<String, dynamic> j) =>
      ChallengeModel(
        id:           j['id'] as String,
        type:         j['type'] as String,
        status:       j['status'] as String,
        durationDays: j['duration_days'] as int? ?? 7,
        startsAt:     j['starts_at'] != null
            ? DateTime.parse(j['starts_at'] as String)
            : null,
        endsAt: j['ends_at'] != null
            ? DateTime.parse(j['ends_at'] as String)
            : null,
        createdAt: DateTime.parse(j['created_at'] as String),
        sender:    PublicUserProfile.fromJson(j['sender'] as Map<String, dynamic>),
        recipient: PublicUserProfile.fromJson(j['recipient'] as Map<String, dynamic>),
        senderScore:    (j['sender_score'] as num?)?.toDouble(),
        recipientScore: (j['recipient_score'] as num?)?.toDouble(),
        winnerId:       j['winner_id'] as String?,
      );

  int? get daysLeft {
    if (endsAt == null) return null;
    final diff = endsAt!.difference(DateTime.now()).inDays;
    return diff < 0 ? 0 : diff;
  }
}

// ── Friends state ──────────────────────────────────────────────────────────────

class FriendsState {
  final List<FriendshipModel>  friends;
  final List<FriendshipModel>  incomingRequests;
  final List<FriendshipModel>  sentRequests;
  final List<LeaderboardEntry> leaderboard;
  final List<ChallengeModel>   activeChallenges;
  final List<ChallengeModel>   challengeHistory;
  final List<PublicUserProfile> searchResults;
  final bool   isSearching;
  final bool   isLoading;
  final String? error;

  const FriendsState({
    this.friends           = const [],
    this.incomingRequests  = const [],
    this.sentRequests      = const [],
    this.leaderboard       = const [],
    this.activeChallenges  = const [],
    this.challengeHistory  = const [],
    this.searchResults     = const [],
    this.isSearching       = false,
    this.isLoading         = false,
    this.error,
  });

  FriendsState copyWith({
    List<FriendshipModel>?   friends,
    List<FriendshipModel>?   incomingRequests,
    List<FriendshipModel>?   sentRequests,
    List<LeaderboardEntry>?  leaderboard,
    List<ChallengeModel>?    activeChallenges,
    List<ChallengeModel>?    challengeHistory,
    List<PublicUserProfile>? searchResults,
    bool?   isSearching,
    bool?   isLoading,
    String? error,
    bool    clearError = false,
  }) =>
      FriendsState(
        friends:          friends          ?? this.friends,
        incomingRequests: incomingRequests ?? this.incomingRequests,
        sentRequests:     sentRequests     ?? this.sentRequests,
        leaderboard:      leaderboard      ?? this.leaderboard,
        activeChallenges: activeChallenges ?? this.activeChallenges,
        challengeHistory: challengeHistory ?? this.challengeHistory,
        searchResults:    searchResults    ?? this.searchResults,
        isSearching:      isSearching      ?? this.isSearching,
        isLoading:        isLoading        ?? this.isLoading,
        error:            clearError ? null : error ?? this.error,
      );

  int get pendingIncoming => incomingRequests.length;
}

// ── Notifier ──────────────────────────────────────────────────────────────────

class FriendsNotifier extends StateNotifier<FriendsState> {
  final Ref _ref;

  FriendsNotifier(this._ref) : super(const FriendsState()) {
    load();
  }

  Future<String?> _token() =>
      _ref.read(authProvider.notifier).getAccessToken();

  // ── Full reload ────────────────────────────────────────────────────────────

  Future<void> load() async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final token = await _token();
      if (token == null) {
        state = state.copyWith(isLoading: false);
        return;
      }
      final api = _ref.read(apiServiceProvider);

      final results = await Future.wait([
        api.getFriends(accessToken: token),
        api.getIncomingRequests(accessToken: token),
        api.getSentRequests(accessToken: token),
        api.getLeaderboard(accessToken: token),
        api.getChallenges(accessToken: token),
        api.getChallengeHistory(accessToken: token),
      ]);

      state = state.copyWith(
        friends:          (results[0]).map(FriendshipModel.fromJson).toList(),
        incomingRequests: (results[1]).map(FriendshipModel.fromJson).toList(),
        sentRequests:     (results[2]).map(FriendshipModel.fromJson).toList(),
        leaderboard:      (results[3]).map(LeaderboardEntry.fromJson).toList(),
        activeChallenges: (results[4]).map(ChallengeModel.fromJson).toList(),
        challengeHistory: (results[5]).map(ChallengeModel.fromJson).toList(),
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to load friends: $e',
      );
    }
  }

  // ── Friend requests ────────────────────────────────────────────────────────

  Future<void> sendRequest(String addresseeId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      final raw = await api.sendFriendRequest(
          accessToken: token, addresseeId: addresseeId);
      final friendship = FriendshipModel.fromJson(raw);
      state = state.copyWith(
          sentRequests: [...state.sentRequests, friendship]);
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> acceptRequest(String friendshipId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      final raw = await api.acceptFriendRequest(
          accessToken: token, friendshipId: friendshipId);
      final accepted = FriendshipModel.fromJson(raw);
      state = state.copyWith(
        incomingRequests:
            state.incomingRequests.where((f) => f.id != friendshipId).toList(),
        friends: [...state.friends, accepted],
      );
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> declineRequest(String friendshipId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.declineFriendRequest(
          accessToken: token, friendshipId: friendshipId);
      state = state.copyWith(
        incomingRequests:
            state.incomingRequests.where((f) => f.id != friendshipId).toList(),
      );
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> removeFriend(String userId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.unfriend(accessToken: token, userId: userId);
      state = state.copyWith(
        friends:
            state.friends.where((f) => f.otherUser.id != userId).toList(),
        leaderboard:
            state.leaderboard.where((e) => e.user.id != userId).toList(),
      );
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  // ── Challenges ─────────────────────────────────────────────────────────────

  Future<void> sendChallenge({
    required String recipientId,
    required String type,
    int durationDays = 7,
  }) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      final raw = await api.createChallenge(
        accessToken: token,
        recipientId: recipientId,
        type: type,
        durationDays: durationDays,
      );
      final challenge = ChallengeModel.fromJson(raw);
      state =
          state.copyWith(activeChallenges: [challenge, ...state.activeChallenges]);
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> acceptChallenge(String challengeId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      final raw = await api.acceptChallenge(
          accessToken: token, challengeId: challengeId);
      final updated = ChallengeModel.fromJson(raw);
      state = state.copyWith(
        activeChallenges: state.activeChallenges
            .map((c) => c.id == challengeId ? updated : c)
            .toList(),
      );
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> declineChallenge(String challengeId) async {
    try {
      final token = await _token();
      if (token == null) return;
      final api = _ref.read(apiServiceProvider);
      await api.declineChallenge(
          accessToken: token, challengeId: challengeId);
      state = state.copyWith(
        activeChallenges:
            state.activeChallenges.where((c) => c.id != challengeId).toList(),
      );
    } catch (e) {
      state = state.copyWith(error: '$e');
    }
  }

  Future<void> searchUsers(String query) async {
    if (query.length < 2) {
      state = state.copyWith(searchResults: [], isSearching: false);
      return;
    }
    state = state.copyWith(isSearching: true);
    try {
      final token = await _token();
      if (token == null) {
        state = state.copyWith(isSearching: false);
        return;
      }
      final api     = _ref.read(apiServiceProvider);
      final raw     = await api.searchUsers(accessToken: token, query: query);
      final results = raw.map(PublicUserProfile.fromJson).toList();
      state = state.copyWith(searchResults: results, isSearching: false);
    } catch (e) {
      state = state.copyWith(isSearching: false, error: '$e');
    }
  }

  void clearSearch() => state = state.copyWith(searchResults: []);

  void clearError() => state = state.copyWith(clearError: true);
}

// ── Provider ──────────────────────────────────────────────────────────────────

final friendsProvider =
    StateNotifierProvider<FriendsNotifier, FriendsState>(
  (ref) => FriendsNotifier(ref),
);
