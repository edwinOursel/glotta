"""
Friends router — /api/friends

Endpoints:
  GET    /api/friends                     List accepted friends with public stats
  GET    /api/friends/requests            Incoming pending friend requests
  GET    /api/friends/sent                Outgoing pending friend requests
  POST   /api/friends/request             Send a friend request  { addressee_id }
  POST   /api/friends/request/{id}/accept Accept an incoming request
  POST   /api/friends/request/{id}/decline Decline an incoming request
  DELETE /api/friends/{user_id}           Unfriend (removes accepted friendship)
  GET    /api/friends/search?q=...        Search users by username or email
  GET    /api/friends/leaderboard         Friends leaderboard ranked by vocab + mastery
"""

from datetime import datetime, timedelta, date as Date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, VocabularyItem, Friendship, LearningSession
from schemas import (
    FriendRequestBody, FriendshipResponse,
    PublicUserProfile, FriendsLeaderboardEntry,
)

router = APIRouter(prefix="/api/friends", tags=["friends"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _streak_from_dates(dates: list[Date]) -> int:
    """Compute consecutive daily streak from a list of activity dates (pure Python)."""
    if not dates:
        return 0
    unique = sorted(set(dates), reverse=True)
    today = datetime.utcnow().date()
    # No activity today or yesterday → streak is broken
    if unique[0] < today - timedelta(days=1):
        return 0
    streak = 0
    expected = unique[0]
    for d in unique:
        if d == expected:
            streak += 1
            expected = d - timedelta(days=1)
        else:
            break
    return streak


async def _batch_profiles(
    users: list[User],
    db: AsyncSession,
) -> dict[str, PublicUserProfile]:
    """
    Build PublicUserProfile for multiple users in 2 queries instead of N×3.

    Query 1: vocab size + mastered count per user (GROUP BY)
    Query 2: all relevant session ended_at dates for streak computation
    """
    if not users:
        return {}

    ids = [u.id for u in users]

    # 1. Vocab stats for all users in one aggregate query
    vocab_result = await db.execute(
        select(
            VocabularyItem.user_id,
            func.count(VocabularyItem.id).label("vocab_size"),
            func.count(VocabularyItem.id).filter(
                VocabularyItem.mastery_level >= 4
            ).label("words_mastered"),
        )
        .where(VocabularyItem.user_id.in_(ids))
        .group_by(VocabularyItem.user_id)
    )
    vocab_stats = {row.user_id: row for row in vocab_result.fetchall()}

    # 2. Session dates for streak — one query for all users
    sessions_result = await db.execute(
        select(LearningSession.user_id, LearningSession.ended_at)
        .where(
            LearningSession.user_id.in_(ids),
            LearningSession.ended_at.isnot(None),
        )
    )
    sessions_by_user: dict[str, list[Date]] = {uid: [] for uid in ids}
    for row in sessions_result.fetchall():
        sessions_by_user[row.user_id].append(row.ended_at.date())

    return {
        u.id: PublicUserProfile(
            id=u.id,
            username=u.username,
            jlpt_level=u.jlpt_level,
            vocabulary_size=int(getattr(vocab_stats.get(u.id), "vocab_size", 0) or 0),
            words_mastered=int(getattr(vocab_stats.get(u.id), "words_mastered", 0) or 0),
            current_streak_days=_streak_from_dates(sessions_by_user.get(u.id, [])),
        )
        for u in users
    }


async def _public_profile(user: User, db: AsyncSession) -> PublicUserProfile:
    """Single-user convenience wrapper around _batch_profiles."""
    profiles = await _batch_profiles([user], db)
    return profiles[user.id]


async def _friendship_between(
    user_a: str, user_b: str, db: AsyncSession
) -> Friendship | None:
    result = await db.execute(
        select(Friendship).where(
            or_(
                and_(Friendship.requester_id == user_a, Friendship.addressee_id == user_b),
                and_(Friendship.requester_id == user_b, Friendship.addressee_id == user_a),
            )
        )
    )
    return result.scalar_one_or_none()


async def _to_response(
    friendship: Friendship,
    current_user_id: str,
    db: AsyncSession,
) -> FriendshipResponse:
    """Single-friendship response (for mutation endpoints where N=1)."""
    other_id = (
        friendship.addressee_id
        if friendship.requester_id == current_user_id
        else friendship.requester_id
    )
    other_result = await db.execute(select(User).where(User.id == other_id))
    other = other_result.scalar_one()
    profile = await _public_profile(other, db)
    return FriendshipResponse(
        id=friendship.id,
        status=friendship.status,
        created_at=friendship.created_at,
        updated_at=friendship.updated_at,
        other_user=profile,
    )


async def _list_to_responses(
    friendships: list[Friendship],
    current_user_id: str,
    db: AsyncSession,
) -> list[FriendshipResponse]:
    """
    Convert a list of Friendships with batched profile loading.
    3 queries total regardless of list length (vs N×4 previously).
    """
    if not friendships:
        return []

    other_ids = [
        f.addressee_id if f.requester_id == current_user_id else f.requester_id
        for f in friendships
    ]
    users_result = await db.execute(select(User).where(User.id.in_(other_ids)))
    users_by_id = {u.id: u for u in users_result.scalars().all()}
    profiles = await _batch_profiles(list(users_by_id.values()), db)

    return [
        FriendshipResponse(
            id=f.id,
            status=f.status,
            created_at=f.created_at,
            updated_at=f.updated_at,
            other_user=profiles[
                f.addressee_id if f.requester_id == current_user_id else f.requester_id
            ],
        )
        for f in friendships
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[FriendshipResponse])
async def list_friends(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """List all accepted friends."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == current_user.id,
                Friendship.addressee_id == current_user.id,
            ),
        )
    )
    return await _list_to_responses(result.scalars().all(), current_user.id, db)


@router.get("/requests", response_model=list[FriendshipResponse])
async def incoming_requests(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Pending friend requests sent to me."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.addressee_id == current_user.id,
            Friendship.status == "pending",
        )
    )
    return await _list_to_responses(result.scalars().all(), current_user.id, db)


@router.get("/sent", response_model=list[FriendshipResponse])
async def sent_requests(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Pending friend requests I sent."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.requester_id == current_user.id,
            Friendship.status == "pending",
        )
    )
    return await _list_to_responses(result.scalars().all(), current_user.id, db)


@router.post("/request", response_model=FriendshipResponse, status_code=201)
async def send_friend_request(
    body: FriendRequestBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Send a friend request to another user."""
    if body.addressee_id == current_user.id:
        raise HTTPException(400, "Cannot send a friend request to yourself")

    target = await db.get(User, body.addressee_id)
    if target is None:
        raise HTTPException(404, "User not found")

    existing = await _friendship_between(current_user.id, body.addressee_id, db)
    if existing:
        if existing.status == "accepted":
            raise HTTPException(409, "Already friends")
        if existing.status == "pending":
            raise HTTPException(409, "Friend request already pending")
        # declined → allow re-send
        existing.status = "pending"
        existing.requester_id = current_user.id
        existing.addressee_id = body.addressee_id
        await db.commit()
        await db.refresh(existing)
        return await _to_response(existing, current_user.id, db)

    friendship = Friendship(
        requester_id=current_user.id,
        addressee_id=body.addressee_id,
    )
    db.add(friendship)
    await db.commit()
    await db.refresh(friendship)
    return await _to_response(friendship, current_user.id, db)


@router.post("/request/{friendship_id}/accept", response_model=FriendshipResponse)
async def accept_request(
    friendship_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Accept an incoming friend request."""
    friendship = await db.get(Friendship, friendship_id)
    if friendship is None:
        raise HTTPException(404, "Friend request not found")
    if friendship.addressee_id != current_user.id:
        raise HTTPException(403, "Not your request to accept")
    if friendship.status != "pending":
        raise HTTPException(409, f"Request is already {friendship.status}")

    friendship.status = "accepted"
    friendship.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(friendship)
    return await _to_response(friendship, current_user.id, db)


@router.post("/request/{friendship_id}/decline", response_model=FriendshipResponse)
async def decline_request(
    friendship_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Decline an incoming friend request."""
    friendship = await db.get(Friendship, friendship_id)
    if friendship is None:
        raise HTTPException(404, "Friend request not found")
    if friendship.addressee_id != current_user.id:
        raise HTTPException(403, "Not your request to decline")
    if friendship.status != "pending":
        raise HTTPException(409, f"Request is already {friendship.status}")

    friendship.status = "declined"
    friendship.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(friendship)
    return await _to_response(friendship, current_user.id, db)


@router.delete("/{user_id}", status_code=204)
async def unfriend(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Remove an accepted friendship."""
    friendship = await _friendship_between(current_user.id, user_id, db)
    if friendship is None or friendship.status != "accepted":
        raise HTTPException(404, "Friendship not found")
    await db.delete(friendship)
    await db.commit()


@router.get("/search", response_model=list[PublicUserProfile])
async def search_users(
    q: str             = Query(..., min_length=2),
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Search users by username or email (partial match, case-insensitive)."""
    pattern = f"%{q.lower()}%"
    result = await db.execute(
        select(User).where(
            User.id != current_user.id,
            or_(
                func.lower(User.username).like(pattern),
                func.lower(User.email).like(pattern),
            ),
        ).limit(20)
    )
    users = result.scalars().all()
    profiles = await _batch_profiles(list(users), db)
    return [profiles[u.id] for u in users]


@router.get("/leaderboard", response_model=list[FriendsLeaderboardEntry])
async def leaderboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """
    Leaderboard of accepted friends + self, ranked by:
      1. vocabulary_size DESC
      2. words_mastered DESC
    """
    result = await db.execute(
        select(Friendship).where(
            Friendship.status == "accepted",
            or_(
                Friendship.requester_id == current_user.id,
                Friendship.addressee_id == current_user.id,
            ),
        )
    )
    friendships = result.scalars().all()
    friend_ids = {
        f.addressee_id if f.requester_id == current_user.id else f.requester_id
        for f in friendships
    }
    friend_ids.add(current_user.id)

    users_result = await db.execute(select(User).where(User.id.in_(friend_ids)))
    profiles_dict = await _batch_profiles(list(users_result.scalars().all()), db)

    profiles = sorted(
        profiles_dict.values(),
        key=lambda p: (-p.vocabulary_size, -p.words_mastered),
    )
    return [
        FriendsLeaderboardEntry(rank=i + 1, user=p, is_self=(p.id == current_user.id))
        for i, p in enumerate(profiles)
    ]
