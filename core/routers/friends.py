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

from datetime import datetime
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

async def _public_profile(user: User, db: AsyncSession) -> PublicUserProfile:
    """Build a PublicUserProfile by querying stats for the given user."""
    vocab_q = await db.execute(
        select(func.count(VocabularyItem.id)).where(VocabularyItem.user_id == user.id)
    )
    vocab_size = vocab_q.scalar() or 0

    mastered_q = await db.execute(
        select(func.count(VocabularyItem.id)).where(
            VocabularyItem.user_id == user.id,
            VocabularyItem.mastery_level >= 4,
        )
    )
    words_mastered = mastered_q.scalar() or 0

    # Streak: count consecutive days with at least one session ending
    streak = await _compute_streak(user.id, db)

    return PublicUserProfile(
        id=user.id,
        username=user.username,
        jlpt_level=user.jlpt_level,
        vocabulary_size=vocab_size,
        words_mastered=words_mastered,
        current_streak_days=streak,
    )


async def _compute_streak(user_id: str, db: AsyncSession) -> int:
    """Approximate daily streak from learning_sessions.ended_at."""
    result = await db.execute(
        select(LearningSession.ended_at)
        .where(
            LearningSession.user_id == user_id,
            LearningSession.ended_at.isnot(None),
        )
        .order_by(LearningSession.ended_at.desc())
    )
    dates = [row[0].date() for row in result.fetchall()]
    if not dates:
        return 0
    unique = sorted(set(dates), reverse=True)
    today = datetime.utcnow().date()
    streak = 0
    expected = today
    for d in unique:
        if d == expected or d == today:
            streak += 1
            expected = d.replace(day=d.day - 1) if d.day > 1 else d  # simple decrement
            from datetime import timedelta
            expected = d - timedelta(days=1)
        else:
            break
    return streak


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
    friendships = result.scalars().all()
    return [await _to_response(f, current_user.id, db) for f in friendships]


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
    friendships = result.scalars().all()
    return [await _to_response(f, current_user.id, db) for f in friendships]


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
    friendships = result.scalars().all()
    return [await _to_response(f, current_user.id, db) for f in friendships]


@router.post("/request", response_model=FriendshipResponse, status_code=201)
async def send_friend_request(
    body: FriendRequestBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Send a friend request to another user."""
    if body.addressee_id == current_user.id:
        raise HTTPException(400, "Cannot send a friend request to yourself")

    # Check target user exists
    target = await db.get(User, body.addressee_id)
    if target is None:
        raise HTTPException(404, "User not found")

    # Check no existing relationship
    existing = await _friendship_between(current_user.id, body.addressee_id, db)
    if existing:
        if existing.status == "accepted":
            raise HTTPException(409, "Already friends")
        if existing.status == "pending":
            raise HTTPException(409, "Friend request already pending")
        # declined → allow re-send by updating status
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
    return [await _public_profile(u, db) for u in users]


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
    # Collect friend IDs
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

    # Build profiles
    profiles: list[PublicUserProfile] = []
    for uid in friend_ids:
        user = await db.get(User, uid)
        if user:
            profiles.append(await _public_profile(user, db))

    # Rank by vocabulary_size then words_mastered
    profiles.sort(key=lambda p: (-p.vocabulary_size, -p.words_mastered))

    return [
        FriendsLeaderboardEntry(
            rank=i + 1,
            user=p,
            is_self=(p.id == current_user.id),
        )
        for i, p in enumerate(profiles)
    ]
