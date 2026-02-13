"""
Challenges router — /api/challenges

Endpoints:
  GET    /api/challenges              My active/pending challenges
  GET    /api/challenges/history      Completed/expired/declined challenges
  GET    /api/challenges/{id}         Detail with live scores
  POST   /api/challenges              Create a challenge {recipient_id, type, duration_days}
  POST   /api/challenges/{id}/accept  Accept a pending challenge
  POST   /api/challenges/{id}/decline Decline a pending challenge

Score semantics (computed live):
  vocab_sprint  — words added between starts_at and ends_at
  mastery_race  — words with mastery_level >= 4 at query time
  accuracy_duel — avg accuracy (times_correct / times_seen) on reviewed words
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from models import User, VocabularyItem, Challenge, Friendship
from schemas import ChallengeCreateRequest, ChallengeResponse, PublicUserProfile
from routers.friends import _public_profile, _friendship_between

router = APIRouter(prefix="/api/challenges", tags=["challenges"])


# ── Score helpers ──────────────────────────────────────────────────────────────

async def _compute_score(
    user_id: str,
    challenge_type: str,
    starts_at: Optional[datetime],
    ends_at: Optional[datetime],
    db: AsyncSession,
) -> float:
    if challenge_type == "vocab_sprint":
        # Words added during the challenge window
        q = select(func.count(VocabularyItem.id)).where(
            VocabularyItem.user_id == user_id,
            VocabularyItem.date_added >= starts_at,
            VocabularyItem.date_added <= (ends_at or datetime.utcnow()),
        )
        result = await db.execute(q)
        return float(result.scalar() or 0)

    elif challenge_type == "mastery_race":
        # Current count of words with mastery_level >= 4
        q = select(func.count(VocabularyItem.id)).where(
            VocabularyItem.user_id == user_id,
            VocabularyItem.mastery_level >= 4,
        )
        result = await db.execute(q)
        return float(result.scalar() or 0)

    elif challenge_type == "accuracy_duel":
        # Average accuracy on reviewed words (times_seen > 0)
        q = select(
            func.avg(
                VocabularyItem.times_correct * 1.0 / VocabularyItem.times_seen
            )
        ).where(
            VocabularyItem.user_id == user_id,
            VocabularyItem.times_seen > 0,
        )
        result = await db.execute(q)
        val = result.scalar()
        return round(float(val or 0) * 100, 1)  # as percentage

    return 0.0


async def _finalize_if_expired(challenge: Challenge, db: AsyncSession) -> Challenge:
    """If the challenge window has passed, compute final scores and set winner."""
    if challenge.status != "active":
        return challenge
    now = datetime.utcnow()
    if challenge.ends_at is None or now < challenge.ends_at:
        return challenge

    sender_score = await _compute_score(
        challenge.sender_id, challenge.type,
        challenge.starts_at, challenge.ends_at, db
    )
    recipient_score = await _compute_score(
        challenge.recipient_id, challenge.type,
        challenge.starts_at, challenge.ends_at, db
    )

    challenge.sender_score    = sender_score
    challenge.recipient_score = recipient_score
    if sender_score > recipient_score:
        challenge.winner_id = challenge.sender_id
    elif recipient_score > sender_score:
        challenge.winner_id = challenge.recipient_id
    else:
        challenge.winner_id = None  # tie
    challenge.status = "completed"
    await db.commit()
    await db.refresh(challenge)
    return challenge


async def _to_response(
    challenge: Challenge,
    current_user_id: str,
    db: AsyncSession,
) -> ChallengeResponse:
    challenge = await _finalize_if_expired(challenge, db)

    sender_user    = await db.get(User, challenge.sender_id)
    recipient_user = await db.get(User, challenge.recipient_id)
    sender_profile    = await _public_profile(sender_user, db)
    recipient_profile = await _public_profile(recipient_user, db)

    # Live scores for active challenges
    sender_score    = challenge.sender_score
    recipient_score = challenge.recipient_score
    if challenge.status == "active":
        sender_score    = await _compute_score(
            challenge.sender_id, challenge.type,
            challenge.starts_at, challenge.ends_at, db
        )
        recipient_score = await _compute_score(
            challenge.recipient_id, challenge.type,
            challenge.starts_at, challenge.ends_at, db
        )

    return ChallengeResponse(
        id=challenge.id,
        type=challenge.type,
        status=challenge.status,
        duration_days=challenge.duration_days,
        starts_at=challenge.starts_at,
        ends_at=challenge.ends_at,
        created_at=challenge.created_at,
        sender=sender_profile,
        recipient=recipient_profile,
        sender_score=sender_score,
        recipient_score=recipient_score,
        winner_id=challenge.winner_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ChallengeResponse])
async def list_active_challenges(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """List my pending and active challenges."""
    result = await db.execute(
        select(Challenge).where(
            Challenge.status.in_(["pending", "active"]),
            (Challenge.sender_id == current_user.id)
            | (Challenge.recipient_id == current_user.id),
        ).order_by(Challenge.created_at.desc())
    )
    challenges = result.scalars().all()
    return [await _to_response(c, current_user.id, db) for c in challenges]


@router.get("/history", response_model=list[ChallengeResponse])
async def list_past_challenges(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """List completed, declined, and expired challenges."""
    result = await db.execute(
        select(Challenge).where(
            Challenge.status.in_(["completed", "declined", "expired"]),
            (Challenge.sender_id == current_user.id)
            | (Challenge.recipient_id == current_user.id),
        ).order_by(Challenge.created_at.desc()).limit(50)
    )
    challenges = result.scalars().all()
    return [await _to_response(c, current_user.id, db) for c in challenges]


@router.get("/{challenge_id}", response_model=ChallengeResponse)
async def get_challenge(
    challenge_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    challenge = await db.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(404, "Challenge not found")
    if current_user.id not in (challenge.sender_id, challenge.recipient_id):
        raise HTTPException(403, "Not your challenge")
    return await _to_response(challenge, current_user.id, db)


@router.post("", response_model=ChallengeResponse, status_code=201)
async def create_challenge(
    body: ChallengeCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Send a challenge to a friend."""
    if body.recipient_id == current_user.id:
        raise HTTPException(400, "Cannot challenge yourself")

    # Must be friends
    friendship = await _friendship_between(current_user.id, body.recipient_id, db)
    if friendship is None or friendship.status != "accepted":
        raise HTTPException(403, "You can only challenge friends")

    # Check no already-active challenge of same type between these two
    existing = await db.execute(
        select(Challenge).where(
            Challenge.type == body.type,
            Challenge.status.in_(["pending", "active"]),
            (
                (Challenge.sender_id == current_user.id)
                & (Challenge.recipient_id == body.recipient_id)
            )
            | (
                (Challenge.sender_id == body.recipient_id)
                & (Challenge.recipient_id == current_user.id)
            ),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, "An active challenge of this type already exists")

    challenge = Challenge(
        sender_id=current_user.id,
        recipient_id=body.recipient_id,
        type=body.type,
        duration_days=body.duration_days,
    )
    db.add(challenge)
    await db.commit()
    await db.refresh(challenge)
    return await _to_response(challenge, current_user.id, db)


@router.post("/{challenge_id}/accept", response_model=ChallengeResponse)
async def accept_challenge(
    challenge_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    challenge = await db.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(404, "Challenge not found")
    if challenge.recipient_id != current_user.id:
        raise HTTPException(403, "Not your challenge to accept")
    if challenge.status != "pending":
        raise HTTPException(409, f"Challenge is already {challenge.status}")

    now = datetime.utcnow()
    challenge.status    = "active"
    challenge.starts_at = now
    challenge.ends_at   = now + timedelta(days=challenge.duration_days)
    await db.commit()
    await db.refresh(challenge)
    return await _to_response(challenge, current_user.id, db)


@router.post("/{challenge_id}/decline", response_model=ChallengeResponse)
async def decline_challenge(
    challenge_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    challenge = await db.get(Challenge, challenge_id)
    if challenge is None:
        raise HTTPException(404, "Challenge not found")
    if challenge.recipient_id != current_user.id:
        raise HTTPException(403, "Not your challenge to decline")
    if challenge.status != "pending":
        raise HTTPException(409, f"Challenge is already {challenge.status}")

    challenge.status = "declined"
    await db.commit()
    await db.refresh(challenge)
    return await _to_response(challenge, current_user.id, db)
