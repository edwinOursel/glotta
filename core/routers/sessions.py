"""
Learning sessions router — start/end sessions, progress summary.
"""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from database import get_db
from models import User, LearningSession, VocabularyItem
from auth import get_current_user
from utils import streak_from_dates
from schemas import (
    StartSessionRequest, EndSessionRequest,
    SessionResponse, ProgressResponse,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/start", response_model=SessionResponse)
async def start_session(
    body:         StartSessionRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    session = LearningSession(
        user_id         = current_user.id,
        session_type    = body.session_type,
        constraint_mode = body.constraint_mode,
    )
    db.add(session)
    await db.flush()
    return session


@router.post("/end", response_model=SessionResponse)
async def end_session(
    body:         EndSessionRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(LearningSession).where(
            and_(LearningSession.id == body.session_id,
                 LearningSession.user_id == current_user.id)
        )
    )
    session: LearningSession | None = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    now = datetime.now(timezone.utc)
    session.ended_at        = now
    session.duration_secs   = int((now - session.started_at).total_seconds())
    session.words_practiced = body.words_practiced
    session.words_correct   = body.words_correct
    db.add(session)
    return session


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 20,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(LearningSession)
        .where(LearningSession.user_id == current_user.id)
        .order_by(LearningSession.started_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/progress", response_model=ProgressResponse)
async def get_progress(
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    uid = current_user.id

    # Aggregate session stats
    agg = await db.execute(
        select(
            func.count().label("total_sessions"),
            func.coalesce(func.sum(LearningSession.words_practiced), 0).label("total_words_practiced"),
            func.coalesce(func.sum(LearningSession.words_correct),   0).label("total_words_correct"),
        ).where(LearningSession.user_id == uid)
    )
    row = agg.one()
    total_sessions       = row.total_sessions
    total_words_practiced = row.total_words_practiced
    total_words_correct   = row.total_words_correct
    accuracy = round(total_words_correct / total_words_practiced * 100, 1) if total_words_practiced else 0.0

    # Vocabulary stats
    vocab_count_r = await db.execute(
        select(func.count()).where(VocabularyItem.user_id == uid)
    )
    vocab_size = vocab_count_r.scalar_one()

    mastered_r = await db.execute(
        select(func.count()).where(
            and_(VocabularyItem.user_id == uid, VocabularyItem.mastery_level == 5)
        )
    )
    words_mastered = mastered_r.scalar_one()

    # Streak: count consecutive days with at least one session
    sessions_r = await db.execute(
        select(LearningSession.started_at)
        .where(LearningSession.user_id == uid)
        .order_by(LearningSession.started_at.desc())
        .limit(365)
    )
    streak = streak_from_dates(list(sessions_r.scalars()))

    return ProgressResponse(
        total_sessions        = total_sessions,
        total_words_practiced = total_words_practiced,
        total_words_correct   = total_words_correct,
        accuracy_percent      = accuracy,
        vocabulary_size       = vocab_size,
        words_mastered        = words_mastered,
        current_streak_days   = streak,
    )
