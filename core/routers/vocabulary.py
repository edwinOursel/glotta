"""
Vocabulary router — per-user CRUD + SRS review queue.

Endpoints:
  GET    /api/vocabulary              — list (filtered by level, search)
  POST   /api/vocabulary              — add one word
  POST   /api/vocabulary/batch        — add many words
  POST   /api/vocabulary/seed/{level} — seed from the JLPT JSON dataset
  GET    /api/vocabulary/due          — words due for review today
  GET    /api/vocabulary/stats        — counts, mastery distribution
  GET    /api/vocabulary/{id}         — single word
  PATCH  /api/vocabulary/{id}         — update metadata
  DELETE /api/vocabulary/{id}         — delete
  POST   /api/vocabulary/{id}/review  — submit SM-2 review result
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from database import get_db
from models import User, VocabularyItem
from auth import get_current_user
from srs import sm2_review, words_due
from schemas import (
    VocabularyItemResponse, AddWordRequest, AddWordsRequest,
    UpdateWordRequest, SeedLevelRequest,
    ReviewRequest, ReviewResponse,
    VocabularyStatsResponse,
)

router = APIRouter(prefix="/api/vocabulary", tags=["vocabulary"])

# Path to the bundled JLPT JSON dataset
JLPT_DATA_PATH = Path(__file__).parent.parent / "data" / "jlpt_vocabulary.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_word_or_404(word_id: str, user_id: str, db: AsyncSession) -> VocabularyItem:
    result = await db.execute(
        select(VocabularyItem).where(
            and_(VocabularyItem.id == word_id, VocabularyItem.user_id == user_id)
        )
    )
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Word not found")
    return item


# ── List / search ─────────────────────────────────────────────────────────────

@router.get("", response_model=List[VocabularyItemResponse])
async def list_vocabulary(
    level:  Optional[str] = Query(None, description="Filter by JLPT level (N1-N5)"),
    search: Optional[str] = Query(None, description="Search word / reading / meaning"),
    limit:  int           = Query(50,  ge=1, le=500),
    offset: int           = Query(0,   ge=0),
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    q = select(VocabularyItem).where(VocabularyItem.user_id == current_user.id)

    if level:
        q = q.where(VocabularyItem.jlpt_level == level)

    if search:
        # Escape LIKE special characters so user input is treated as a literal string
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        q = q.where(
            VocabularyItem.word.ilike(pattern, escape="\\")
            | VocabularyItem.reading.ilike(pattern, escape="\\")
            | VocabularyItem.meaning.ilike(pattern, escape="\\")
        )

    q = q.order_by(VocabularyItem.date_added.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=VocabularyStatsResponse)
async def vocabulary_stats(
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    uid = current_user.id

    # Total
    total_r = await db.execute(
        select(func.count()).where(VocabularyItem.user_id == uid)
    )
    total = total_r.scalar_one()

    # By level
    level_r = await db.execute(
        select(VocabularyItem.jlpt_level, func.count())
        .where(VocabularyItem.user_id == uid)
        .group_by(VocabularyItem.jlpt_level)
    )
    by_level = {row[0] or "unknown": row[1] for row in level_r.all()}

    # Due today
    now = datetime.now(timezone.utc)
    due_r = await db.execute(
        select(func.count()).where(
            and_(VocabularyItem.user_id == uid, VocabularyItem.next_review_at <= now)
        )
    )
    due_today = due_r.scalar_one()

    # Mastery distribution
    mastery_r = await db.execute(
        select(VocabularyItem.mastery_level, func.count())
        .where(VocabularyItem.user_id == uid)
        .group_by(VocabularyItem.mastery_level)
    )
    mastery_dist = {str(row[0]): row[1] for row in mastery_r.all()}

    return VocabularyStatsResponse(
        total=total,
        by_level=by_level,
        due_today=due_today,
        mastery_distribution=mastery_dist,
    )


# ── SRS review queue ──────────────────────────────────────────────────────────

@router.get("/due", response_model=List[VocabularyItemResponse])
async def get_due_words(
    limit:        int           = Query(20, ge=1, le=100),
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Return words whose next_review_at ≤ now, oldest first."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(VocabularyItem)
        .where(
            and_(VocabularyItem.user_id == current_user.id,
                 VocabularyItem.next_review_at <= now)
        )
        .order_by(VocabularyItem.next_review_at)
        .limit(limit)
    )
    return result.scalars().all()


# ── Single word ───────────────────────────────────────────────────────────────

@router.get("/{word_id}", response_model=VocabularyItemResponse)
async def get_word(
    word_id: str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    return await _get_word_or_404(word_id, current_user.id, db)


# ── Add one word ──────────────────────────────────────────────────────────────

@router.post("", response_model=VocabularyItemResponse, status_code=status.HTTP_201_CREATED)
async def add_word(
    body:         AddWordRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    # Idempotent: return existing if already in vocabulary
    existing = await db.execute(
        select(VocabularyItem).where(
            and_(VocabularyItem.user_id == current_user.id,
                 VocabularyItem.word == body.word)
        )
    )
    if (item := existing.scalar_one_or_none()):
        return item

    item = VocabularyItem(user_id=current_user.id, **body.model_dump())
    db.add(item)
    await db.flush()
    return item


# ── Add many words ────────────────────────────────────────────────────────────

@router.post("/batch", status_code=status.HTTP_201_CREATED)
async def add_words_batch(
    body:         AddWordsRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    added = 0
    skipped = 0
    for w in body.words:
        existing = await db.execute(
            select(VocabularyItem).where(
                and_(VocabularyItem.user_id == current_user.id,
                     VocabularyItem.word == w.word)
            )
        )
        if existing.scalar_one_or_none():
            skipped += 1
            continue
        db.add(VocabularyItem(user_id=current_user.id, **w.model_dump()))
        added += 1

    await db.flush()
    return {"added": added, "skipped": skipped}


# ── Seed from JLPT dataset ────────────────────────────────────────────────────

@router.post("/seed/{jlpt_level}", status_code=status.HTTP_201_CREATED)
async def seed_from_jlpt(
    jlpt_level:   str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Import all words for a given JLPT level from the bundled dataset."""
    if jlpt_level not in {"N1", "N2", "N3", "N4", "N5"}:
        raise HTTPException(status_code=400, detail="level must be N1-N5")

    if not JLPT_DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="JLPT dataset not found on server")

    with open(JLPT_DATA_PATH, encoding="utf-8") as f:
        dataset: dict = json.load(f)

    words_for_level = dataset.get(jlpt_level, [])
    if not words_for_level:
        raise HTTPException(status_code=404, detail=f"No words found for {jlpt_level}")

    added = 0
    skipped = 0
    for entry in words_for_level:
        existing = await db.execute(
            select(VocabularyItem).where(
                and_(VocabularyItem.user_id == current_user.id,
                     VocabularyItem.word == entry["word"])
            )
        )
        if existing.scalar_one_or_none():
            skipped += 1
            continue

        db.add(VocabularyItem(
            user_id        = current_user.id,
            word           = entry["word"],
            reading        = entry.get("reading"),
            meaning        = entry.get("meaning"),
            part_of_speech = entry.get("part_of_speech"),
            jlpt_level     = jlpt_level,
        ))
        added += 1

    await db.flush()
    return {"level": jlpt_level, "added": added, "skipped": skipped}


# ── Update metadata ───────────────────────────────────────────────────────────

@router.patch("/{word_id}", response_model=VocabularyItemResponse)
async def update_word(
    word_id: str,
    body:    UpdateWordRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    item = await _get_word_or_404(word_id, current_user.id, db)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(item, field, value)
    db.add(item)
    return item


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{word_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_word(
    word_id: str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    item = await _get_word_or_404(word_id, current_user.id, db)
    await db.delete(item)


# ── SRS review ────────────────────────────────────────────────────────────────

@router.post("/{word_id}/review", response_model=ReviewResponse)
async def review_word(
    word_id: str,
    body:    ReviewRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Submit a review result for a word.

    quality 0-5:
        5 — perfect recall
        4 — correct, slight hesitation
        3 — correct with difficulty
        2 — incorrect, felt familiar
        1 — incorrect, hard to recall
        0 — complete blackout
    """
    item = await _get_word_or_404(word_id, current_user.id, db)
    sm2_review(item, body.quality)
    db.add(item)
    return ReviewResponse(
        word_id        = item.id,
        word           = item.word,
        interval       = item.interval,
        ease_factor    = item.ease_factor,
        next_review_at = item.next_review_at,
        mastery_level  = item.mastery_level,
    )
