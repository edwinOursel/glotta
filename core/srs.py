"""
SM-2 Spaced Repetition System algorithm.

Original paper: Wozniak, P.A. (1990) — https://www.supermemo.com/en/archives1990-2015/english/ol/sm2

Quality scale (passed by the caller / mobile app):
    5 — perfect response
    4 — correct after slight hesitation
    3 — correct with serious difficulty
    2 — incorrect, but the answer felt familiar
    1 — incorrect, answer was hard to recall
    0 — complete blackout

Rules:
    quality >= 3  → correct; interval grows, repetition counter increments
    quality <  3  → incorrect; repetition resets to 0, interval back to 1

Ease factor stays in [1.3, ∞), updated after every review.
"""

from datetime import datetime, timedelta, timezone
from models import VocabularyItem


def sm2_review(item: VocabularyItem, quality: int) -> VocabularyItem:
    """
    Apply one SM-2 review to a VocabularyItem in-place and return it.

    Args:
        item:    The word being reviewed (mutated directly).
        quality: Integer 0-5 describing how well the user remembered it.

    Returns:
        The mutated item (convenient for chaining).
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be 0-5, got {quality}")

    item.times_seen += 1

    if quality >= 3:
        # ── Correct answer ─────────────────────────────────────────────
        item.times_correct += 1

        if item.repetition == 0:
            item.interval = 1
        elif item.repetition == 1:
            item.interval = 6
        else:
            item.interval = round(item.interval * item.ease_factor)

        item.repetition += 1

    else:
        # ── Incorrect answer — reset streak ────────────────────────────
        item.repetition = 0
        item.interval   = 1

    # ── Ease-factor update (always) ────────────────────────────────────
    # Formula keeps EF ≥ 1.3 to prevent reviews from becoming daily grinds.
    item.ease_factor = max(
        1.3,
        item.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02),
    )

    # ── Mastery level 0-5 (coarse proxy from repetition streak) ────────
    item.mastery_level = min(5, item.repetition // 2)

    # ── Schedule next review ───────────────────────────────────────────
    item.next_review_at = datetime.now(timezone.utc) + timedelta(days=item.interval)

    return item


def words_due(items: list[VocabularyItem], limit: int = 20) -> list[VocabularyItem]:
    """
    Return items whose next_review_at is now or in the past,
    sorted by most overdue first, capped at *limit*.
    """
    now = datetime.now(timezone.utc)
    due = [w for w in items if w.next_review_at <= now]
    due.sort(key=lambda w: w.next_review_at)
    return due[:limit]
