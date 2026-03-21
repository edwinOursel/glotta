"""Shared utilities."""

from datetime import datetime, timedelta, timezone
from typing import List


def streak_from_dates(dates: List[datetime]) -> int:
    """
    Compute consecutive daily streak from a list of activity datetimes.

    A streak counts today and every preceding day that has at least one entry.
    If the most recent entry is older than yesterday the streak is 0.
    """
    if not dates:
        return 0
    unique = sorted({d.date() for d in dates}, reverse=True)
    today = datetime.now(timezone.utc).date()
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
