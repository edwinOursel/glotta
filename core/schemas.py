"""
Pydantic schemas — request bodies and response shapes for every router.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: Optional[str] = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── User / Profile ────────────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    id: str
    email: str
    username: Optional[str]
    jlpt_level: str
    constraint_mode: str
    created_at: datetime
    last_active: datetime


class UpdateProfileRequest(BaseModel):
    username: Optional[str] = Field(default=None, max_length=64)
    jlpt_level: Optional[str] = None
    constraint_mode: Optional[str] = None

    @field_validator("jlpt_level")
    @classmethod
    def validate_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"N1", "N2", "N3", "N4", "N5"}:
            raise ValueError("jlpt_level must be N1-N5")
        return v

    @field_validator("constraint_mode")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"hard", "soft", "adaptive"}:
            raise ValueError("constraint_mode must be hard | soft | adaptive")
        return v


# ── Vocabulary ────────────────────────────────────────────────────────────────

class VocabularyItemResponse(BaseModel):
    id: str
    word: str
    reading: Optional[str]
    meaning: Optional[str]
    part_of_speech: Optional[str]
    jlpt_level: Optional[str]
    notes: Optional[str]
    # SRS
    repetition: int
    interval: int
    ease_factor: float
    next_review_at: datetime
    # Stats
    times_seen: int
    times_correct: int
    mastery_level: int
    date_added: datetime


class AddWordRequest(BaseModel):
    word: str = Field(max_length=100)
    reading: Optional[str] = Field(default=None, max_length=200)
    meaning: Optional[str] = Field(default=None, max_length=500)
    part_of_speech: Optional[str] = Field(default=None, max_length=50)
    jlpt_level: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


class AddWordsRequest(BaseModel):
    words: List[AddWordRequest]


class UpdateWordRequest(BaseModel):
    reading: Optional[str] = Field(default=None, max_length=200)
    meaning: Optional[str] = Field(default=None, max_length=500)
    part_of_speech: Optional[str] = Field(default=None, max_length=50)
    jlpt_level: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


class SeedLevelRequest(BaseModel):
    jlpt_level: str

    @field_validator("jlpt_level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        if v not in {"N1", "N2", "N3", "N4", "N5"}:
            raise ValueError("jlpt_level must be N1-N5")
        return v


class ReviewRequest(BaseModel):
    quality: int  # 0-5

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, v: int) -> int:
        if not 0 <= v <= 5:
            raise ValueError("quality must be 0-5")
        return v


class ReviewResponse(BaseModel):
    word_id: str
    word: str
    interval: int
    ease_factor: float
    next_review_at: datetime
    mastery_level: int


class VocabularyStatsResponse(BaseModel):
    total: int
    by_level: dict
    due_today: int
    mastery_distribution: dict  # {0: n, 1: n, …, 5: n}


# ── Learning Sessions ─────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    session_type: str = "generate"
    constraint_mode: Optional[str] = None


class EndSessionRequest(BaseModel):
    session_id: str
    words_practiced: int = 0
    words_correct: int = 0


class SessionResponse(BaseModel):
    id: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_secs: Optional[int]
    words_practiced: int
    words_correct: int
    constraint_mode: Optional[str]
    session_type: str


class ProgressResponse(BaseModel):
    total_sessions: int
    total_words_practiced: int
    total_words_correct: int
    accuracy_percent: float
    vocabulary_size: int
    words_mastered: int          # mastery_level == 5
    current_streak_days: int


# ── Friends ────────────────────────────────────────────────────────────────────

class FriendRequestBody(BaseModel):
    """Send a friend request to a user by their ID."""
    addressee_id: str


class PublicUserProfile(BaseModel):
    """Public-facing subset of a user's profile."""
    id: str
    username: Optional[str]
    jlpt_level: str
    # public stats
    vocabulary_size: int
    words_mastered: int
    current_streak_days: int


class FriendshipResponse(BaseModel):
    id: str
    status: str                    # pending | accepted | declined
    created_at: datetime
    updated_at: datetime
    # the other party
    other_user: PublicUserProfile


class FriendsLeaderboardEntry(BaseModel):
    rank: int
    user: PublicUserProfile
    is_self: bool


# ── Challenges ────────────────────────────────────────────────────────────────

CHALLENGE_TYPES = {"vocab_sprint", "mastery_race", "accuracy_duel"}


class ChallengeCreateRequest(BaseModel):
    recipient_id: str
    type: str           # vocab_sprint | mastery_race | accuracy_duel
    duration_days: int = 7

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in CHALLENGE_TYPES:
            raise ValueError(f"type must be one of: {', '.join(CHALLENGE_TYPES)}")
        return v

    @field_validator("duration_days")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if not 1 <= v <= 30:
            raise ValueError("duration_days must be 1-30")
        return v


class ChallengeResponse(BaseModel):
    id: str
    type: str
    status: str
    duration_days: int
    starts_at: Optional[datetime]
    ends_at: Optional[datetime]
    created_at: datetime
    sender: PublicUserProfile
    recipient: PublicUserProfile
    sender_score: Optional[float]
    recipient_score: Optional[float]
    winner_id: Optional[str]
