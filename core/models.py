"""
SQLAlchemy ORM models.

User → owns → VocabularyItem (with SRS fields)
User → owns → LearningSession
User → owns → GeneratedText
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id              = Column(String, primary_key=True, default=_uuid)
    email           = Column(String, unique=True, nullable=False, index=True)
    password_hash   = Column(String, nullable=False)
    username        = Column(String, nullable=True)
    jlpt_level      = Column(String, default="N5")
    constraint_mode = Column(String, default="soft")
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_active     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    vocabulary       = relationship("VocabularyItem",  back_populates="user", cascade="all, delete-orphan")
    sessions         = relationship("LearningSession", back_populates="user", cascade="all, delete-orphan")
    generated_texts  = relationship("GeneratedText",   back_populates="user", cascade="all, delete-orphan")
    sent_requests    = relationship("Friendship", foreign_keys="Friendship.requester_id", cascade="all, delete-orphan")
    received_requests = relationship("Friendship", foreign_keys="Friendship.addressee_id", cascade="all, delete-orphan")


class VocabularyItem(Base):
    """
    One word in a user's personal vocabulary, augmented with SM-2 SRS fields.

    SM-2 fields:
        repetition   — consecutive correct answers (resets to 0 on failure)
        interval     — days until next review
        ease_factor  — difficulty multiplier, min 1.3, starts at 2.5
        next_review_at — next scheduled review date
    """
    __tablename__ = "vocabulary"
    __table_args__ = (UniqueConstraint("user_id", "word", name="uq_user_word"),)

    id             = Column(String, primary_key=True, default=_uuid)
    user_id        = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Lexical data
    word           = Column(String, nullable=False)
    reading        = Column(String)
    meaning        = Column(String)
    part_of_speech = Column(String)
    jlpt_level     = Column(String)
    notes          = Column(Text)

    # SM-2 spaced repetition
    repetition     = Column(Integer, default=0)
    interval       = Column(Integer, default=1)       # days
    ease_factor    = Column(Float,   default=2.5)
    next_review_at = Column(DateTime, default=datetime.utcnow)

    # Stats
    times_seen     = Column(Integer, default=0)
    times_correct  = Column(Integer, default=0)
    mastery_level  = Column(Integer, default=0)  # 0-5
    date_added     = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="vocabulary")


class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id              = Column(String, primary_key=True, default=_uuid)
    user_id         = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at      = Column(DateTime, default=datetime.utcnow)
    ended_at        = Column(DateTime, nullable=True)
    duration_secs   = Column(Integer, nullable=True)
    words_practiced = Column(Integer, default=0)
    words_correct   = Column(Integer, default=0)
    constraint_mode = Column(String)
    session_type    = Column(String, default="generate")  # generate | review | quiz

    user = relationship("User", back_populates="sessions")


class GeneratedText(Base):
    __tablename__ = "generated_texts"

    id             = Column(String, primary_key=True, default=_uuid)
    user_id        = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    text           = Column(Text, nullable=False)
    prompt         = Column(Text, nullable=False)
    constraint_mode = Column(String)
    intent         = Column(String)
    created_at     = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="generated_texts")


class Friendship(Base):
    """
    Bidirectional friend relationship.
    requester sends the request; addressee accepts/declines.
    status: 'pending' | 'accepted' | 'declined'
    """
    __tablename__ = "friendships"
    __table_args__ = (
        UniqueConstraint("requester_id", "addressee_id", name="uq_friendship"),
    )

    id           = Column(String, primary_key=True, default=_uuid)
    requester_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    addressee_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status       = Column(String, default="pending")   # pending | accepted | declined
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    requester = relationship("User", foreign_keys=[requester_id])
    addressee = relationship("User", foreign_keys=[addressee_id])


class Challenge(Base):
    """
    Peer-to-peer learning challenge.
    type: 'vocab_sprint' | 'mastery_race' | 'accuracy_duel'
    status: 'pending' | 'active' | 'completed' | 'declined' | 'expired'

    Score semantics:
      vocab_sprint  — words added between starts_at and ends_at
      mastery_race  — words with mastery_level >= 4 at ends_at
      accuracy_duel — avg accuracy (times_correct / times_seen) on reviewed words at ends_at
    """
    __tablename__ = "challenges"

    id               = Column(String, primary_key=True, default=_uuid)
    sender_id        = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    recipient_id     = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type             = Column(String, nullable=False)   # vocab_sprint | mastery_race | accuracy_duel
    status           = Column(String, default="pending")
    duration_days    = Column(Integer, default=7)
    starts_at        = Column(DateTime, nullable=True)  # set when accepted
    ends_at          = Column(DateTime, nullable=True)  # starts_at + duration_days
    sender_score     = Column(Float, nullable=True)
    recipient_score  = Column(Float, nullable=True)
    winner_id        = Column(String, nullable=True)    # user_id or None for tie
    created_at       = Column(DateTime, default=datetime.utcnow)

    sender    = relationship("User", foreign_keys=[sender_id])
    recipient = relationship("User", foreign_keys=[recipient_id])
