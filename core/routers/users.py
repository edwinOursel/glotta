"""
User profile router — read and update the authenticated user's profile.
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import User
from auth import get_current_user
from schemas import UserProfileResponse, UpdateProfileRequest

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/profile", response_model=UserProfileResponse)
async def update_profile(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.username is not None:
        current_user.username = body.username
    if body.jlpt_level is not None:
        current_user.jlpt_level = body.jlpt_level
    if body.constraint_mode is not None:
        current_user.constraint_mode = body.constraint_mode

    current_user.last_active = datetime.utcnow()
    db.add(current_user)
    return current_user
