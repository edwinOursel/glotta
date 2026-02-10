"""
Database setup — async SQLAlchemy with SQLite.

SQLite for now (zero-config), swap DATABASE_URL for PostgreSQL in production:
  postgresql+asyncpg://user:password@localhost/glotta
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./glotta.db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields an async DB session per request."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables on startup (dev convenience — use Alembic in prod)."""
    from models import Base as ModelsBase  # noqa: import triggers model registration
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
