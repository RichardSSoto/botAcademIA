"""
Async SQLAlchemy engine + session factory for PostgreSQL.
Uses asyncpg driver for non-blocking I/O with FastAPI.
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncSession:  # type: ignore[return]
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables on startup and apply incremental column migrations (idempotent)."""
    async with engine.begin() as conn:
        # Import models so Base knows about all tables
        from app.models import db_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

        # ── Incremental column additions ─ safe to run on existing databases ───────────
        # PostgreSQL supports ADD COLUMN IF NOT EXISTS → fully idempotent
        migrations = [
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS turn_number INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS session_status VARCHAR(32)",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS tokens_in INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS tokens_out INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS tokens_total INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS preprocess_tokens_in INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS preprocess_tokens_out INTEGER",
            "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS all_tokens_total INTEGER",
            "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS total_tokens_in INTEGER DEFAULT 0",
            "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS total_tokens_out INTEGER DEFAULT 0",
            "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0",
            "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS user_messages JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE conversation_sessions ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMPTZ DEFAULT now()",
        ]
        for sql in migrations:
            try:
                await conn.execute(text(sql))
            except Exception as exc:  # pragma: no cover
                # Should never fail with IF NOT EXISTS, but log just in case
                from app.core.logging import get_logger
                get_logger(__name__).warning("Migration skipped: %s | %s", sql, exc)
