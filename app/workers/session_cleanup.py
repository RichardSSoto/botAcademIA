"""
Session Cleanup Worker – closes idle conversations automatically.

Runs as a background asyncio task inside the FastAPI process.
Every SESSION_CLEANUP_INTERVAL_SECONDS seconds it queries PostgreSQL for
sessions that are still 'active' but whose last_activity_at is older than
SESSION_IDLE_TIMEOUT_MINUTES.  For each stale session it:

  1. Calls invalidate_session() → deletes the Redis key (stops LLM cache cost)
  2. Updates conversation_sessions.status = 'timeout' and sets closed_at

The timeout and check interval are both configurable via .env:
  SESSION_IDLE_TIMEOUT_MINUTES=3    (default)
  SESSION_CLEANUP_INTERVAL_SECONDS=60  (default, checked every minute)
"""
import asyncio
from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.models.db_models import ConversationSession
from app.services.redis_cache import invalidate_session

logger = get_logger(__name__)

# How often the job wakes up to scan for stale sessions (seconds)
_CHECK_INTERVAL = settings.SESSION_CLEANUP_INTERVAL_SECONDS

_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    """Main loop: wake up every _CHECK_INTERVAL seconds and close idle sessions."""
    logger.info(
        "Session cleanup worker started | idle_timeout=%dmin check_interval=%ds",
        settings.SESSION_IDLE_TIMEOUT_MINUTES, _CHECK_INTERVAL,
    )
    while True:
        try:
            await _run_cleanup()
        except Exception as exc:
            # Never let a crash kill the loop
            logger.error("Session cleanup error: %s", exc, exc_info=True)
        await asyncio.sleep(_CHECK_INTERVAL)


async def _run_cleanup() -> None:
    """Find and close all sessions idle longer than SESSION_IDLE_TIMEOUT_MINUTES."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.SESSION_IDLE_TIMEOUT_MINUTES)

    async with AsyncSessionLocal() as db:
        # Find active sessions whose last activity predates the cutoff
        stmt = select(ConversationSession).where(
            ConversationSession.status == "active",
            ConversationSession.last_activity_at < cutoff,
        )
        result = await db.execute(stmt)
        stale_sessions = result.scalars().all()

        if not stale_sessions:
            logger.debug(
                "Session cleanup | no idle sessions found (cutoff=%s)",
                cutoff.strftime("%H:%M:%S"),
            )
            return

        logger.info(
            "Session cleanup | found %d idle session(s) to close (idle > %dmin)",
            len(stale_sessions), settings.SESSION_IDLE_TIMEOUT_MINUTES,
        )

        now = datetime.now(timezone.utc)
        for session in stale_sessions:
            idle_seconds = (now - session.last_activity_at).total_seconds()

            # 1. Clear Redis cache → stops any LLM context cost
            await invalidate_session(session.interaction_id)

            # 2. Mark session as timed out in PostgreSQL
            session.status = "timeout"
            session.closed_at = now

            logger.info(
                "Session cleanup | CLOSED interaction=%s materia=%s "
                "turns=%d idle=%.0fs status→timeout",
                session.interaction_id, session.materia_id,
                session.turn_count, idle_seconds,
            )

        await db.commit()
        logger.info("Session cleanup | committed %d closure(s)", len(stale_sessions))


async def start_cleanup_worker() -> None:
    """Start the background cleanup task. Called from app lifespan."""
    global _task
    _task = asyncio.create_task(_cleanup_loop(), name="session_cleanup")
    logger.info("Session cleanup worker task created")


async def stop_cleanup_worker() -> None:
    """Cancel the background task on shutdown."""
    global _task
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
        logger.info("Session cleanup worker stopped")
