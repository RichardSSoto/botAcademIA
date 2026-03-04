"""
Redis cache service – session-level cache for interaction history.
Used in production (USE_REDIS=True) to avoid sending full chat history
to the LLM on every call; instead caches the last N exchanges per interaction_id.

In POC mode (USE_REDIS=False) all functions are no-ops so the rest of
the codebase doesn't need to change when Redis is added later.
"""
import json
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis | None:
    if not settings.USE_REDIS:
        return None
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        logger.info("Redis connected at %s", settings.REDIS_URL)
    return _redis_pool


async def get_session_context(interaction_id: str) -> list[dict]:
    """Return cached message history for an interaction (last N exchanges)."""
    r = await get_redis()
    if not r:
        return []
    try:
        raw = await r.get(f"session:{interaction_id}")
        return json.loads(raw) if raw else []
    except Exception as exc:
        logger.warning("Redis GET error: %s", exc)
        return []


async def update_session_context(
    interaction_id: str,
    user_msg: str,
    assistant_msg: str,
    max_turns: int = 5,
) -> None:
    """Append a Q&A turn to the session cache, keeping only the last max_turns."""
    r = await get_redis()
    if not r:
        return
    try:
        key = f"session:{interaction_id}"
        history = await get_session_context(interaction_id)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        # Keep only last max_turns pairs (2 messages per turn)
        history = history[-(max_turns * 2):]
        await r.setex(key, settings.REDIS_TTL_SECONDS, json.dumps(history, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Redis SET error: %s", exc)


async def invalidate_session(interaction_id: str) -> None:
    """Remove session cache for an interaction (e.g. conversation ended)."""
    r = await get_redis()
    if not r:
        return
    try:
        await r.delete(f"session:{interaction_id}")
    except Exception as exc:
        logger.warning("Redis DEL error: %s", exc)


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
