"""
Semantic response cache — avoids re-calling Gemini for paraphrases of identical questions.

Design rules:
  - Cacheable when the question is SELF-CONTAINED, regardless of whether
    there is prior conversation history.
    "¿Qué es la probabilidad?" in turn 3 is still self-contained → cache hit.
  - NOT cacheable when the question REFERENCES prior context:
    pronouns ("eso", "ese", "lo mismo"), continuation connectors ("y si...",
    "pero si...", "entonces..."), back-references ("el mismo", "lo que dijiste").
  - Key namespace: semcache:{materia_id}:{md5}  — same question in different materias
    can never collide, even if the text is identical.
  - Embeddings: Gemini text-embedding-004  (~10x cheaper than flash-lite per token).
  - Similarity: cosine ≥ SEMANTIC_CACHE_THRESHOLD (default 0.92, configurable in .env).
  - TTL: SEMANTIC_CACHE_TTL_SECONDS  (default 86400 = 24 h).
  - Storage backend: existing Redis connection (reuses get_redis() pool).
"""
import asyncio
import hashlib
import json
import re

import numpy as np
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger
from app.services.redis_cache import get_redis

logger = get_logger(__name__)

_KEY_PREFIX = "semcache"

# ── Context-dependency detection ──────────────────────────────────────────────
#
# STRICT patterns: these ALWAYS indicate the message needs prior context.
# We intentionally exclude loose pronouns like "eso"/"este" because they
# frequently appear in self-contained questions:
#   "¿Es esto parte del examen?"  → cacheable, even though it contains "esto"
#   "no entiendo esto"            → borderline but cacheable (topic-agnostic doubt)
# Only flag patterns that are UNAMBIGUOUSLY back-references.
_CONTEXT_PATTERNS = re.compile(
    r"""
    \b(
        lo\s+mismo|lo\s+anterior|lo\s+de\s+antes|lo\s+de\s+recién|
        y\s+si\b|y\s+luego\b|y\s+despu[eé]s\b|
        pero\s+si\b|pero\s+cu[aá]l\b|
        entonces\s+si\b|entonces\s+c[oó]mo\b|
        lo\s+que\s+(?:dijiste|mencionaste|explicaste|dije|pusiste)|
        siguiendo\s+con|volviendo\s+a|
        la\s+respuesta\s+anterior|el\s+ejemplo\s+anterior|
        eso\s+que\s+(?:dijiste|mencionaste|explicaste)|    
        ese\s+(?:concepto|tema|ejemplo|punto)\s+que
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Connectors that at the START of a message almost always mean follow-up.
_STARTER_PATTERN = re.compile(
    r"^(y|pero|entonces|tambi[eé]n|adem[aá]s|o\s+sea|es\s+decir|igual|igualmente)\b",
    re.IGNORECASE,
)


def _is_context_dependent(message: str) -> bool:
    """
    Returns True when the message likely requires prior conversation history.

    Rules (applied in order, $0 cost — pure Python):
      1. Ultra-short messages (≤ 3 words) containing a connector/pronoun → always context.
         e.g. "y ese?", "pero por qué?"
      2. Starts with continuation connector AND short (< 50 chars) → context.
         e.g. "y si el dado tiene 8 caras?"
         Exception: long messages (≥ 10 words) that start with connector are usually
         self-contained formal questions, so we only apply this to short messages.
      3. Contains a strict back-reference pattern → context.
         e.g. "lo anterior", "lo que dijiste", "siguiendo con eso que explicaste"
      4. Everything else → self-contained, allow cache lookup.
    """
    msg = message.strip()
    words = msg.split()
    n_words = len(words)

    # Rule 1: ultra-short with connector/pronoun
    if n_words <= 3:
        _SHORT_CONTEXT = re.compile(
            r"\b(y|pero|entonces|ese|esa|eso|esto|estos|estas|él|ella|ellos)\b",
            re.IGNORECASE,
        )
        if _SHORT_CONTEXT.search(msg):
            return True

    # Rule 2: starts with connector + not a full question (< 50 chars)
    if len(msg) < 50 and _STARTER_PATTERN.match(msg):
        return True

    # Rule 3: strict back-reference patterns (always unambiguous)
    if _CONTEXT_PATTERNS.search(msg):
        return True

    # Rule 4: long self-contained question — let the embedding decide relevance
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


async def _embed(text: str) -> list[float]:
    """Embed text using the configured Gemini embedding model (sync SDK → thread)."""
    result = await asyncio.to_thread(
        genai.embed_content,
        model=settings.EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


def _normalize_message(message: str) -> str:
    """
    Normalize message text before hashing so minor surface variations
    (accents on question marks, double spaces, trailing punctuation)
    map to the same cache key.
      "¿Cuál es la probabilidad?"  →  "cual es la probabilidad"
      "cual es la probabilidad???" →  "cual es la probabilidad"
    """
    msg = message.lower().strip()
    # Remove Spanish opening/closing punctuation and common trailing chars
    msg = re.sub(r"[¿?¡!.,;:]", "", msg)
    # Collapse multiple whitespace
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def _redis_key(materia_id: str, message: str) -> str:
    """Deterministic Redis key for a (materia, message) pair."""
    normalized = _normalize_message(message)
    h = hashlib.md5(f"{materia_id}|{normalized}".encode()).hexdigest()[:16]
    return f"{_KEY_PREFIX}:{materia_id}:{h}"


# ── Public API ────────────────────────────────────────────────────────────────

async def get_cached_response(
    message: str,
    materia_id: str,
    history: list[dict],
) -> dict | None:
    """
    Look up the semantic cache for a context-free question.

    Returns a dict with keys  {response, intent, sentiment, _cache_similarity}
    or None if:
      - cache is disabled
      - Redis is unavailable
      - message is context-dependent (references prior turns)
      - no stored entry has similarity ≥ SEMANTIC_CACHE_THRESHOLD
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    if settings.USE_MOCK_LLM:
        return None  # Gemini embedding SDK not configured in mock mode
    if history and _is_context_dependent(message):
        # Message references prior context (eso, y si, el mismo, short connector, etc.)
        # Cannot safely serve a cached answer — meaning depends on conversation history.
        logger.debug(
            "Semantic cache SKIP (context-dependent) | msg='%s'", message[:60]
        )
        return None

    r = await get_redis()
    if not r:
        return None

    try:
        query_emb = await _embed(f"{materia_id} | {message}")

        best_score = 0.0
        best_entry: dict | None = None

        # Scan all entries stored for this materia
        async for key in r.scan_iter(match=f"{_KEY_PREFIX}:{materia_id}:*", count=200):
            raw = await r.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            score = _cosine_sim(query_emb, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score >= settings.SEMANTIC_CACHE_THRESHOLD and best_entry:
            logger.info(
                "Semantic cache HIT | materia=%s | similarity=%.4f | cached_msg='%s'",
                materia_id, best_score, best_entry.get("message", "")[:80],
            )
            return {
                "response":          best_entry["response"],
                "intent":            best_entry["intent"],
                "sentiment":         best_entry["sentiment"],
                "_cache_similarity": best_score,
            }

        logger.debug(
            "Semantic cache MISS | materia=%s | best_score=%.4f (threshold=%.2f)",
            materia_id, best_score, settings.SEMANTIC_CACHE_THRESHOLD,
        )

    except Exception as exc:
        logger.warning("Semantic cache lookup error: %s", exc)

    return None


async def cache_response(
    message: str,
    materia_id: str,
    response: str,
    intent: str,
    sentiment: str,
    history: list[dict],
) -> None:
    """
    Store a response in the semantic cache.
    No-op when cache is disabled, Redis is unavailable, or history was non-empty at query time.
    Called via asyncio.create_task() from pipeline.py so it does NOT block the HTTP response.
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return
    if settings.USE_MOCK_LLM:
        return  # Gemini embedding SDK not configured in mock mode
    if history and _is_context_dependent(message):
        return  # context-dependent — answer is tied to prior turns, don't cache

    r = await get_redis()
    if not r:
        return

    try:
        emb = await _embed(f"{materia_id} | {message}")
        key = _redis_key(materia_id, message)
        entry = {
            "message":   message,
            "embedding": emb,
            "response":  response,
            "intent":    intent,
            "sentiment": sentiment,
        }
        await r.setex(
            key,
            settings.SEMANTIC_CACHE_TTL_SECONDS,
            json.dumps(entry, ensure_ascii=False),
        )
        logger.info(
            "Semantic cache STORED | materia=%s | key=%s | ttl=%ds",
            materia_id, key, settings.SEMANTIC_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Semantic cache write error: %s", exc)
