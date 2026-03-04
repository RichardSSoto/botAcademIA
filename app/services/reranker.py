"""
FlashRank reranker: reorders ChromaDB candidates by true semantic relevance.

Model: ms-marco-TinyBERT-L-2-v2 (default)
  - ~4MB ONNX model, NO Torch/Transformers dependency
  - Runs on CPU via ONNX Runtime (~20-50ms for 10 candidates)
  - vs previous sentence-transformers approach: ~500ms + ~500MB Docker image bloat

"Lost in the Middle" reordering (Liu et al., 2023 — arxiv 2307.03172):
  LLMs recall information best from the START and END of the context window.
  Chunks placed in the middle are systematically under-attended.
  After reranking by relevance score, we reorder so that:
    position 0 (first)  → rank 1  (most relevant)
    position 1          → rank 3
    position 2          → rank 5
    position 3 (last-1) → rank 4
    position 4 (last)   → rank 2  (2nd most relevant)
  → Maximum signal at both ends of the context window.

Flow:
  ChromaDB returns RAG_FETCH_K candidates (cosine distance, fast ANN)
  → FlashRank scores each (query, chunk) pair (~20-50ms on CPU)
  → "Lost in the Middle" reordering of top RAG_TOP_K chunks
  → LLM receives context with maximum signal at both ends
"""

import asyncio
import time
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_ranker = None


def _get_ranker():
    """Lazy singleton: loads the FlashRank ONNX model once on first call."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker  # deferred import
        t = time.perf_counter()
        logger.info("Loading FlashRank reranker (ms-marco-TinyBERT-L-2-v2, ~4MB ONNX)...")
        _ranker = Ranker(
            model_name="ms-marco-TinyBERT-L-2-v2",
            max_length=256,   # chunks ~200 tokens + query ~10 → 256 is safe, faster than 512
        )
        logger.info("FlashRank reranker loaded in %.0fms", (time.perf_counter() - t) * 1000)
    return _ranker


def _lost_in_the_middle(chunks: list[dict]) -> list[dict]:
    """
    Reorder relevance-ranked chunks so the most important appear at both
    the beginning and end of the context — minimises the "lost in the middle" effect.

    Algorithm: interleave from both ends of the ranked list.
      Input  (rank order): [A=best, B=2nd, C=3rd, D=4th, E=5th]
      Fill positions from outside in:
        pos 0  ← A   (from left)
        pos -1 ← B   (from right)
        pos 1  ← C   (from left)
        pos -2 ← D   (from right)
        pos 2  ← E   (middle)
      Result: [A, C, E, D, B]
    """
    n = len(chunks)
    if n <= 2:
        return chunks

    result = [None] * n
    lo, hi = 0, n - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:        # even index → fill from the left (start of context)
            result[lo] = chunk
            lo += 1
        else:                  # odd index → fill from the right (end of context)
            result[hi] = chunk
            hi -= 1

    return result


def _rerank_sync(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    CPU-bound reranking via FlashRank ONNX — called via run_in_executor.
    Returns top_k chunks in "Lost in the Middle" order for LLM context.
    """
    from flashrank import RerankRequest

    ranker = _get_ranker()
    passages = [
        {"id": i, "text": c["content"], "meta": c}
        for i, c in enumerate(chunks)
    ]

    t = time.perf_counter()
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    elapsed = (time.perf_counter() - t) * 1000

    # Sort by score descending, keep top_k
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    logger.debug(
        "FlashRank scored %d→top %d in %.0fms | best=%.6f worst=%.6f",
        len(chunks), len(results_sorted), elapsed,
        results_sorted[0]["score"] if results_sorted else 0,
        results_sorted[-1]["score"] if results_sorted else 0,
    )

    # Rebuild chunk dicts with rerank_score attached
    ranked_chunks = []
    for r in results_sorted:
        chunk = dict(r["meta"])           # original chunk dict (copy)
        chunk["rerank_score"] = round(float(r["score"]), 6)
        ranked_chunks.append(chunk)

    # Apply "Lost in the Middle" reordering for LLM context
    return _lost_in_the_middle(ranked_chunks)


async def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> tuple[list[dict], int]:
    """
    Async wrapper: offloads ONNX inference to a thread pool.

    Returns:
        (reranked_chunks, elapsed_ms)
        reranked_chunks — top_k chunks in "Lost in the Middle" order
        elapsed_ms      — wall time for the full reranking step
    """
    k = top_k or settings.RAG_TOP_K

    if not chunks:
        return [], 0

    if len(chunks) <= k:
        logger.debug("Reranker skipped: %d chunks <= top_k=%d", len(chunks), k)
        return chunks, 0

    t = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _rerank_sync, query, chunks, k)
    elapsed = round((time.perf_counter() - t) * 1000)

    return result, elapsed
