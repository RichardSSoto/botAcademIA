"""
ChromaDB vector store service.
Manages collections per materia and performs semantic search.
Each materia_id maps to its own ChromaDB collection for isolation.

Collections:
  materia_{id}  — academic content for a specific course
  materia_utel_faq — shared Q&A FAQ (identical across all materias, stored once)
"""
from __future__ import annotations

import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logging import get_logger

FAQ_COLLECTION_ID = "utel_faq"

logger = get_logger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────

_client: chromadb.AsyncHttpClient | None = None


async def get_chroma_client() -> chromadb.AsyncHttpClient:
    global _client
    if _client is None:
        _client = await chromadb.AsyncHttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
        logger.info("ChromaDB client connected at %s:%s", settings.CHROMA_HOST, settings.CHROMA_PORT)
    return _client


def _collection_name(materia_id: str) -> str:
    """
    ChromaDB collection names must be lowercase alphanumeric + underscores.
    Example: '258_Criminologia_B' -> 'materia_258_criminologia_b'
    """
    clean = materia_id.lower().replace("-", "_").replace(" ", "_")
    return f"materia_{clean}"


# ── Collection operations ─────────────────────────────────────────────────────

async def collection_exists(materia_id: str) -> bool:
    client = await get_chroma_client()
    try:
        cols = await client.list_collections()
        names = [c.name for c in cols]
        return _collection_name(materia_id) in names
    except Exception as exc:
        logger.warning("Could not verify collection existence: %s", exc)
        return False


async def delete_collection(materia_id: str) -> None:
    client = await get_chroma_client()
    name = _collection_name(materia_id)
    try:
        await client.delete_collection(name)
        logger.info("Deleted collection: %s", name)
    except Exception as exc:
        logger.warning("Could not delete collection %s: %s", name, exc)


async def add_documents(
    materia_id: str,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    """
    Add text chunks to the materia's collection.
    ChromaDB generates embeddings using its default embedding function.
    For production replace with Gemini embedding via chromadb.EmbeddingFunction.
    """
    client = await get_chroma_client()
    col = await client.get_or_create_collection(
        name=_collection_name(materia_id),
        metadata={"materia_id": materia_id},
    )
    # ChromaDB handles batching internally; pass all at once
    await col.add(documents=documents, metadatas=metadatas, ids=ids)
    count = await col.count()
    logger.info(
        "Collection '%s' now has %d documents (added %d chunks)",
        _collection_name(materia_id), count, len(documents),
    )
    return count


async def semantic_search(
    materia_id: str,
    query: str,
    n_results: int | None = None,
) -> list[dict]:
    """
    Query the materia's collection for relevant chunks.
    Returns list of dicts with 'content', 'source', 'distance'.
    """
    client = await get_chroma_client()
    n = n_results or settings.RAG_TOP_K
    col_name = _collection_name(materia_id)

    try:
        col = await client.get_collection(col_name)
    except Exception:
        logger.error("Collection '%s' not found. Has materia been ingested?", col_name)
        return []

    results = await col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # FAQ chunks: document = Q: text (for embedding precision)
        # but full_qa metadata holds the complete Q+A for the LLM.
        content = meta.get("full_qa") or doc
        chunks.append({
            "content": content,
            "source": meta.get("source", "unknown"),
            "materia_id": materia_id,
            "distance": round(dist, 4),
        })

    logger.debug("RAG search '%s' returned %d chunks", materia_id, len(chunks))
    return chunks


async def get_collection_count(materia_id: str) -> int:
    """Return the number of documents in a collection."""
    client = await get_chroma_client()
    try:
        col = await client.get_collection(_collection_name(materia_id))
        return await col.count()
    except Exception:
        return 0


async def semantic_search_combined(
    materia_id: str,
    query: str,
    n_results: int | None = None,
) -> list[dict]:
    """
    Query BOTH the materia collection (academic content) AND the shared FAQ
    collection in parallel, then merge all candidates.

    Why combined?
    - Academic collection: course theory, unit content, exam structure.
    - FAQ collection: operational Q&As with complete step-by-step answers.
    - FlashRank reranker picks the best top_k from the merged pool.

    Each collection is queried with n_results candidates.
    Total returned: up to 2 * n_results (reranker reduces to RAG_TOP_K).
    """
    n = n_results or settings.RAG_FETCH_K

    materia_task = semantic_search(materia_id, query, n_results=n)
    faq_task = semantic_search(FAQ_COLLECTION_ID, query, n_results=n)

    materia_chunks, faq_chunks = await asyncio.gather(
        materia_task, faq_task, return_exceptions=True
    )

    all_chunks: list[dict] = []

    if isinstance(materia_chunks, list):
        all_chunks.extend(materia_chunks)

    if isinstance(faq_chunks, list) and faq_chunks:
        # Keep materia_id consistent so pipeline logging shows correct materia
        for c in faq_chunks:
            c["materia_id"] = materia_id
            c["source_collection"] = FAQ_COLLECTION_ID
        all_chunks.extend(faq_chunks)
    elif isinstance(faq_chunks, Exception):
        logger.warning("FAQ collection search failed (not yet ingested?): %s", faq_chunks)

    # Sort by cosine distance (ascending — lower = more similar) before reranking
    all_chunks.sort(key=lambda x: x["distance"])

    logger.debug(
        "Combined search '%s': %d academic + %d faq = %d total candidates",
        materia_id,
        len(materia_chunks) if isinstance(materia_chunks, list) else 0,
        len(faq_chunks) if isinstance(faq_chunks, list) else 0,
        len(all_chunks),
    )
    return all_chunks
