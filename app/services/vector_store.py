"""
ChromaDB vector store service.
Manages collections per materia and performs semantic search.
Each materia_id maps to its own ChromaDB collection for isolation.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logging import get_logger

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
        chunks.append({
            "content": doc,
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
