"""
Data ingestion service.
Reads the 3 source files per materia, chunks them, and loads them into ChromaDB.

Supports:
  - contenido.json  (structured course content with units/weeks/classes)
  - contenido.txt   (flat text representation)
  - descripcion.json (course metadata)

Designed to be called from:
  - scripts/ingest_materia.py (CLI tool - current POC)
  - Future: frontend upload endpoint (POST /ingest)
"""
import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Generator

from app.core.config import settings
from app.core.logging import get_logger
from app.services import vector_store

logger = get_logger(__name__)


# ── Text chunking ─────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> Generator[str, None, None]:
    """Split text into overlapping chunks by character count."""
    text = text.strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def _chunk_id(materia_id: str, source: str, idx: int) -> str:
    """Deterministic, reproducible chunk ID."""
    raw = f"{materia_id}::{source}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_from_txt(path: Path) -> list[tuple[str, dict]]:
    """Read .txt file and return list of (chunk_text, metadata)."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    results = []
    for i, chunk in enumerate(_chunk_text(text)):
        meta = {"source": "contenido.txt", "chunk_index": i}
        results.append((chunk, meta))
    return results


def _flatten_json(obj, prefix: str = "") -> list[str]:
    """
    Recursively flatten a JSON object into human-readable text segments.
    Handles nested dicts and lists automatically.
    """
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = f"{prefix}.{k}" if prefix else k
            parts.extend(_flatten_json(v, label))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            parts.extend(_flatten_json(item, f"{prefix}[{i}]"))
    elif isinstance(obj, str) and obj.strip():
        parts.append(f"{prefix}: {obj.strip()}")
    elif obj is not None:
        parts.append(f"{prefix}: {obj}")
    return parts


def _extract_from_json(path: Path, source_label: str) -> list[tuple[str, dict]]:
    """Read a .json file, flatten it, then chunk the text."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return []

    flat_lines = _flatten_json(data)
    full_text = "\n".join(flat_lines)

    results = []
    for i, chunk in enumerate(_chunk_text(full_text)):
        meta = {"source": source_label, "chunk_index": i}
        results.append((chunk, meta))
    return results


# ── Main ingest function ──────────────────────────────────────────────────────

async def ingest_materia(
    materia_id: str,
    force_reingest: bool = False,
) -> dict:
    """
    Full ingestion pipeline for one materia.
    Returns a status dict: { materia_id, status, num_chunks, message }
    """
    logger.info("Ingesting materia: %s (force=%s)", materia_id, force_reingest)

    # Locate the materia folder
    base_dir = Path(settings.MATERIAS_DATA_DIR)
    materia_dir = base_dir / materia_id
    if not materia_dir.exists():
        msg = f"Materia folder not found: {materia_dir}"
        logger.error(msg)
        return {"materia_id": materia_id, "status": "error", "num_chunks": 0, "message": msg}

    # Check if already ingested
    if not force_reingest and await vector_store.collection_exists(materia_id):
        count = await vector_store.get_collection_count(materia_id)
        msg = f"Already ingested ({count} chunks). Use force_reingest=True to overwrite."
        logger.info(msg)
        return {"materia_id": materia_id, "status": "already_indexed", "num_chunks": count, "message": msg}

    if force_reingest:
        await vector_store.delete_collection(materia_id)

    # ── Collect chunks from all 3 source files ─────────────────────────────
    all_docs: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []

    global_idx = 0

    # 1. contenido.txt
    txt_path = materia_dir / "contenido.txt"
    if txt_path.exists():
        for chunk, meta in _extract_from_txt(txt_path):
            all_docs.append(chunk)
            all_metas.append({**meta, "materia_id": materia_id})
            all_ids.append(_chunk_id(materia_id, meta["source"], global_idx))
            global_idx += 1
        logger.debug("Extracted %d chunks from contenido.txt", len(all_docs))
    else:
        logger.warning("contenido.txt not found for %s", materia_id)

    # 2. contenido.json
    json_path = materia_dir / "contenido.json"
    pre_count = len(all_docs)
    if json_path.exists():
        for chunk, meta in _extract_from_json(json_path, "contenido.json"):
            all_docs.append(chunk)
            all_metas.append({**meta, "materia_id": materia_id})
            all_ids.append(_chunk_id(materia_id, meta["source"], global_idx))
            global_idx += 1
        logger.debug("Extracted %d chunks from contenido.json", len(all_docs) - pre_count)
    else:
        logger.warning("contenido.json not found for %s", materia_id)

    # 3. descripcion.json
    desc_path = materia_dir / "descripcion.json"
    pre_count = len(all_docs)
    if desc_path.exists():
        for chunk, meta in _extract_from_json(desc_path, "descripcion.json"):
            all_docs.append(chunk)
            all_metas.append({**meta, "materia_id": materia_id})
            all_ids.append(_chunk_id(materia_id, meta["source"], global_idx))
            global_idx += 1
        logger.debug("Extracted %d chunks from descripcion.json", len(all_docs) - pre_count)
    else:
        logger.warning("descripcion.json not found for %s", materia_id)

    if not all_docs:
        msg = "No extractable content found. Check that source files are not empty."
        return {"materia_id": materia_id, "status": "error", "num_chunks": 0, "message": msg}

    # ── Upload to ChromaDB in batches ──────────────────────────────────────
    BATCH_SIZE = 500
    total_uploaded = 0
    for batch_start in range(0, len(all_docs), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        count = await vector_store.add_documents(
            materia_id=materia_id,
            documents=all_docs[batch_start:batch_end],
            metadatas=all_metas[batch_start:batch_end],
            ids=all_ids[batch_start:batch_end],
        )
        total_uploaded = count
        logger.info(
            "Batch %d-%d uploaded | total in collection: %d",
            batch_start, batch_end, count,
        )

    msg = f"Successfully ingested {total_uploaded} chunks from {len(all_docs)} extracted."
    logger.info(msg)
    return {
        "materia_id": materia_id,
        "status": "indexed",
        "num_chunks": total_uploaded,
        "message": msg,
    }
