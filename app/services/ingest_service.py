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


# ── FAQ collection name (shared across all materias) ─────────────────────────
FAQ_COLLECTION_ID = "utel_faq"


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


# ── Unit-aware txt chunking ───────────────────────────────────────────────────

UNIT_TITLE_MARKER = r"El titulo de esta unidad es:"


def _extract_unit_chunks_from_txt(
    academic_text: str,
) -> list[tuple[str, dict]]:
    """
    Split the academic section by unit boundary markers and chunk each unit
    independently so every chunk carries metadata['unidad'] and
    metadata['unidad_num'].

    Unit boundary pattern inside contenido.txt:
        "El titulo de esta unidad es: Unidad N <name>. ..."
    """
    segments = re.split(UNIT_TITLE_MARKER, academic_text)

    # Segments[0] is pre-unit header (course intro, evaluations, etc.)
    # Tag it without a specific unit number.
    results: list[tuple[str, dict]] = []

    if segments[0].strip():
        for i, chunk in enumerate(_chunk_text(segments[0])):
            results.append((f"[Introduccion] {chunk}", {
                "source": "contenido.txt",
                "tipo": "academico",
                "unidad": "Introduccion",
                "unidad_num": 0,
                "chunk_index": i,
            }))

    for segment in segments[1:]:
        # First characters: "Unidad N <title>. El texto de introduccion ..."
        name_match = re.match(r"\s*(Unidad\s+\d+[^.]*)", segment)
        unit_name = name_match.group(1).strip() if name_match else "Sin unidad"
        num_match = re.search(r"Unidad\s+(\d+)", unit_name)
        unit_num = int(num_match.group(1)) if num_match else 0

        base_idx = len(results)
        for i, chunk in enumerate(_chunk_text(segment)):
            results.append((f"[{unit_name}] {chunk}", {
                "source": "contenido.txt",
                "tipo": "academico",
                "unidad": unit_name,
                "unidad_num": unit_num,
                "chunk_index": base_idx + i,
            }))

    return results


# ── Structured contenido.json extractor ──────────────────────────────────────


def _clean_resources(raw_list: list) -> list[dict]:
    """Filter and normalise a list of resource dicts from contenido.json."""
    cleaned = []
    for r in raw_list or []:
        url = (r.get("url") or "").strip()
        if not url or url.startswith("//view.vzaar"):
            # Skip empty URLs and raw vzaar embed paths (no useful link for students)
            continue
        tipo = (r.get("tipo") or "").strip()
        titulo = (r.get("titulo") or "").strip()
        cleaned.append({"titulo": titulo, "tipo": tipo, "url": url})
    return cleaned


def _extract_from_contenido_json(
    path: Path,
) -> tuple[list[tuple[str, dict]], dict[int, list[dict]]]:
    """
    Parse contenido.json and return:
      - chunks: list[(text, metadata)] tagged with unidad / semana
      - resources_by_unit: {unit_num: [{titulo, tipo, url}, ...]} for query-time lookup

    Each text chunk carries:
      metadata['source']      = 'contenido.json'
      metadata['tipo']        = 'academico'
      metadata['unidad']      = e.g. 'Unidad 2 Probabilidad'
      metadata['unidad_num']  = 2
      metadata['semana']      = e.g. 'Semana 2'
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return [], {}

    curso = data.get("curso", {})
    units = curso.get("unidades", [])
    chunks: list[tuple[str, dict]] = []
    resources_by_unit: dict[int, list[dict]] = {}
    global_idx = 0

    for unit in units:
        unit_title = (unit.get("titulo") or "").strip()
        num_match = re.search(r"Unidad\s+(\d+)", unit_title)
        unit_num = int(num_match.group(1)) if num_match else 0

        unit_resources: list[dict] = []

        # -- Unit-level text (intro, learning outcomes, competencies) ----------
        unit_text_parts: list[str] = []
        if unit.get("introduccion"):
            unit_text_parts.append(f"Introduccion: {unit['introduccion']}")
        if unit.get("resultados_de_aprendizaje"):
            unit_text_parts.append(
                f"Resultados de aprendizaje: {unit['resultados_de_aprendizaje']}"
            )
        if unit.get("competencias"):
            unit_text_parts.append(f"Competencias: {unit['competencias']}")

        unit_text = "\n".join(unit_text_parts).strip()
        if unit_text:
            for chunk in _chunk_text(unit_text):
                chunks.append((f"[{unit_title}] {chunk}", {
                    "source": "contenido.json",
                    "tipo": "academico",
                    "unidad": unit_title,
                    "unidad_num": unit_num,
                    "semana": "",
                    "chunk_index": global_idx,
                }))
                global_idx += 1

        # -- Per-week content --------------------------------------------------
        for semana in unit.get("semanas", []):
            semana_titulo = (semana.get("titulo") or "").strip()

            # Class content text
            clase = semana.get("clase") or {}
            clase_text = (clase.get("contenido") or "").strip()
            if clase_text:
                for chunk in _chunk_text(clase_text):
                    chunks.append((f"[{unit_title}] {chunk}", {
                        "source": "contenido.json",
                        "tipo": "academico",
                        "unidad": unit_title,
                        "unidad_num": unit_num,
                        "semana": semana_titulo,
                        "chunk_index": global_idx,
                    }))
                    global_idx += 1

            # Resources from class (videos)
            unit_resources.extend(_clean_resources(clase.get("recursos", [])))

            # Resources from semana (PDFs, additional videos)
            unit_resources.extend(_clean_resources(semana.get("recursos", [])))

            # Resource text content (e.g. PDF extracts, video transcripts)
            for recurso in semana.get("recursos", []):
                contenido = (recurso.get("contenido") or "").strip()
                if contenido and len(contenido) > 100:
                    for chunk in _chunk_text(contenido):
                        chunks.append((f"[{unit_title}] {chunk}", {
                            "source": "contenido.json",
                            "tipo": "academico",
                            "unidad": unit_title,
                            "unidad_num": unit_num,
                            "semana": semana_titulo,
                            "chunk_index": global_idx,
                        }))
                        global_idx += 1

        # Deduplicate resources within the unit
        seen_urls: set[str] = set()
        deduped: list[dict] = []
        for r in unit_resources:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                deduped.append(r)
        resources_by_unit[unit_num] = deduped

    return chunks, resources_by_unit


# ── Resource lookup (called at query time by llm_service) ─────────────────────


def get_materia_resources(
    materia_id: str,
    unit_nums: list[int] | None = None,
) -> dict[int, list[dict]]:
    """
    Return support materials from contenido.json for the given materia.
    If unit_nums is provided, filter to only those units.
    Returns {unit_num: [{titulo, tipo, url}, ...]}.
    Called at query time to enrich the LLM prompt with relevant URLs.
    """
    base_dir = Path(settings.MATERIAS_DATA_DIR)
    path = base_dir / materia_id / "contenido.json"
    if not path.exists():
        return {}
    _, resources_by_unit = _extract_from_contenido_json(path)
    if unit_nums is not None:
        return {k: v for k, v in resources_by_unit.items() if k in unit_nums}
    return resources_by_unit


def _split_qa_sections(text: str) -> tuple[str, str]:
    """
    Split contenido.txt into (academic_text, faq_text).
    The FAQ section begins at the first line starting with 'Q:'.
    Returns (academic, faq) — faq may be empty if no Q: lines found.
    """
    lines = text.splitlines()
    q_start = next((i for i, line in enumerate(lines) if line.startswith("Q:")), len(lines))
    academic = "\n".join(lines[:q_start])
    faq = "\n".join(lines[q_start:])
    return academic, faq


def _extract_qa_chunks(faq_text: str) -> list[tuple[str, dict]]:
    """
    Split FAQ text into individual Q&A pair chunks.
    Each complete 'Q: ... A: ...' block = 1 chunk.

    This is the core improvement over fixed-size chunking:
    - Fixed-size (1000 chars) would split a 5-step answer across 2 chunks,
      so the LLM receives only Paso 1-2 and invents the rest.
    - Q&A-aware: the complete answer (all 5 steps) fits in 1 chunk.
    - Each chunk is tagged tipo='faq' for source tracing.
    """
    lines = faq_text.splitlines()
    results = []
    current: list[str] = []

    for line in lines:
        if line.startswith("Q:") and current:
            chunk = "\n".join(current).strip()
            if chunk:
                results.append((chunk, {"source": "contenido.txt", "tipo": "faq"}))
            current = [line]
        else:
            current.append(line)

    # flush last block
    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            results.append((chunk, {"source": "contenido.txt", "tipo": "faq"}))

    # Build final list:
    #   document (embedded by ChromaDB) = Q: line only  → precise semantic match to student queries
    #   metadata["full_qa"]             = full Q+A text  → returned to FlashRank + LLM
    #
    # Why? All FAQ questions share UTEL phrasing ("¿cómo puedo X?"). If we embed
    # the full Q+A (question + 5-step answer), the embedding becomes "diluted" and
    # ChromaDB retrieves topically similar but wrong FAQ items. Embedding only the
    # Q: line keeps retrieval sharp; storing full_qa preserves the complete answer.
    final = []
    for i, (full_qa, meta) in enumerate(results):
        q_lines = [l for l in full_qa.splitlines() if l.startswith("Q:")]
        q_text = " ".join(q_lines) if q_lines else full_qa[:300]
        final.append((q_text, {**meta, "chunk_index": i, "full_qa": full_qa}))
    return final


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_from_txt(path: Path) -> list[tuple[str, dict]]:
    """
    Read .txt file and return ONLY the academic section as unit-aware chunks.
    Each chunk is tagged tipo='academico' plus unidad / unidad_num metadata.
    The Q&A FAQ section is excluded — handled separately via ingest_faq().
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    academic_text, faq_text = _split_qa_sections(text)
    logger.debug(
        "%s | academic=%d chars  faq=%d chars  (%d Q&A pairs)",
        path.name, len(academic_text), len(faq_text),
        len([l for l in faq_text.splitlines() if l.startswith("Q:")]),
    )

    results = _extract_unit_chunks_from_txt(academic_text)

    # Log unit distribution for debugging
    from collections import Counter
    unit_counts = Counter(meta.get("unidad", "?") for _, meta in results)
    logger.debug("%s | unit chunks: %s", path.name, dict(unit_counts))

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
    """
    Read a .json file and chunk its text.
    For contenido.json, uses the structured extractor that preserves
    unit/semana hierarchy and resource metadata.
    For other JSON files (e.g. descripcion.json), falls back to flat extraction.
    """
    if path.name == "contenido.json":
        chunks, _ = _extract_from_contenido_json(path)
        return chunks

    # Fallback: flat extraction for other JSON files
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return []

    flat_lines = _flatten_json(data)
    full_text = "\n".join(flat_lines)

    results = []
    for i, chunk in enumerate(_chunk_text(full_text)):
        meta = {"source": source_label, "tipo": "academico", "chunk_index": i}
        results.append((chunk, meta))
    return results


# ── FAQ ingest (shared collection, called once) ───────────────────────────────

async def ingest_faq(force_reingest: bool = False) -> dict:
    """
    Ingest the shared UTEL Q&A FAQ into a single 'utel_faq' ChromaDB collection.

    The FAQ section is identical across ALL materias (confirmed by MD5 comparison).
    Storing it once instead of 5× reduces duplication and prevents the reranker
    from returning the same chunk 5 times with near-identical distances.

    Each Q&A pair becomes one atomic chunk — no step-by-step answers split mid-way.
    """
    logger.info("Ingesting shared FAQ collection (force=%s)", force_reingest)

    if not force_reingest and await vector_store.collection_exists(FAQ_COLLECTION_ID):
        count = await vector_store.get_collection_count(FAQ_COLLECTION_ID)
        msg = f"Already indexed ({count} Q&A chunks). Use force_reingest=True to overwrite."
        logger.info(msg)
        return {"materia_id": FAQ_COLLECTION_ID, "status": "already_indexed",
                "num_chunks": count, "message": msg}

    if force_reingest:
        await vector_store.delete_collection(FAQ_COLLECTION_ID)

    # Use any materia's contenido.txt (FAQ sections are identical)
    base_dir = Path(settings.MATERIAS_DATA_DIR)
    sample_dir = next((d for d in sorted(base_dir.iterdir()) if d.is_dir()), None)
    if not sample_dir:
        msg = "No materia directories found in MATERIAS_DATA_DIR."
        logger.error(msg)
        return {"materia_id": FAQ_COLLECTION_ID, "status": "error", "num_chunks": 0, "message": msg}

    txt_path = sample_dir / "contenido.txt"
    if not txt_path.exists():
        msg = f"contenido.txt not found in {sample_dir.name}"
        logger.error(msg)
        return {"materia_id": FAQ_COLLECTION_ID, "status": "error", "num_chunks": 0, "message": msg}

    text = txt_path.read_text(encoding="utf-8")
    _, faq_text = _split_qa_sections(text)
    faq_items = _extract_qa_chunks(faq_text)

    if not faq_items:
        msg = "No Q&A pairs found in FAQ section."
        logger.error(msg)
        return {"materia_id": FAQ_COLLECTION_ID, "status": "error", "num_chunks": 0, "message": msg}

    all_docs = [chunk for chunk, _ in faq_items]
    all_metas = [{**meta, "materia_id": FAQ_COLLECTION_ID} for _, meta in faq_items]
    all_ids = [_chunk_id(FAQ_COLLECTION_ID, "faq", i) for i in range(len(faq_items))]

    BATCH_SIZE = 500
    total = 0
    for start in range(0, len(all_docs), BATCH_SIZE):
        total = await vector_store.add_documents(
            materia_id=FAQ_COLLECTION_ID,
            documents=all_docs[start:start + BATCH_SIZE],
            metadatas=all_metas[start:start + BATCH_SIZE],
            ids=all_ids[start:start + BATCH_SIZE],
        )

    msg = f"Ingested {total} Q&A pair chunks into '{FAQ_COLLECTION_ID}' (from {sample_dir.name})."
    logger.info(msg)
    return {"materia_id": FAQ_COLLECTION_ID, "status": "indexed", "num_chunks": total, "message": msg}


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
