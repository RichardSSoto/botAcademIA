"""
/api/v1/ingest  - Trigger vectorisation of a materia (POC: manual + future frontend)
/api/v1/ingest/list - List all ingested materias

In production this will be protected by API key / OAuth2.
"""
import os
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import IngestRequest, IngestResponse
from app.models.db_models import MateriaIndex
from app.services.ingest_service import ingest_materia

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest / vectorise a materia's source files into ChromaDB",
    description=(
        "Must be called once per materia before accepting student queries. "
        "Reads contenido.txt, contenido.json and descripcion.json from "
        "data/materias/{materia_id}/ and stores chunks in ChromaDB."
    ),
)
async def trigger_ingest(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:

    result = await ingest_materia(
        materia_id=request.materia_id,
        force_reingest=request.force_reingest,
    )

    # Upsert MateriaIndex record
    stmt = select(MateriaIndex).where(MateriaIndex.materia_id == request.materia_id)
    existing = (await db.execute(stmt)).scalar_one_or_none()

    if existing:
        existing.status = result["status"]
        existing.num_chunks = result["num_chunks"]
        existing.error_message = result["message"] if result["status"] == "error" else None
        if result["status"] in ("indexed", "already_indexed"):
            existing.indexed_at = datetime.now(timezone.utc)
    else:
        db.add(MateriaIndex(
            materia_id=request.materia_id,
            status=result["status"],
            num_chunks=result["num_chunks"],
            indexed_at=datetime.now(timezone.utc) if result["status"] == "indexed" else None,
        ))

    return IngestResponse(**result)


@router.get(
    "/ingest/list",
    summary="List all materias available for ingestion",
)
async def list_materias() -> dict:
    """
    Scans data/materias/ directory and returns which materias exist locally.
    Frontend will use this to show available materias to upload/ingest.
    """
    base = Path(settings.MATERIAS_DATA_DIR)
    if not base.exists():
        return {"materias": [], "base_dir": str(base)}

    materias = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            files = [f.name for f in entry.iterdir() if f.is_file()]
            materias.append({
                "materia_id": entry.name,
                "files": files,
                "ready_for_ingest": all(
                    f in files for f in ["contenido.txt", "contenido.json", "descripcion.json"]
                ),
            })

    return {"materias": materias, "count": len(materias)}
