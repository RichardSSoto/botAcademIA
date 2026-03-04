"""
/api/v1/query  - Main endpoint consumed by BOT LUA
/api/v1/health - Service liveness check
"""
import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse, HealthResponse
from app.models.db_models import QueryLog
from app.services.pipeline import run_pipeline
from app.services import vector_store

logger = get_logger(__name__)
router = APIRouter()


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Process a student query via RAG pipeline",
    description=(
        "Receives a message from BOT LUA with an interaction_id and materia_id. "
        "Runs the full pipeline: pre-process → vector search → LLM response. "
        "Logs the complete interaction to PostgreSQL."
    ),
)
async def process_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:

    # ── Create pending log entry ──────────────────────────────────────────────
    log_entry = QueryLog(
        interaction_id=request.interaction_id,
        materia_id=request.materia_id,
        original_message=request.message,
        status="processing",
    )
    db.add(log_entry)
    await db.flush()  # get the ID without committing

    try:
        result_tuple = await run_pipeline(request)

        # run_pipeline returns (QueryResponse, metadata_dict)
        if isinstance(result_tuple, tuple):
            response, meta = result_tuple
        else:
            response, meta = result_tuple, {}

        # ── Update log entry ──────────────────────────────────────────────────
        log_entry.intent = response.intent
        log_entry.sentiment = response.sentiment
        log_entry.clean_query = meta.get("_clean_query")
        log_entry.retrieved_chunks = json.dumps(meta.get("_chunks", []), ensure_ascii=False)
        log_entry.num_chunks = len(meta.get("_chunks", []))
        log_entry.response = response.response
        log_entry.preprocessing_ms = meta.get("_preprocessing_ms")
        log_entry.rag_ms = meta.get("_rag_ms")
        log_entry.llm_ms = meta.get("_llm_ms")
        log_entry.total_ms = response.processing_time_ms
        log_entry.status = "completed"

        return response

    except Exception as exc:
        logger.exception("Pipeline error for interaction=%s: %s", request.interaction_id, exc)
        log_entry.status = "error"
        log_entry.error_message = str(exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline error: {exc}",
        )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    services: dict = {}

    # ChromaDB
    try:
        client = await vector_store.get_chroma_client()
        cols = await client.list_collections()
        services["chromadb"] = {"status": "ok", "collections": len(cols)}
    except Exception as exc:
        services["chromadb"] = {"status": "error", "detail": str(exc)}

    # Gemini (just config check, no API call)
    services["gemini"] = {
        "status": "configured" if settings.GEMINI_API_KEY else "missing_api_key",
        "model": settings.GEMINI_MODEL,
    }

    overall = "ok" if all(s.get("status") == "ok" for s in services.values()
                          if "status" in s) else "degraded"
    # Still return 200 — actual connectivity issues surface in /query
    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=services,
        timestamp=datetime.now(timezone.utc),
    )
