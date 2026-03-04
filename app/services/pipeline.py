"""
Pipeline orchestrator: runs the full RAG pipeline synchronously for POC.
In production this is split across 2 Kafka workers; here it's a single
async function that the FastAPI route calls directly.

Flow: raw_message → [preprocess (LLM) ‖ semantic_search (ChromaDB)] → generate (LLM)
Note: preprocess and RAG run in PARALLEL via asyncio.gather to reduce latency.
      RAG uses the raw message during parallel phase; clean_query improves intent
      classification and sentiment detection, both logged for debugging.
"""
import asyncio
import time
import json
from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse, PreprocessResult, RAGResult, RetrievedChunk
from app.services import llm_service, vector_store, reranker
from app.core.config import settings

logger = get_logger(__name__)


async def run_pipeline(request: QueryRequest) -> QueryResponse:
    """
    End-to-end pipeline for a single student query.
    Stage 1 (preprocess) and Stage 2 (RAG search) run in parallel.
    """
    t_start = time.perf_counter()
    logger.info(
        "Pipeline start | interaction=%s materia=%s",
        request.interaction_id, request.materia_id,
    )

    # ── Stages 1 + 2: PARALLEL ─ preprocess + RAG search ────────────────────
    # Both run concurrently. RAG uses the raw message (clean_query is logged
    # for comparison but search already done before preprocess finishes).
    t0 = time.perf_counter()

    async def _timed_preprocess():
        t = time.perf_counter()
        result = await llm_service.preprocess_query(
            raw_message=request.message,
            materia_id=request.materia_id,
        )
        return result, round((time.perf_counter() - t) * 1000)

    async def _timed_rag():
        t = time.perf_counter()
        result = await vector_store.semantic_search(
            materia_id=request.materia_id,
            query=request.message,      # raw query — runs in parallel with preprocess
            n_results=settings.RAG_FETCH_K,   # fetch more candidates for reranking
        )
        return result, round((time.perf_counter() - t) * 1000)

    (preprocess_raw, preprocessing_ms), (raw_chunks, rag_ms) = await asyncio.gather(
        _timed_preprocess(),
        _timed_rag(),
    )
    parallel_wall_ms = round((time.perf_counter() - t0) * 1000)
    saved_ms = preprocessing_ms + rag_ms - parallel_wall_ms

    preprocess = PreprocessResult(
        intent=preprocess_raw.get("intent", "academico"),
        sentiment=preprocess_raw.get("sentiment", "neutral"),
        clean_query=preprocess_raw.get("clean_query", request.message),
        confidence=preprocess_raw.get("confidence", 1.0),
    )

    logger.info(
        "Pre-process done | intent=%s sentiment=%s confidence=%.2f %.0fms",
        preprocess.intent, preprocess.sentiment, preprocess.confidence, preprocessing_ms,
    )
    logger.debug(
        "QUERY ORIGINAL   | interaction=%s | %s",
        request.interaction_id, request.message,
    )
    logger.debug(
        "QUERY MEJORADA   | interaction=%s | %s",
        request.interaction_id, preprocess.clean_query,
    )

    chunks = [RetrievedChunk(**c) for c in raw_chunks]
    logger.info(
        "RAG search done | materia=%s chunks=%d %.0fms",
        request.materia_id, len(chunks), rag_ms,
    )
    logger.info(
        "Parallel wall time | %.0fms (preprocess=%.0f rag=%.0f saved=~%.0fms vs serial)",
        parallel_wall_ms, preprocessing_ms, rag_ms, saved_ms,
    )

    # ── Reranking: reorder RAG_FETCH_K candidates → top RAG_TOP_K ────────────────────
    if settings.USE_RERANKER and raw_chunks:
        reranked_chunks, rerank_ms = await reranker.rerank(
            query=preprocess.clean_query,   # use cleaned query for better scoring
            chunks=raw_chunks,
            top_k=settings.RAG_TOP_K,
        )
        logger.info(
            "Reranker done | %d→%d chunks %.0fms | top score=%.3f",
            len(raw_chunks), len(reranked_chunks), rerank_ms,
            reranked_chunks[0].get("rerank_score", 0) if reranked_chunks else 0,
        )
        for i, chunk in enumerate(reranked_chunks, 1):
            logger.debug(
                "RERANKED [%d/%d] | score=%.3f dist=%.4f | fuente=%s | %s",
                i, len(reranked_chunks),
                chunk.get("rerank_score", 0),
                chunk.get("distance", 0),
                chunk.get("source", "?"),
                chunk.get("content", "")[:200].replace("\n", " "),
            )
        final_chunks = reranked_chunks
    else:
        rerank_ms = 0
        final_chunks = raw_chunks[:settings.RAG_TOP_K]
    # Log raw ChromaDB candidates (before reranking)
    for i, chunk in enumerate(raw_chunks, 1):
        logger.debug(
            "FRAGMENTO [%d/%d] | fuente=%s dist=%.4f | %s",
            i, len(raw_chunks),
            chunk.get("source", "?"),
            chunk.get("distance", 0),
            chunk.get("content", "")[:200].replace("\n", " "),
        )

    # If the query is off-topic, return a graceful short-circuit
    if preprocess.intent == "fuera_de_tema":
        return QueryResponse(
            interaction_id=request.interaction_id,
            materia_id=request.materia_id,
            response="¡Hola! Estoy aquí para ayudarte con dudas académicas sobre tu materia. ¿En qué puedo ayudarte?",
            intent=preprocess.intent,
            sentiment=preprocess.sentiment,
            processing_time_ms=(time.perf_counter() - t_start) * 1000,
        )

    if preprocess.intent == "saludo":
        return QueryResponse(
            interaction_id=request.interaction_id,
            materia_id=request.materia_id,
            response="¡Hola! Estoy aquí para ayudarte con tu materia. ¿Qué duda tienes hoy?",
            intent=preprocess.intent,
            sentiment=preprocess.sentiment,
            processing_time_ms=(time.perf_counter() - t_start) * 1000,
        )

    # ── Stage 3: LLM Response Generation ─────────────────────────────────────
    t0 = time.perf_counter()
    response_text = await llm_service.generate_rag_response(
        original_message=request.message,
        clean_query=preprocess.clean_query,
        sentiment=preprocess.sentiment,
        intent=preprocess.intent,
        context_chunks=final_chunks,        # reranked top-K (or raw top-K if reranker off)
        materia_id=request.materia_id,
        interaction_id=request.interaction_id,
    )
    llm_ms = round((time.perf_counter() - t0) * 1000)

    total_ms = round((time.perf_counter() - t_start) * 1000)
    logger.debug(
        "RESPUESTA LLM    | interaction=%s | %s",
        request.interaction_id, response_text[:500].replace("\n", " "),
    )
    logger.info(
        "Pipeline complete | interaction=%s total=%.0fms (parallel=%.0f rerank=%.0f llm=%.0f)",
        request.interaction_id, total_ms, parallel_wall_ms, rerank_ms, llm_ms,
    )

    return QueryResponse(
        interaction_id=request.interaction_id,
        materia_id=request.materia_id,
        response=response_text,
        intent=preprocess.intent,
        sentiment=preprocess.sentiment,
        processing_time_ms=total_ms,
    ), {
        "_preprocessing_ms": preprocessing_ms,
        "_rag_ms": rag_ms,
        "_rerank_ms": rerank_ms,
        "_parallel_wall_ms": parallel_wall_ms,
        "_llm_ms": llm_ms,
        "_chunks": [c.model_dump() for c in chunks],
        "_clean_query": preprocess.clean_query,
    }
