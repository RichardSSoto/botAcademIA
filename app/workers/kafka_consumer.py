"""
Kafka Consumers – production event-driven workers.
Only active when USE_KAFKA=True.

Worker 1 (preprocessor_consumer):
  Reads from 'botacademia.incoming_messages'
  → Calls LLM pre-processor
  → Publishes to 'botacademia.processed_queries'

Worker 2 (rag_consumer):
  Reads from 'botacademia.processed_queries'
  → Runs RAG vector search + LLM generation
  → Stores response in PostgreSQL
  → (future) pushes result back to BOT LUA via webhook

Both workers use:
  enable_auto_commit=False  – manual commit for at-least-once delivery
  max_poll_interval_ms      – avoid rebalance during LLM calls
"""
import asyncio
import json
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.models.db_models import QueryLog
from app.services import llm_service, vector_store
from app.services.redis_cache import get_session_context, update_session_context
from app.workers.kafka_producer import publish_processed_query

logger = get_logger(__name__)

_consumer_tasks: list[asyncio.Task] = []


# ── Worker 1: Pre-processor ───────────────────────────────────────────────────

async def preprocessor_consumer() -> None:
    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC_INCOMING,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID_PREPROCESSOR,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=300_000,   # 5 min – LLM calls can be slow
    )
    await consumer.start()
    logger.info("Worker 1 (preprocessor) listening on '%s'", settings.KAFKA_TOPIC_INCOMING)

    try:
        async for msg in consumer:
            payload = msg.value
            interaction_id = payload.get("interaction_id", "?")
            try:
                preprocess_result = await llm_service.preprocess_query(
                    raw_message=payload["message"],
                    materia_id=payload["materia_id"],
                )
                enriched = {**payload, **preprocess_result}
                await publish_processed_query(enriched)
                logger.info("Worker 1 processed interaction=%s", interaction_id)
                await consumer.commit()

            except Exception as exc:
                logger.error("Worker 1 error interaction=%s: %s", interaction_id, exc)
                # Do NOT commit – message will be redelivered
    finally:
        await consumer.stop()


# ── Worker 2: RAG + Response generator ───────────────────────────────────────

async def rag_consumer() -> None:
    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC_PROCESSED,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_GROUP_ID_RAG,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=300_000,
    )
    await consumer.start()
    logger.info("Worker 2 (RAG engine) listening on '%s'", settings.KAFKA_TOPIC_PROCESSED)

    try:
        async for msg in consumer:
            payload = msg.value
            interaction_id = payload.get("interaction_id", "?")
            try:
                # Vector search
                chunks = await vector_store.semantic_search_combined(
                    materia_id=payload["materia_id"],
                    query=payload["message"],
                )

                # Load session history from Redis for multi-turn context
                history = await get_session_context(interaction_id)

                # LLM generation
                response_text, _, _ = await llm_service.generate_rag_response(
                    original_message=payload["message"],
                    sentiment=payload.get("sentiment", "neutral"),
                    intent=payload.get("intent", "academico"),
                    context_chunks=chunks,
                    materia_id=payload["materia_id"],
                    interaction_id=interaction_id,
                    conversation_history=history,
                )

                # Update Redis session cache
                await update_session_context(
                    interaction_id, payload["message"], response_text,
                    max_turns=settings.SESSION_MAX_TURNS,
                )

                # Persist to PostgreSQL
                async with AsyncSessionLocal() as db:
                    stmt = select(QueryLog).where(QueryLog.interaction_id == interaction_id)
                    log_entry = (await db.execute(stmt)).scalar_one_or_none()
                    if log_entry:
                        log_entry.response = response_text
                        log_entry.intent = payload.get("intent")
                        log_entry.sentiment = payload.get("sentiment")
                        log_entry.retrieved_chunks = json.dumps(chunks, ensure_ascii=False)
                        log_entry.num_chunks = len(chunks)
                        log_entry.status = "completed"
                        await db.commit()

                logger.info("Worker 2 completed interaction=%s chunks=%d", interaction_id, len(chunks))
                await consumer.commit()

            except Exception as exc:
                logger.error("Worker 2 error interaction=%s: %s", interaction_id, exc)
    finally:
        await consumer.stop()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def start_consumers() -> None:
    _consumer_tasks.append(asyncio.create_task(preprocessor_consumer()))
    _consumer_tasks.append(asyncio.create_task(rag_consumer()))
    logger.info("Kafka consumer tasks started (%d workers)", len(_consumer_tasks))


async def stop_consumers() -> None:
    for task in _consumer_tasks:
        task.cancel()
    await asyncio.gather(*_consumer_tasks, return_exceptions=True)
    _consumer_tasks.clear()
    logger.info("Kafka consumer tasks stopped")
