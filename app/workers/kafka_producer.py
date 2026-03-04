"""
Kafka Producer – publishes incoming messages to 'botacademia.incoming_messages'.
Only used when USE_KAFKA=True (production mode).

Configured for durability:
  - acks=all       : all replicas must acknowledge
  - retries=5      : retry transient failures
  - idempotence    : exactly-once delivery guarantee
"""
import json
from aiokafka import AIOKafkaProducer
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",
            enable_idempotence=True,
            retries=5,
            max_in_flight_requests_per_connection=1,  # required for idempotence
        )
        await _producer.start()
        logger.info("Kafka producer connected to %s", settings.KAFKA_BOOTSTRAP_SERVERS)
    return _producer


async def publish_incoming_message(payload: dict) -> None:
    """Publish a raw incoming query to the incoming_messages topic."""
    producer = await get_producer()
    await producer.send_and_wait(
        topic=settings.KAFKA_TOPIC_INCOMING,
        value=payload,
        key=payload.get("interaction_id", "").encode("utf-8"),
    )
    logger.debug("Published to %s: interaction=%s", settings.KAFKA_TOPIC_INCOMING, payload.get("interaction_id"))


async def publish_processed_query(payload: dict) -> None:
    """Publish pre-processed query (after Worker 1) to the processed_queries topic."""
    producer = await get_producer()
    await producer.send_and_wait(
        topic=settings.KAFKA_TOPIC_PROCESSED,
        value=payload,
        key=payload.get("interaction_id", "").encode("utf-8"),
    )


async def stop_producer() -> None:
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")
