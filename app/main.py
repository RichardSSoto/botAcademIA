"""
BotAcademia Engine – FastAPI entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.database import init_db
from app.api.v1.routes import messages, ingest

setup_logging()
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 BotAcademia Engine starting up... mode=POC")

    # Initialise database tables
    await init_db()
    logger.info("PostgreSQL tables ready")

    # Kafka consumer (only if enabled in production)
    if settings.USE_KAFKA:
        from app.workers.kafka_consumer import start_consumers
        await start_consumers()
        logger.info("Kafka consumers started")

    # Session cleanup worker (always-on: closes idle Redis sessions + DB records)
    from app.workers.session_cleanup import start_cleanup_worker, stop_cleanup_worker
    await start_cleanup_worker()
    logger.info(
        "Session cleanup worker started (idle_timeout=%dmin)",
        settings.SESSION_IDLE_TIMEOUT_MINUTES,
    )

    yield

    logger.info("BotAcademia Engine shutting down")
    if settings.USE_KAFKA:
        from app.workers.kafka_consumer import stop_consumers
        await stop_consumers()

    await stop_cleanup_worker()


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "RAG-based AI tutoring engine. Processes student queries from BOT LUA "
        "using Gemini LLM + ChromaDB vector search."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS – tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(messages.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/docs"})
