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

    # ChromaDB ONNX warmup: force model load now so first real query is not slow
    # Without this, the first RAG call after a restart takes ~70s loading the ONNX encoder.
    try:
        import asyncio
        from app.services.vector_store import semantic_search_combined
        logger.info("ChromaDB ONNX warmup starting...")
        await asyncio.wait_for(
            semantic_search_combined("test", "warmup", n_results=1),
            timeout=120,
        )
        logger.info("ChromaDB ONNX warmup complete ✅")
    except Exception as e:
        logger.warning("ChromaDB warmup skipped: %s", e)

    # ── ngrok tunnel (local dev only) ──────────────────────────────────────────
    # Controlled by TUNNEL_ENABLED in .env. Never runs in production.
    ngrok_tunnel = None
    if settings.TUNNEL_ENABLED:
        try:
            from pyngrok import ngrok
            if settings.NGROK_AUTHTOKEN:
                ngrok.set_auth_token(settings.NGROK_AUTHTOKEN)
            connect_kwargs = {"proto": "http"}
            if settings.NGROK_DOMAIN:
                connect_kwargs["domain"] = settings.NGROK_DOMAIN
            ngrok_tunnel = ngrok.connect(settings.NGROK_PORT, **connect_kwargs)
            public_url = ngrok_tunnel.public_url
            logger.info("=" * 62)
            logger.info("NGROK TUNNEL ACTIVE")
            logger.info("  Public URL  : %s", public_url)
            logger.info("  Query endpt : %s/api/v1/query", public_url)
            logger.info("  Swagger UI  : %s/docs", public_url)
            logger.info("  (desactiva con TUNNEL_ENABLED=false en .env)")
            logger.info("=" * 62)
        except Exception as e:
            logger.warning("Ngrok tunnel failed to start: %s", e)

    yield

    logger.info("BotAcademia Engine shutting down")
    if settings.USE_KAFKA:
        from app.workers.kafka_consumer import stop_consumers
        await stop_consumers()

    await stop_cleanup_worker()

    # Close ngrok tunnel cleanly
    if ngrok_tunnel:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()
            logger.info("Ngrok tunnel closed")
        except Exception as e:
            logger.warning("Ngrok shutdown error: %s", e)


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
