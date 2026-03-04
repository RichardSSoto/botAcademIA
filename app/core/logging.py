"""
Structured logging configuration for BotAcademia.

Features:
  - Console handler  : coloured text (DEBUG) or JSON (production).
  - Rotating file handler per service: /app/logs/<service>.log
      max 10 MB per file, 5 rotations kept → 50 MB cap per service.
  - One shared root setup + per-service named loggers.

Services / log files:
  api       → fastapi request handling, lifespan events
  ingest    → vectorisation pipeline (ingest_service, vector_store)
  pipeline  → orchestration / pipeline.py
  kafka     → producer & consumer workers
  rag       → RAG retrieval + LLM calls
  db        → database / SQLAlchemy events
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from app.core.config import settings

# Where log files land.  Docker mounts ./logs → /app/logs so they appear
# on the host inside the project's logs/ folder.
LOG_DIR = Path(os.getenv("LOG_DIR", "/app/logs"))

# Shared formatters
_DEV_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
_JSON_FMT = logging.Formatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","service":"%(service)s",'
    '"logger":"%(name)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_FILE_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Map logger-name prefixes → service log file
_SERVICE_MAP: dict[str, str] = {
    "app.api":                       "api",
    "app.core":                      "api",
    "uvicorn":                       "api",
    "fastapi":                       "api",
    "app.services.ingest_service":   "ingest",
    "app.services.vector_store":     "ingest",
    "app.services.pipeline":         "pipeline",
    "app.workers.kafka_producer":    "kafka",
    "app.workers.kafka_consumer":    "kafka",
    "app.workers":                   "kafka",
    "app.services.llm_service":      "rag",
    "app.services.redis_cache":      "rag",
    "app.core.database":             "db",
    "sqlalchemy":                    "db",
    "alembic":                       "db",
}

_file_handlers: dict[str, logging.Handler] = {}


def _get_file_handler(service: str) -> logging.Handler:
    """Return (cached) rotating file handler for *service*."""
    if service not in _file_handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{service}.log"
        h = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        h.setFormatter(_FILE_FMT)
        _file_handlers[service] = h
    return _file_handlers[service]


def setup_logging() -> None:
    """Configure root logger + per-service file handlers."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(_DEV_FMT if settings.DEBUG else _JSON_FMT)
    root.addHandler(console)

    # ── Per-service rotating file handlers ───────────────────────────────
    # Attach a file handler to every known service-prefix logger directly
    # so each gets its own file regardless of root propagation.
    for prefix, service in _SERVICE_MAP.items():
        svc_logger = logging.getLogger(prefix)
        svc_logger.setLevel(log_level)
        fh = _get_file_handler(service)
        # Avoid duplicate handlers if setup_logging() is called twice
        if not any(isinstance(h, logging.handlers.RotatingFileHandler)
                   and getattr(h, 'baseFilename', None) == str(fh.baseFilename)  # type: ignore[attr-defined]
                   for h in svc_logger.handlers):
            svc_logger.addHandler(fh)

    # ── Shared "app" file for anything not covered above ────────────────
    app_fh = _get_file_handler("app")
    app_logger = logging.getLogger("app")
    if not any(isinstance(h, logging.handlers.RotatingFileHandler)
               for h in app_logger.handlers):
        app_logger.addHandler(app_fh)

    # ── Suppress noisy third-party loggers ───────────────────────────────
    for noisy in ("httpx", "httpcore", "chromadb.telemetry", "chromadb.segment"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Reduce uvicorn.access verbosity in non-debug mode
    if not settings.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("app.core").info(
        "Logging initialised | level=%s | log_dir=%s", settings.LOG_LEVEL, LOG_DIR
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; file routing is handled by setup_logging()."""
    return logging.getLogger(name)
