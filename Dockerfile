# ────────────────────────────────────────────────────────────────────────────
# BotAcademia Engine – Multi-stage Docker image
# Python 3.11 slim for minimal image size
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies (PostgreSQL client for asyncpg, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ────── dependencies layer (cached unless requirements.txt changes) ──────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ────── final image ──────────────────────────────────────────────────────────
FROM deps AS final

COPY . .

# Create non-root user and required directories before switching user
RUN addgroup --system botacademia && adduser --system --ingroup botacademia botacademia \
    && mkdir -p /app/data/materias /app/data/chroma /app/logs \
    && chown -R botacademia:botacademia /app

USER botacademia

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
