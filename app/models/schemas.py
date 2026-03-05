"""
Pydantic schemas for API request/response validation.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal
from datetime import datetime


# ── Inbound (from BOT LUA) ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload sent by BOT LUA when a student asks a question."""
    interaction_id: str = Field(
        ...,
        description="Unique conversation ID. All messages sharing this ID belong to the same session.",
        examples=["lua_conv_abc123"],
    )
    materia_id: str = Field(
        ...,
        description="Subject/course ID. Must match a vectorized ChromaDB collection.",
        examples=["258_Criminologia_B"],
    )
    message: str = Field(
        default="",
        max_length=4096,
        description=(
            "Raw student message. Required and non-empty when status='active'. "
            "Can be empty or omitted when status='closed'."
        ),
        examples=["no entiendo para que sirve la criminología??"],
    )
    status: Literal["active", "closed"] = Field(
        default="active",
        description=(
            "'active' — normal conversational query. "
            "'closed' — external system signals conversation end; message may be empty."
        ),
    )

    @model_validator(mode="after")
    def _validate_message_for_active(self) -> QueryRequest:
        if self.status == "active" and not self.message.strip():
            raise ValueError("'message' is required and cannot be empty when status is 'active'")
        return self


# ── Pre-processor output ─────────────────────────────────────────────────────

class PreprocessResult(BaseModel):
    """Structured output from the AI pre-processor (Worker 1)."""
    intent: str = Field(
        ...,
        description="Classified intent: academico | saludo | queja | fuera_de_tema",
    )
    sentiment: str = Field(
        ...,
        description="Emotional state: estresado | molesto | neutral | positivo",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in the classification (0-1).",
    )


# ── RAG output ───────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single document chunk returned by ChromaDB."""
    content: str
    source: str                    # e.g. "contenido.txt" or "contenido.json"
    materia_id: str
    distance: float | None = None
    unidad: str | None = None      # e.g. "Unidad 2 Probabilidad"
    unidad_num: int | None = None  # e.g. 2
    semana: str | None = None      # e.g. "Semana 2"


class RAGResult(BaseModel):
    """Result from the vector search step."""
    chunks: list[RetrievedChunk]
    num_results: int


# ── Outbound (to BOT LUA) ────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Full response returned to BOT LUA."""
    interaction_id: str
    materia_id: str
    response: str = Field(..., description="LLM-generated tutoring answer.")
    intent: str
    sentiment: str
    processing_time_ms: float
    status: str = "completed"          # pipeline status:  completed | error
    session_status: str = "active"     # conversation lifecycle: created | active | closed | timeout
    # Token consumption for this turn (None for short-circuit intents: saludo/fuera_de_tema)
    tokens_in:    int | None = Field(None, description="Prompt tokens sent to Gemini")
    tokens_out:   int | None = Field(None, description="Response tokens received from Gemini")
    tokens_total: int | None = Field(None, description="Total tokens for this turn (in + out)")
    # True when the response was served from the semantic cache (zero LLM cost)
    cache_hit: bool = False


class ErrorResponse(BaseModel):
    detail: str
    interaction_id: str | None = None
    status: str = "error"


# ── Ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request to ingest/vectorize a materia's source files."""
    materia_id: str = Field(..., examples=["258_Criminologia_B"])
    force_reingest: bool = Field(
        default=False,
        description="If True, deletes existing collection and re-ingests.",
    )


class IngestResponse(BaseModel):
    materia_id: str
    status: str
    num_chunks: int = 0
    message: str


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, Any]
    timestamp: datetime
