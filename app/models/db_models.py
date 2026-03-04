"""
SQLAlchemy ORM models.
Represents log/history storage in PostgreSQL.
"""
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Float, func, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class QueryLog(Base):
    """Full log of every query processed by BotAcademia."""
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    materia_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    # Original message from BOT LUA
    original_message: Mapped[str] = mapped_column(Text, nullable=False)

    # Pre-processor output
    intent: Mapped[str | None] = mapped_column(String(64))
    sentiment: Mapped[str | None] = mapped_column(String(64))

    # RAG context (stored as JSON string)
    retrieved_chunks: Mapped[str | None] = mapped_column(Text)
    num_chunks: Mapped[int | None] = mapped_column(Integer)

    # Final LLM response
    response: Mapped[str | None] = mapped_column(Text)

    # Performance metrics
    preprocessing_ms: Mapped[float | None] = mapped_column(Float)
    rag_ms: Mapped[float | None] = mapped_column(Float)
    llm_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float | None] = mapped_column(Float)

    # Status: pending | processing | completed | error
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)

    # Session tracking
    turn_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="Conversation state after this turn: created | active | closed | timeout"
    )

    # Token consumption (main RAG LLM call only; preprocess call excluded)
    tokens_in:    Mapped[int | None] = mapped_column(Integer, nullable=True, comment="Prompt tokens sent to Gemini (RAG call)")
    tokens_out:   Mapped[int | None] = mapped_column(Integer, nullable=True, comment="Response tokens received from Gemini (RAG call)")
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="tokens_in + tokens_out (RAG call)")

    # Token consumption for the preprocess call (separate Gemini invocation)
    preprocess_tokens_in:  Mapped[int | None] = mapped_column(Integer, nullable=True, comment="Prompt tokens sent to Gemini (preprocess call)")
    preprocess_tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="Response tokens from Gemini (preprocess call)")

    # Grand total: both LLM calls combined
    all_tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="preprocess + RAG tokens combined")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} interaction={self.interaction_id} status={self.status}>"


class ConversationSession(Base):
    """
    Tracks the lifecycle of a conversation session.
    One session per interaction_id; a session groups all turns (QueryLog rows)
    that share the same interaction_id.
    """
    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    materia_id: Mapped[str] = mapped_column(String(128), nullable=False)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)

    # created | active | closed (farewell / external signal) | timeout (idle cleanup)
    status: Mapped[str] = mapped_column(String(32), default="created")

    # Cumulative token consumption across all turns in this session
    total_tokens_in:  Mapped[int] = mapped_column(Integer, default=0, comment="Sum of prompt tokens across all turns")
    total_tokens_out: Mapped[int] = mapped_column(Integer, default=0, comment="Sum of response tokens across all turns")
    total_tokens:     Mapped[int] = mapped_column(Integer, default=0, comment="Sum of all tokens (in + out) in this session")

    # Running list of user messages: [{turn, message, intent, ts}, ...]
    user_messages: Mapped[list] = mapped_column(
        JSONB, default=list, server_default="'[]'::jsonb",
        comment="Array of user messages in chronological order"
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        comment="Timestamp of the last turn; used by cleanup job to detect idle sessions"
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationSession {self.interaction_id} "
            f"turns={self.turn_count} status={self.status}>"
        )


class MateriaIndex(Base):
    """Tracks which materias have been ingested into ChromaDB."""
    __tablename__ = "materia_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    materia_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    materia_nombre: Mapped[str | None] = mapped_column(String(256))
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|indexed|error
    error_message: Mapped[str | None] = mapped_column(Text)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<MateriaIndex {self.materia_id} chunks={self.num_chunks} status={self.status}>"
