"""
SQLAlchemy ORM models.
Represents log/history storage in PostgreSQL.
"""
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Float, func
from sqlalchemy.orm import Mapped, mapped_column
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
    clean_query: Mapped[str | None] = mapped_column(Text)

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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<QueryLog id={self.id} interaction={self.interaction_id} status={self.status}>"


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
