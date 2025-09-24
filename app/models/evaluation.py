import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EvaluationType(enum.Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    END_TO_END = "end_to_end"


class GoldenDataset(Base):
    __tablename__ = "golden_datasets"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Dataset metadata
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    dataset_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    # Relationships
    queries: Mapped[list["GoldenQuery"]] = relationship(
        "GoldenQuery",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    runs: Mapped[list["EvaluationRun"]] = relationship(
        "EvaluationRun",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<GoldenDataset(id={self.id}, name='{self.name}')>"


class GoldenQuery(Base):
    __tablename__ = "golden_queries"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Foreign Keys
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("golden_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Query and Expectations
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Expected result can be structured; JSONB enables flexible evaluation schemas
    expected_result: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )
    # Optional plain-text expected answer (some evaluations might use exact text)
    expected_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Expected article IDs and context
    expected_article_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    query_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    # Relationships
    dataset: Mapped["GoldenDataset"] = relationship(
        "GoldenDataset", back_populates="queries"
    )
    results: Mapped[list["EvaluationResult"]] = relationship(
        "EvaluationResult",
        back_populates="query",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        qt = (
            (self.query_text[:50] + "...")
            if len(self.query_text) > 50
            else self.query_text
        )
        return f"<GoldenQuery(id={self.id}, dataset_id={self.dataset_id}, query_text='{qt}')>"


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Foreign Keys
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("golden_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Run configuration and status
    evaluation_type: Mapped[EvaluationType] = mapped_column(
        SAEnum(EvaluationType, name="evaluation_type"), nullable=False
    )
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_model: Mapped[str] = mapped_column(String(255), nullable=False)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mlflow_experiment_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    dataset: Mapped["GoldenDataset"] = relationship(
        "GoldenDataset", back_populates="runs"
    )
    results: Mapped[list["EvaluationResult"]] = relationship(
        "EvaluationResult",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<EvaluationRun(id={self.id}, dataset_id={self.dataset_id}, evaluation_type={self.evaluation_type.value}, status={self.status})>"


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Foreign Keys
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("golden_queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Result data
    retrieved_article_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    generated_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    # Relationships
    run: Mapped["EvaluationRun"] = relationship(
        "EvaluationRun", back_populates="results"
    )
    query: Mapped["GoldenQuery"] = relationship("GoldenQuery", back_populates="results")

    __table_args__ = (
        # Ensure one result per (run, query)
        UniqueConstraint("run_id", "query_id", name="uq_evaluation_results_run_query"),
    )

    def __repr__(self) -> str:
        return f"<EvaluationResult(id={self.id}, run_id={self.run_id}, query_id={self.query_id})>"
