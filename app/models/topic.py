import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SourceType(str, Enum):
    RSS = "rss"
    API = "api"
    WEB_SCRAPE = "web_scrape"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some articles ingested, some failed


class Topic(Base):
    __tablename__ = "topics"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )

    # Configuration
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Schedule configuration (cron expression or interval in minutes)
    schedule_interval_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60  # Default hourly
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    last_ingestion_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    sources: Mapped[list["NewsSource"]] = relationship(
        "NewsSource", back_populates="topic", cascade="all, delete-orphan"
    )
    ingestion_runs: Mapped[list["IngestionRun"]] = relationship(
        "IngestionRun", back_populates="topic", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Topic(id={self.id}, name='{self.name}', active={self.is_active})>"


class NewsSource(Base):
    __tablename__ = "news_sources"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )

    # Source Configuration
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        String(20), nullable=False, default=SourceType.RSS
    )

    # Additional configuration (API keys, headers, etc.)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_successful_fetch: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    consecutive_failures: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="sources")

    # Indexes
    __table_args__ = (
        Index("ix_news_sources_topic_id", "topic_id"),
        Index("ix_news_sources_source_type", "source_type"),
    )

    def __repr__(self):
        return (
            f"<NewsSource(id={self.id}, name='{self.name}', type={self.source_type})>"
        )


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )

    # Run metadata
    status: Mapped[IngestionStatus] = mapped_column(
        String(20), nullable=False, default=IngestionStatus.PENDING
    )

    # Statistics
    articles_discovered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    articles_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    articles_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    articles_duplicates: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Error tracking
    error_messages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="ingestion_runs")

    # Indexes
    __table_args__ = (
        Index("ix_ingestion_runs_topic_id", "topic_id"),
        Index("ix_ingestion_runs_started_at", "started_at"),
        Index("ix_ingestion_runs_status", "status"),
    )

    def __repr__(self):
        return f"<IngestionRun(id={self.id}, topic_id={self.topic_id}, status={self.status})>"


class ArticleTopic(Base):
    """Association table for many-to-many relationship between Articles and Topics"""

    __tablename__ = "article_topics"

    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    ingestion_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    __table_args__ = (
        Index("ix_article_topics_article_id", "article_id"),
        Index("ix_article_topics_topic_id", "topic_id"),
    )
