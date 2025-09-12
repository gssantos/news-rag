import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID

# Using SQLAlchemy 2.0 style (Mapped)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.db.base import Base


class Article(Base):
    __tablename__ = "articles"

    # Core Identifiers
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Requirement: unique index on url
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False)

    # Metadata
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Requirement: btree indexes on published_at
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # LLM and Embedding Metadata
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)

    # Requirement: embedding: vector(embed_dim). We explicitly use the configured dimension.
    embedding: Mapped[Vector] = mapped_column(Vector(settings.EMB_DIM), nullable=False)

    # Timestamps
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    # updated_at handled by application logic during UPSERT, but server_default ensures it's set on creation.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    # Requirement: vector index on embedding using ivfflat with vector_cosine_ops
    __table_args__ = (
        Index(
            "ix_articles_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            # Requirement: lists (tunable; default 200)
            postgresql_with={"lists": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:50] if self.title else ''}...', url='{self.url}')>"
