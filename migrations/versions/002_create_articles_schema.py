"""Create articles table and indexes

Revision ID: 002_create_articles_schema
Revises: 001_enable_pgvector
Create Date: 2025-09-09T19:49:06.392943

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Import the Vector type required for the migration
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

try:
    emb_dim_str = os.getenv("EMB_DIM", "1536")
    EMB_DIM = int(emb_dim_str)
    if EMB_DIM <= 0:
        raise ValueError(f"EMB_DIM must be positive, got {EMB_DIM}")
except (ValueError, TypeError):
    # Fallback to default if environment variable is invalid
    EMB_DIM = 1536  # Default dimension for text-embedding-3-small

IVFFLAT_LISTS = 200  # Requirement: default 200

# revision identifiers, used by Alembic.
revision: str = "002_create_articles_schema"
down_revision: Union[str, None] = "001_enable_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Requirement: Migration 2: create articles schema
    op.create_table(
        "articles",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), nullable=False, primary_key=True
        ),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source_domain", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("llm_model", sa.String(length=255), nullable=False),
        sa.Column("embed_model", sa.String(length=255), nullable=False),
        sa.Column("embed_dim", sa.Integer(), nullable=False),
        # Define the embedding column with the specific dimension
        sa.Column("embedding", Vector(EMB_DIM), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # Create indexes
    # Requirement: unique index on url
    op.create_index(op.f("ix_articles_url"), "articles", ["url"], unique=True)
    # Requirement: btree indexes on published_at
    op.create_index(
        op.f("ix_articles_published_at"), "articles", ["published_at"], unique=False
    )

    # Requirement: vector index on embedding using ivfflat with vector_cosine_ops
    op.create_index(
        "ix_articles_embedding_ivfflat",
        "articles",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_with={"lists": IVFFLAT_LISTS},
        postgresql_ops={"embedding": "vector_cosine_ops"},
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_articles_embedding_ivfflat",
        table_name="articles",
        postgresql_using="ivfflat",
    )
    op.drop_index(op.f("ix_articles_published_at"), table_name="articles")
    op.drop_index(op.f("ix_articles_url"), table_name="articles")
    op.drop_table("articles")
