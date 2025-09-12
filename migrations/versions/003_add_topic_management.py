"""Add topic management tables

Revision ID: 003_add_topic_management
Revises: 002_create_articles_schema
Create Date: 2025-01-12T10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_add_topic_management"
down_revision: Union[str, None] = "002_create_articles_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create topics table
    op.create_table(
        "topics",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), nullable=False, primary_key=True
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("keywords", postgresql.JSONB(), nullable=False, default=list),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column(
            "schedule_interval_minutes", sa.Integer(), nullable=False, default=60
        ),
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
        sa.Column("last_ingestion_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Create indexes for topics
    op.create_index(op.f("ix_topics_name"), "topics", ["name"], unique=True)
    op.create_index(op.f("ix_topics_slug"), "topics", ["slug"], unique=True)

    # Create news_sources table
    op.create_table(
        "news_sources",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), nullable=False, primary_key=True
        ),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=False, default="rss"),
        sa.Column("config", postgresql.JSONB(), nullable=False, default=dict),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("last_successful_fetch", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, default=0),
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
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
    )

    # Create indexes for news_sources
    op.create_index(op.f("ix_news_sources_topic_id"), "news_sources", ["topic_id"])
    op.create_index(
        op.f("ix_news_sources_source_type"), "news_sources", ["source_type"]
    )

    # Create ingestion_runs table
    op.create_table(
        "ingestion_runs",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), nullable=False, primary_key=True
        ),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, default="pending"),
        sa.Column("articles_discovered", sa.Integer(), nullable=False, default=0),
        sa.Column("articles_ingested", sa.Integer(), nullable=False, default=0),
        sa.Column("articles_failed", sa.Integer(), nullable=False, default=0),
        sa.Column("articles_duplicates", sa.Integer(), nullable=False, default=0),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_messages", postgresql.JSONB(), nullable=False, default=list),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
    )

    # Create indexes for ingestion_runs
    op.create_index(op.f("ix_ingestion_runs_topic_id"), "ingestion_runs", ["topic_id"])
    op.create_index(
        op.f("ix_ingestion_runs_started_at"), "ingestion_runs", ["started_at"]
    )
    op.create_index(op.f("ix_ingestion_runs_status"), "ingestion_runs", ["status"])

    # Create article_topics association table
    op.create_table(
        "article_topics",
        sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ingestion_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"], ["ingestion_runs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("article_id", "topic_id"),
    )

    # Create indexes for article_topics
    op.create_index(
        op.f("ix_article_topics_article_id"), "article_topics", ["article_id"]
    )
    op.create_index(op.f("ix_article_topics_topic_id"), "article_topics", ["topic_id"])


def downgrade() -> None:
    # Drop indexes and tables in reverse order
    op.drop_index(op.f("ix_article_topics_topic_id"), table_name="article_topics")
    op.drop_index(op.f("ix_article_topics_article_id"), table_name="article_topics")
    op.drop_table("article_topics")

    op.drop_index(op.f("ix_ingestion_runs_status"), table_name="ingestion_runs")
    op.drop_index(op.f("ix_ingestion_runs_started_at"), table_name="ingestion_runs")
    op.drop_index(op.f("ix_ingestion_runs_topic_id"), table_name="ingestion_runs")
    op.drop_table("ingestion_runs")

    op.drop_index(op.f("ix_news_sources_source_type"), table_name="news_sources")
    op.drop_index(op.f("ix_news_sources_topic_id"), table_name="news_sources")
    op.drop_table("news_sources")

    op.drop_index(op.f("ix_topics_slug"), table_name="topics")
    op.drop_index(op.f("ix_topics_name"), table_name="topics")
    op.drop_table("topics")
