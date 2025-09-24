"""Add evaluation schema for golden datasets and evaluation runs

Revision ID: 004
Revises: 003_add_topic_management
Create Date: 2025-09-12
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003_add_topic_management"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # Create golden datasets table
    op.create_table(
        "golden_datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )

    # Create golden queries table
    op.create_table(
        "golden_queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column(
            "expected_article_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=True,
        ),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["golden_datasets.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create evaluation runs table
    op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "evaluation_type",
            postgresql.ENUM(
                "retrieval", "generation", "end_to_end", name="evaluation_type"
            ),
            nullable=False,
        ),
        sa.Column("llm_model", sa.String(255), nullable=False),
        sa.Column("embed_model", sa.String(255), nullable=False),
        sa.Column("mlflow_run_id", sa.String(255), nullable=True),
        sa.Column("mlflow_experiment_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["golden_datasets.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create evaluation results table (detailed results per query)
    op.create_table(
        "evaluation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "retrieved_article_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=True,
        ),
        sa.Column("generated_answer", sa.Text(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["golden_queries.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for performance
    op.create_index("ix_golden_datasets_name", "golden_datasets", ["name"])
    op.create_index("ix_golden_datasets_is_active", "golden_datasets", ["is_active"])
    op.create_index("ix_golden_queries_dataset_id", "golden_queries", ["dataset_id"])
    op.create_index("ix_evaluation_runs_dataset_id", "evaluation_runs", ["dataset_id"])
    op.create_index(
        "ix_evaluation_runs_mlflow_run_id", "evaluation_runs", ["mlflow_run_id"]
    )
    op.create_index("ix_evaluation_runs_status", "evaluation_runs", ["status"])
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])
    op.create_index(
        "ix_evaluation_results_query_id", "evaluation_results", ["query_id"]
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_table("golden_queries")
    op.drop_table("golden_datasets")

    # Drop enum type
    op.execute("DROP TYPE IF EXISTS evaluation_type")
