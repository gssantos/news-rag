import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Add project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Base and models - no app settings dependency
from app.db.base import Base

# Load environment variables if .env exists (for local CLI usage)
load_dotenv()

# this is the Alembic Config object.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None and os.path.exists(config.config_file_name):
    fileConfig(config.config_file_name)


# Get database URL from environment - no app settings dependency
def get_database_url():
    """Get DATABASE_URL from environment variables."""
    return os.getenv("DATABASE_URL")


# Set the database URL dynamically from environment
db_url = get_database_url()
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# add your model's MetaData object here
target_metadata = Base.metadata


# Custom rendering function to ensure pgvector types are imported in migrations
def render_item(type_, obj, autogen_context):
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector.sqlalchemy."):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.{obj.__class__.__name__}"
    return False  # Continue with default rendering


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations asynchronously."""

    # Create async engine using async_engine_from_config
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
