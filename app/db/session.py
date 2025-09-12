import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create the async engine
# Requirement: SQLAlchemy 2.x with async engine (asyncpg driver).
engine = create_async_engine(
    str(settings.DATABASE_URL),
    # Requirement: Connection pool defaults: pool_size 10, max_overflow 10
    pool_size=10,
    max_overflow=10,
    pool_recycle=3600,  # Recycle connections hourly
    echo=settings.LOG_LEVEL == "DEBUG",
)

# Create a sessionmaker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}", exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()
