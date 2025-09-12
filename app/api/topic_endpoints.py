import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.scheduler import scheduler
from app.core.security import get_api_key
from app.db.session import get_db_session
from app.models.topic import IngestionRun, NewsSource, Topic
from app.schemas.topic import (
    IngestionRunResponse,
    IngestionStatsResponse,
    NewsSourceCreateRequest,
    NewsSourceResponse,
    TopicCreateRequest,
    TopicDetailResponse,
    TopicListResponse,
    TopicResponse,
    TopicUpdateRequest,
)
from app.services.scheduled_ingestion_service import ScheduledIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/topics", tags=["Topics"], dependencies=[Depends(get_api_key)]
)


@router.post(
    "",
    response_model=TopicResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new topic",
)
async def create_topic(
    request: TopicCreateRequest, db: AsyncSession = Depends(get_db_session)
):
    """Create a new topic for automatic article ingestion."""
    # Check if topic with same name or slug exists
    stmt = select(Topic).where(
        (Topic.name == request.name) | (Topic.slug == request.slug)
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Topic with this name or slug already exists",
        )

    # Create new topic
    topic = Topic(
        name=request.name,
        slug=request.slug,
        description=request.description,
        keywords=request.keywords,
        is_active=request.is_active,
        schedule_interval_minutes=request.schedule_interval_minutes,
    )

    db.add(topic)
    await db.commit()
    await db.refresh(topic)

    # Schedule if active
    if topic.is_active:
        scheduler.schedule_topic(topic)

    return TopicResponse.model_validate(topic)


@router.get("", response_model=TopicListResponse, summary="List all topics")
async def list_topics(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    skip: int = Query(0, ge=0, description="Number of topics to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of topics to return"),
    db: AsyncSession = Depends(get_db_session),
):
    """List all topics with optional filtering."""
    stmt = select(Topic)

    if is_active is not None:
        stmt = stmt.where(Topic.is_active == is_active)

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    topics = result.scalars().all()

    return TopicListResponse(topics=[TopicResponse.model_validate(t) for t in topics])


@router.get(
    "/{topic_id}", response_model=TopicDetailResponse, summary="Get topic details"
)
async def get_topic(topic_id: UUID, db: AsyncSession = Depends(get_db_session)):
    """Get detailed information about a specific topic."""
    stmt = (
        select(Topic).where(Topic.id == topic_id).options(selectinload(Topic.sources))
    )
    result = await db.execute(stmt)
    topic = result.scalar_one_or_none()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    return TopicDetailResponse.model_validate(topic)


@router.patch(
    "/{topic_id}",
    response_model=TopicResponse,
    summary="Update a topic",
    description="Update an existing topic by ID.",
)
async def update_topic(
    topic_id: UUID,
    request: TopicUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Update an existing topic."""
    topic = await db.get(Topic, topic_id)

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    # Update fields if provided
    if request.name is not None:
        topic.name = request.name
    if request.description is not None:
        topic.description = request.description
    if request.keywords is not None:
        topic.keywords = request.keywords
    if request.is_active is not None:
        topic.is_active = request.is_active
    if request.schedule_interval_minutes is not None:
        topic.schedule_interval_minutes = request.schedule_interval_minutes

    topic.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(topic)

    # Update scheduler
    if topic.is_active:
        scheduler.schedule_topic(topic)
    else:
        scheduler.remove_topic(str(topic.id))

    return TopicResponse.model_validate(topic)


@router.delete(
    "/{topic_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a topic"
)
async def delete_topic(topic_id: UUID, db: AsyncSession = Depends(get_db_session)):
    """Delete a topic and all associated data."""
    topic = await db.get(Topic, topic_id)

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    # Remove from scheduler
    scheduler.remove_topic(str(topic_id))

    # Delete topic (cascade will handle related records)
    await db.delete(topic)
    await db.commit()


@router.post(
    "/{topic_id}/sources",
    response_model=NewsSourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a source to a topic",
)
async def add_source(
    topic_id: UUID,
    request: NewsSourceCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Add a new source to a topic."""
    topic = await db.get(Topic, topic_id)

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    source = NewsSource(
        topic_id=topic_id,
        name=request.name,
        url=request.url,
        source_type=request.source_type,
        config=request.config or {},
        is_active=request.is_active,
    )

    db.add(source)
    await db.commit()
    await db.refresh(source)

    return NewsSourceResponse.model_validate(source)


@router.get(
    "/{topic_id}/sources",
    response_model=List[NewsSourceResponse],
    summary="List sources for a topic",
)
async def list_sources(topic_id: UUID, db: AsyncSession = Depends(get_db_session)):
    """List all sources for a specific topic."""
    stmt = select(NewsSource).where(NewsSource.topic_id == topic_id)
    result = await db.execute(stmt)
    sources = result.scalars().all()

    return [NewsSourceResponse.model_validate(s) for s in sources]


@router.delete(
    "/{topic_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a source",
)
async def delete_source(
    topic_id: UUID, source_id: UUID, db: AsyncSession = Depends(get_db_session)
):
    """Delete a source from a topic."""
    stmt = select(NewsSource).where(
        (NewsSource.id == source_id) & (NewsSource.topic_id == topic_id)
    )
    result = await db.execute(stmt)
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source not found"
        )

    await db.delete(source)
    await db.commit()


@router.post(
    "/{topic_id}/ingest",
    response_model=IngestionRunResponse,
    summary="Trigger manual ingestion",
)
async def trigger_ingestion(topic_id: UUID, db: AsyncSession = Depends(get_db_session)):
    """Manually trigger ingestion for a specific topic."""
    topic = await db.get(Topic, topic_id)

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found"
        )

    # Run ingestion
    async with ScheduledIngestionService() as service:
        run = await service.ingest_topic(str(topic_id), db)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run ingestion",
        )

    return IngestionRunResponse.model_validate(run)


@router.get(
    "/{topic_id}/runs",
    response_model=List[IngestionRunResponse],
    summary="Get ingestion history",
)
async def get_ingestion_runs(
    topic_id: UUID,
    limit: int = Query(10, ge=1, le=100, description="Number of runs to return"),
    db: AsyncSession = Depends(get_db_session),
):
    """Get ingestion run history for a topic."""
    stmt = (
        select(IngestionRun)
        .where(IngestionRun.topic_id == topic_id)
        .order_by(IngestionRun.started_at.desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    runs = result.scalars().all()

    return [IngestionRunResponse.model_validate(r) for r in runs]


@router.get(
    "/stats/summary",
    response_model=IngestionStatsResponse,
    summary="Get ingestion statistics",
)
async def get_ingestion_stats(db: AsyncSession = Depends(get_db_session)):
    """Get overall ingestion statistics."""
    # Count active topics
    active_topics_stmt = select(Topic).where(Topic.is_active)
    active_topics_result = await db.execute(active_topics_stmt)
    active_topics = len(active_topics_result.scalars().all())

    # Count total topics
    total_topics_stmt = select(Topic)
    total_topics_result = await db.execute(total_topics_stmt)
    total_topics = len(total_topics_result.scalars().all())

    # Get recent ingestion runs
    recent_runs_stmt = (
        select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(100)
    )
    recent_runs_result = await db.execute(recent_runs_stmt)
    recent_runs = recent_runs_result.scalars().all()

    # Calculate statistics
    total_articles_ingested = sum(r.articles_ingested for r in recent_runs)
    total_articles_failed = sum(r.articles_failed for r in recent_runs)

    return IngestionStatsResponse(
        total_topics=total_topics,
        active_topics=active_topics,
        total_articles_ingested_recently=total_articles_ingested,
        total_articles_failed_recently=total_articles_failed,
        recent_runs_count=len(recent_runs),
    )
