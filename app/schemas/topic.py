import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.topic import IngestionStatus, SourceType


class TopicCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Topic name")
    slug: str = Field(
        ..., min_length=1, max_length=255, description="URL-friendly slug"
    )
    description: Optional[str] = Field(None, description="Topic description")
    keywords: List[str] = Field(
        default_factory=list, description="Keywords for filtering"
    )
    is_active: bool = Field(True, description="Whether the topic is active")
    schedule_interval_minutes: int = Field(
        60,
        ge=5,
        le=10080,  # 5 minutes to 1 week
        description="Ingestion interval in minutes",
    )

    @field_validator("slug")
    def validate_slug(cls, v):
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError(
                "Slug must contain only lowercase letters, numbers, and hyphens"
            )
        return v


class TopicUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    is_active: Optional[bool] = None
    schedule_interval_minutes: Optional[int] = Field(None, ge=5, le=10080)


class TopicResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    description: Optional[str]
    keywords: List[str]
    is_active: bool
    schedule_interval_minutes: int
    created_at: datetime
    updated_at: datetime
    last_ingestion_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class NewsSourceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Source name")
    url: str = Field(..., description="Source URL (RSS feed, API endpoint, etc.)")
    source_type: SourceType = Field(SourceType.RSS, description="Type of source")
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional configuration (API keys, headers, etc.)",
    )
    is_active: bool = Field(True, description="Whether the source is active")


class NewsSourceUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    url: Optional[str] = None
    source_type: Optional[SourceType] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class NewsSourceResponse(BaseModel):
    id: UUID
    topic_id: UUID
    name: str
    url: str
    source_type: SourceType
    config: Dict[str, Any]
    is_active: bool
    last_successful_fetch: Optional[datetime]
    consecutive_failures: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TopicDetailResponse(TopicResponse):
    sources: List[NewsSourceResponse] = Field(default_factory=list)


class TopicListResponse(BaseModel):
    topics: List[TopicResponse]


class IngestionRunResponse(BaseModel):
    id: UUID
    topic_id: UUID
    status: IngestionStatus
    articles_discovered: int
    articles_ingested: int
    articles_failed: int
    articles_duplicates: int
    started_at: datetime
    completed_at: Optional[datetime]
    error_messages: List[str]

    model_config = ConfigDict(from_attributes=True)


class IngestionStatsResponse(BaseModel):
    total_topics: int
    active_topics: int
    total_articles_ingested_recently: int
    total_articles_failed_recently: int
    recent_runs_count: int
