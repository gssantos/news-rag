from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

# --- Request Schemas ---


class ArticleIngestRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the article to ingest (http/https)")
    published_at: Optional[datetime] = Field(
        None, description="Optional override for the publication date (ISO 8601)"
    )
    force: bool = Field(
        False, description="If true, re-fetch and re-process even if the URL exists"
    )


# --- Response Schemas ---


class ArticleBase(BaseModel):
    id: UUID
    url: str  # Return URL as string in responses
    title: Optional[str]
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    # Configure Pydantic V2 for ORM mode
    model_config = ConfigDict(from_attributes=True)

    @field_validator("url", mode="before")
    def convert_url_to_str(cls, v: Any) -> str:
        # Handle both string input and HttpUrl objects if coming from ORM/Pydantic
        return str(v)


class ArticleIngestResponse(ArticleBase):
    summary: str


class ArticleDetailResponse(ArticleBase):
    source_domain: str
    fetched_at: datetime
    summary: str
    content: str
    llm_model: str
    embed_model: str
    embed_dim: int


class ArticleSearchResult(BaseModel):
    id: UUID
    url: str
    title: Optional[str]
    summary: str
    published_at: Optional[datetime]
    score: float = Field(
        ..., description="Similarity score (cosine similarity, 0.0 to 1.0)"
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("url", mode="before")
    def convert_url_to_str(cls, v: Any) -> str:
        return str(v)


class SearchResponse(BaseModel):
    hits: List[ArticleSearchResult]


# --- Error Schema ---
class ErrorDetail(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
