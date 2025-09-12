import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_api_key
from app.db.session import get_db_session
from app.models.article import Article
from app.schemas.article import (
    ArticleDetailResponse,
    ArticleIngestRequest,
    ArticleIngestResponse,
    ErrorDetail,
    SearchResponse,
)
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService, get_llm_service
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[
        Depends(get_api_key)
    ]  # Apply security dependency to all routes if configured
)


# --- Helper Dependencies for Service Injection ---
def get_ingestion_service(
    db: AsyncSession = Depends(get_db_session),
    llm_service: LLMService = Depends(get_llm_service),
) -> IngestionService:
    return IngestionService(db=db, llm_service=llm_service)


def get_search_service(
    db: AsyncSession = Depends(get_db_session),
    llm_service: LLMService = Depends(get_llm_service),
) -> SearchService:
    return SearchService(db=db, llm_service=llm_service)


# --- Endpoints ---


@router.post(
    "/ingest/url",
    response_model=ArticleIngestResponse,
    # Default status code is 201, but we manage it dynamically in the handler
    responses={
        200: {"description": "Success (Existing record returned)"},
        201: {"description": "Success (New record created or updated)"},
        400: {
            "model": ErrorDetail,
            "description": "Invalid URL, unsupported scheme, or SSRF blocked",
        },
        422: {"model": ErrorDetail, "description": "Content extraction failed"},
        502: {"model": ErrorDetail, "description": "LLM or Embeddings provider error"},
        504: {"model": ErrorDetail, "description": "Fetch timeout"},
    },
)
async def ingest_article(
    request: ArticleIngestRequest,
    response: Response,
    service: IngestionService = Depends(get_ingestion_service),
):
    """Ingest a news article by URL, summarize it, embed it, and store it."""

    # Error handling for 5xx is primarily managed by the global exception handler in main.py,
    # but specific HTTPExceptions (like 422, 502) raised in services are passed through.

    article, status_str = await service.ingest_url(request)

    # Requirement 3: Response status code handling
    if status_str == "existing":
        response.status_code = status.HTTP_200_OK
    else:
        # Created or Updated
        response.status_code = status.HTTP_201_CREATED

    return ArticleIngestResponse.model_validate(article)


@router.get(
    "/content/{id}",
    response_model=ArticleDetailResponse,
    responses={
        404: {"model": ErrorDetail, "description": "Article not found"},
        400: {"model": ErrorDetail, "description": "Malformed ID"},
    },
)
async def get_article_content(id: UUID, db: AsyncSession = Depends(get_db_session)):
    """Retrieve the stored content and metadata for a specific article ID."""
    # Requirement 3: 400 malformed id (FastAPI handles UUID validation automatically)

    # Using db.get() for efficient primary key lookup
    article = await db.get(Article, id)

    if not article:
        # Requirement 3: 404 not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Article not found."
        )

    return ArticleDetailResponse.model_validate(article)


@router.get(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {
            "model": ErrorDetail,
            "description": "Missing query or invalid dates/parameters",
        },
        502: {"model": ErrorDetail, "description": "Embeddings provider error"},
    },
)
async def search_articles(
    query: str = Query(..., description="The search query string.", min_length=1),
    start_date: Optional[datetime] = Query(
        None,
        description="Filter results published after this date (ISO 8601 with timezone).",
    ),
    end_date: Optional[datetime] = Query(
        None,
        description="Filter results published before this date (ISO 8601 with timezone).",
    ),
    k: int = Query(
        1, description="Number of results to return (default 1, max 10).", ge=1, le=10
    ),
    service: SearchService = Depends(get_search_service),
):
    """Search for the closest articles to the user query using vector similarity."""
    # Requirement 3: 400 validation errors (FastAPI handles most)

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be before end_date.",
        )

    results = await service.search(
        query=query, k=k, start_date=start_date, end_date=end_date
    )
    return SearchResponse(hits=results)
