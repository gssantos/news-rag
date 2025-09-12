import logging
from datetime import datetime

# Import the cosine distance function
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article import Article
from app.schemas.article import ArticleSearchResult
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, db: AsyncSession, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service

    async def search(
        self, query: str, k: int, start_date: datetime | None, end_date: datetime | None
    ) -> list[ArticleSearchResult]:
        logger.info(
            f"Starting search. Query='{query[:50]}...', K={k}, DateRange=({start_date}, {end_date})"
        )

        # 1. Generate query embedding
        # Requirement 3: Compute query embedding via LangChain (normalized in LLMService)
        query_embedding = await self.llm_service.generate_embedding(query)

        # 2. Define the similarity calculation
        # Requirement 3: Perform vector similarity search with pgvector and cosine distance
        # Cosine Distance ranges [0, 2]. Similarity = 1 - Distance.
        distance = Article.embedding.cosine_distance(query_embedding)
        similarity = (1.0 - distance).label("score")

        # Define the query statement
        stmt = select(
            Article.id,
            Article.url,
            Article.title,
            Article.summary,
            Article.published_at,
            similarity,
        )

        # 3. Apply filters
        # Requirement 3: filtered by published_at when provided
        if start_date:
            stmt = stmt.where(Article.published_at >= start_date)
        if end_date:
            stmt = stmt.where(Article.published_at <= end_date)

        # 4. Order and Limit
        # Order by distance ascending for ivfflat index efficiency
        stmt = stmt.order_by(distance).limit(k)

        # 5. Execute the query
        start_time = datetime.now()
        result = await self.db.execute(stmt)
        end_time = datetime.now()
        rows = result.fetchall()

        logger.info(
            f"Search executed in {(end_time-start_time).total_seconds()*1000:.2f}ms. Found {len(rows)} results."
        )

        # 6. Map results
        search_results = [
            ArticleSearchResult(
                id=row.id,
                url=row.url,
                title=row.title,
                summary=row.summary,
                published_at=row.published_at,
                score=row.score,
            )
            for row in rows
        ]

        return search_results
