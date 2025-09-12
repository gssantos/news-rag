import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from dateutil.parser import ParserError
from dateutil.parser import parse as parse_date
from fastapi import HTTPException, status

# LangChain Loaders
from langchain_community.document_loaders import NewsURLLoader, WebBaseLoader
from langchain_core.documents import Document
from sqlalchemy import literal_column, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import validate_url_security
from app.models.article import Article
from app.schemas.article import ArticleIngestRequest
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, db: AsyncSession, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service

    async def ingest_url(self, request: ArticleIngestRequest) -> tuple[Article, str]:
        """
        Orchestrates the ingestion pipeline.
        Returns the Article object and the status ('created', 'updated', or 'existing').
        """
        url = str(request.url)
        logger.info(f"Starting ingestion for URL: {url}, Force: {request.force}")

        # 1. Validate URL (SSRF and Allowlist) - Async validation
        await validate_url_security(url)

        # 2. Check for existing record
        existing_article = await self._find_article_by_url(url)

        if existing_article and not request.force:
            logger.info(
                f"Article already exists and force=False. Returning existing record {existing_article.id}."
            )
            # Requirement 3: If URL already exists and force is false, return the existing record
            return existing_article, "existing"

        # 3. Fetch and Extract
        try:
            document, metadata = await self._fetch_and_extract(
                url, request.published_at
            )
        except HTTPException:
            raise  # Re-raise HTTP exceptions from fetcher (e.g., 400, 504)
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}", exc_info=True)
            # Requirement 3: 422 content extraction failed
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract content from URL.",
            )

        # 4. Summarize
        summary = await self.llm_service.generate_summary(
            title=metadata["title"],
            content=document.page_content,
            source_domain=metadata["source_domain"],
            published_at=(
                str(metadata["published_at"]) if metadata["published_at"] else None
            ),
        )

        # 5. Embed
        embedding_input = self.llm_service.compose_embedding_input(
            title=metadata["title"], summary=summary, content=document.page_content
        )
        embedding = await self.llm_service.generate_embedding(embedding_input)

        # 6. Persist (UPSERT)
        article_data = {
            "url": url,
            "source_domain": metadata["source_domain"],
            "title": metadata["title"],
            "published_at": metadata["published_at"],
            "content": document.page_content,
            "summary": summary,
            "llm_model": self.llm_service.llm_model_name,
            "embed_model": self.llm_service.emb_model_name,
            "embed_dim": self.llm_service.emb_dim,
            "embedding": embedding,
            "fetched_at": datetime.now(timezone.utc),
        }

        # Requirement 13: Upserts using ON CONFLICT (url) DO UPDATE
        logger.info("Persisting article data (UPSERT).")
        article, is_new = await self._upsert_article(article_data)

        status_str = "created" if is_new else "updated"
        logger.info(
            f"Ingestion successful. Article ID: {article.id}. Status: {status_str}"
        )
        return article, status_str

    async def _fetch_and_extract(
        self, url: str, override_published_at: datetime | None
    ) -> tuple[Document, dict]:
        logger.info(f"Fetching content from {url}...")

        # Requirement 6: Primary: NewsURLLoader
        try:
            loader = NewsURLLoader(
                urls=[url],
                browser_user_agent="Mozilla/5.0 (compatible; SimpleNewsRAG/0.1;)",
            )
            # Newspaper3k is synchronous. We MUST run it in a thread pool.
            # Requirement: Async-first implementation
            documents = await asyncio.wait_for(
                asyncio.to_thread(loader.load),
                timeout=settings.HTTP_FETCH_TIMEOUT_SECONDS,
            )

            if not documents or not documents[0].page_content.strip():
                raise ValueError("NewsURLLoader extracted empty content.")

            document = documents[0]
            metadata = self._extract_metadata(
                document.metadata, url, override_published_at
            )
            logger.info("Successfully extracted content using NewsURLLoader.")
            return document, metadata

        except asyncio.TimeoutError:
            logger.warning(f"NewsURLLoader timed out for {url}. Falling back.")
        except Exception as e:
            logger.warning(f"NewsURLLoader failed for {url}: {e}. Falling back.")

        # Requirement 6: Fallback: WebBaseLoader
        try:
            web_loader = WebBaseLoader(url)
            web_loader.requests_kwargs = {
                "timeout": settings.HTTP_FETCH_TIMEOUT_SECONDS
            }
            # Use sync load() in thread executor to avoid event loop conflict
            documents = await asyncio.wait_for(
                asyncio.to_thread(web_loader.load),
                timeout=settings.HTTP_FETCH_TIMEOUT_SECONDS,
            )

            if not documents or not documents[0].page_content.strip():
                raise ValueError("WebBaseLoader extracted empty content.")

            document = documents[0]
            metadata = self._extract_metadata(
                document.metadata, url, override_published_at
            )
            logger.info("Successfully extracted content using WebBaseLoader.")
            return document, metadata

        except asyncio.TimeoutError:
            # Requirement 3: 504 fetch timeout
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timeout while fetching URL.",
            )
        except Exception as e_fallback:
            logger.error(f"WebBaseLoader fallback failed for {url}: {e_fallback}")
            raise ValueError("Both primary and fallback loaders failed.")

    def _extract_metadata(
        self, loader_metadata: dict, url: str, override_published_at: datetime | None
    ) -> dict:
        # Requirement 3: Extract metadata: title, source domain, and published date
        title = loader_metadata.get("title")
        # Source domain extraction is handled in validate_url_security, but we re-extract here for completeness if needed.
        # We rely on the validated hostname from the security check if available, or re-parse.
        from urllib.parse import urlparse

        source_domain = urlparse(url).netloc

        # Determine published date (Requirement 3: fallback logic)
        published_at = override_published_at

        if not published_at:
            # Try extracting from metadata (e.g., 'publish_date' from NewsURLLoader)
            date_obj = loader_metadata.get("publish_date") or loader_metadata.get(
                "date"
            )
            if date_obj:
                try:
                    if isinstance(date_obj, str):
                        # Use dateutil for robust parsing
                        parsed_date = parse_date(date_obj)
                    elif isinstance(date_obj, datetime):
                        parsed_date = date_obj
                    else:
                        parsed_date = None

                    if parsed_date:
                        # Ensure timezone awareness (assume UTC if naive)
                        if parsed_date.tzinfo is None:
                            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                        published_at = parsed_date

                except (ParserError, TypeError, ValueError):
                    logger.warning(f"Could not parse date '{date_obj}' from metadata.")

        return {
            "title": title,
            "source_domain": source_domain,
            "published_at": published_at,
        }

    async def _find_article_by_url(self, url: str) -> Article | None:
        stmt = select(Article).where(Article.url == url)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def _upsert_article(self, data: dict) -> tuple[Article, bool]:
        # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE
        stmt = insert(Article).values(data)

        # Define the update columns (exclude primary key, unique key, and created_at)
        update_dict: Dict[str, Any] = {
            c.name: c
            for c in stmt.excluded
            if c.name not in ["id", "url", "created_at"]
        }
        # Explicitly update 'updated_at' timestamp
        update_dict["updated_at"] = datetime.now(timezone.utc)

        # Use RETURNING clause with xmax = 0 to detect inserts vs updates
        on_conflict_stmt: Any = stmt.on_conflict_do_update(
            index_elements=["url"], set_=update_dict
        ).returning(Article, literal_column("(xmax = 0)").label("inserted"))

        try:
            # Execute the UPSERT
            result = await self.db.execute(on_conflict_stmt)
            await self.db.commit()

            row = result.fetchone()
            if not row:
                raise Exception("Database upsert failed to return the article.")

            # Extract the Article object and is_new status directly from the result
            article = row[0]  # The Article object from RETURNING Article
            is_new = row[1]  # The is_new boolean from RETURNING (xmax = 0)

            return article, is_new

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Database error during upsert: {e}", exc_info=True)
            # Requirement: 500 unexpected
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist article data.",
            )
