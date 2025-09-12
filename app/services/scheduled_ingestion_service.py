import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import feedparser
import httpx
from pydantic import HttpUrl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.session import get_db_session
from app.models.article import Article
from app.models.topic import (
    ArticleTopic,
    IngestionRun,
    IngestionStatus,
    NewsSource,
    SourceType,
    Topic,
)
from app.schemas.article import ArticleIngestRequest
from app.services.ingestion_service import IngestionService
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class ScheduledIngestionService:
    def __init__(self):
        self.llm_service = get_llm_service()
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.HTTP_FETCH_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SimpleNewsRAG/0.1; +https://github.com/news-rag)"
            },
        )

    async def ingest_topic(self, topic_id: str, db: AsyncSession) -> IngestionRun:
        """
        Main method to ingest articles for a specific topic.
        Returns the IngestionRun with statistics.
        """
        logger.info(f"Starting scheduled ingestion for topic {topic_id}")

        # Get topic with sources
        stmt = (
            select(Topic)
            .where(Topic.id == topic_id)
            .options(selectinload(Topic.sources))
        )
        result = await db.execute(stmt)
        topic = result.scalar_one_or_none()

        if not topic or not topic.is_active:
            logger.warning(f"Topic {topic_id} not found or inactive")
            raise ValueError(f"Topic {topic_id} not found or inactive")

        # Create ingestion run
        ingestion_run = IngestionRun(
            topic_id=topic_id,
            status=IngestionStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            articles_discovered=0,
            articles_ingested=0,
            articles_failed=0,
            articles_duplicates=0,
            error_messages=[],
        )
        db.add(ingestion_run)
        await db.commit()

        # Collect articles from all sources
        all_article_urls = []
        error_messages = []

        for source in topic.sources:
            if not source.is_active:
                continue

            try:
                if source.source_type == SourceType.RSS:
                    urls = await self._fetch_rss_articles(source)
                    all_article_urls.extend(urls)

                    # Update source status on success
                    source.last_successful_fetch = datetime.now(timezone.utc)
                    source.consecutive_failures = 0

                elif source.source_type == SourceType.API:
                    # Future implementation for API sources
                    logger.info(
                        f"API source type not yet implemented for {source.name}"
                    )
                    continue

                elif source.source_type == SourceType.WEB_SCRAPE:
                    # Future implementation for web scraping
                    logger.info(
                        f"Web scrape source type not yet implemented for {source.name}"
                    )
                    continue

            except Exception as e:
                error_msg = f"Failed to fetch from {source.name}: {str(e)}"
                logger.error(error_msg)
                error_messages.append(error_msg)

                # Update source failure count
                source.consecutive_failures += 1

        # Update discovered count
        ingestion_run.articles_discovered = len(all_article_urls)

        # Ingest articles
        ingestion_service = IngestionService(db=db, llm_service=self.llm_service)

        for url in all_article_urls:
            try:
                # Check if article already exists (deduplication by URL)
                existing_article = await self._check_existing_article(db, url)

                if existing_article:
                    # Article exists, just link to topic if not already linked
                    await self._link_article_to_topic(
                        db, str(existing_article.id), topic_id, str(ingestion_run.id)
                    )
                    ingestion_run.articles_duplicates += 1
                else:
                    # Ingest new article
                    request = ArticleIngestRequest(
                        url=HttpUrl(url), published_at=None, force=False
                    )
                    article, status = await ingestion_service.ingest_url(request)

                    if article:
                        # Link article to topic
                        await self._link_article_to_topic(
                            db, str(article.id), topic_id, str(ingestion_run.id)
                        )
                        ingestion_run.articles_ingested += 1

            except Exception as e:
                error_msg = f"Failed to ingest {url}: {str(e)}"
                logger.error(error_msg)
                error_messages.append(error_msg)
                ingestion_run.articles_failed += 1

        # Update ingestion run status
        ingestion_run.completed_at = datetime.now(timezone.utc)
        ingestion_run.error_messages = error_messages

        if ingestion_run.articles_failed == 0 and len(error_messages) == 0:
            ingestion_run.status = IngestionStatus.SUCCESS
        elif ingestion_run.articles_ingested > 0:
            ingestion_run.status = IngestionStatus.PARTIAL
        else:
            ingestion_run.status = IngestionStatus.FAILED

        # Update topic last ingestion time
        topic.last_ingestion_at = datetime.now(timezone.utc)

        await db.commit()

        logger.info(
            f"Completed ingestion for topic {topic.name}: "
            f"discovered={ingestion_run.articles_discovered}, "
            f"ingested={ingestion_run.articles_ingested}, "
            f"duplicates={ingestion_run.articles_duplicates}, "
            f"failed={ingestion_run.articles_failed}"
        )

        return ingestion_run

    async def _fetch_rss_articles(self, source: NewsSource) -> List[str]:
        """Fetch article URLs from an RSS feed, filtering by date threshold."""
        urls = []

        # Calculate date threshold
        threshold_date = datetime.now(timezone.utc) - timedelta(
            hours=settings.RSS_DATE_THRESHOLD_HOURS
        )

        try:
            # Fetch RSS feed
            response = await self.http_client.get(source.url)
            response.raise_for_status()

            # Parse RSS feed
            feed = feedparser.parse(response.text)
            total_entries = len(feed.entries)

            for entry in feed.entries:
                # Check if entry has publication date and is within threshold
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    # Convert published_parsed to datetime
                    published_dt = datetime(*entry.published_parsed[:6]).replace(
                        tzinfo=timezone.utc
                    )

                    # Skip articles older than threshold
                    if published_dt < threshold_date:
                        continue

                # Extract URL
                if hasattr(entry, "link"):
                    urls.append(entry.link)
                elif hasattr(entry, "id"):
                    # Some feeds use 'id' instead of 'link'
                    urls.append(entry.id)

            logger.info(
                f"Found {len(urls)} articles from RSS feed {source.name} "
                f"(filtered from {total_entries} total entries, "
                f"threshold: {settings.RSS_DATE_THRESHOLD_HOURS}h)"
            )

        except Exception as e:
            logger.error(f"Error fetching RSS feed {source.url}: {e}")
            raise

        return urls

    async def _check_existing_article(
        self, db: AsyncSession, url: str
    ) -> Optional[Article]:
        """Check if an article with the given URL already exists."""
        stmt = select(Article).where(Article.url == url)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _link_article_to_topic(
        self, db: AsyncSession, article_id: str, topic_id: str, ingestion_run_id: str
    ):
        """Create a link between an article and a topic if it doesn't exist."""
        # Check if link already exists
        stmt = select(ArticleTopic).where(
            ArticleTopic.article_id == article_id, ArticleTopic.topic_id == topic_id
        )
        result = await db.execute(stmt)
        existing_link = result.scalar_one_or_none()

        if not existing_link:
            article_topic = ArticleTopic(
                article_id=article_id,
                topic_id=topic_id,
                ingestion_run_id=ingestion_run_id,
            )
            db.add(article_topic)
            await db.commit()

    async def ingest_all_active_topics(self) -> List[IngestionRun]:
        """Ingest articles for all active topics."""
        runs = []

        async for db in get_db_session():
            try:
                # Get all active topics
                stmt = select(Topic).where(Topic.is_active)
                result = await db.execute(stmt)
                topics = result.scalars().all()

                logger.info(f"Starting ingestion for {len(topics)} active topics")

                for topic in topics:
                    try:
                        run = await self.ingest_topic(str(topic.id), db)
                        if run:
                            runs.append(run)
                    except Exception as e:
                        logger.error(f"Failed to ingest topic {topic.name}: {e}")

            finally:
                await db.close()

        return runs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
