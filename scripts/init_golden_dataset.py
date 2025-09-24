#!/usr/bin/env python
"""Initialize a golden dataset for evaluation."""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.article import Article
from app.models.evaluation import GoldenDataset
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import get_llm_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_initial_golden_dataset():
    """Create an initial golden dataset with example queries."""

    engine = create_async_engine(str(settings.DATABASE_URL))
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as db:
        # Check if dataset already exists
        result = await db.execute(
            select(GoldenDataset).where(GoldenDataset.name == "Initial Golden Dataset")
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info(f"Golden dataset already exists: {existing.id}")
            return existing

        # Get some existing articles to use as ground truth
        result = await db.execute(select(Article).limit(10))
        articles = result.scalars().all()

        if len(articles) < 5:
            logger.error(
                "Not enough articles in database. Please ingest some articles first."
            )
            return None

        # Create queries based on existing articles
        queries = [
            {
                "query_text": "What are the recent developments in Red Sea freight routes?",
                "expected_answer": "Recent developments in Red Sea freight routes include disruptions due to geopolitical tensions, with many shipping companies rerouting vessels around the Cape of Good Hope to avoid the Suez Canal area.",
                "expected_article_ids": [str(articles[0].id)] if articles else [],
                "tags": ["freight", "shipping", "red-sea"],
            },
            {
                "query_text": "How is AI transforming the logistics industry?",
                "expected_answer": "AI is transforming logistics through route optimization, predictive maintenance, automated warehousing, and improved demand forecasting, leading to increased efficiency and reduced costs.",
                "expected_article_ids": (
                    [str(articles[1].id)] if len(articles) > 1 else []
                ),
                "tags": ["ai", "logistics", "technology"],
            },
            {
                "query_text": "What are the impacts of climate change on global supply chains?",
                "expected_answer": "Climate change is affecting global supply chains through extreme weather events, disrupting transportation routes, damaging infrastructure, and forcing companies to build more resilient and sustainable operations.",
                "expected_article_ids": (
                    [str(articles[2].id)] if len(articles) > 2 else []
                ),
                "tags": ["climate", "supply-chain", "sustainability"],
            },
            {
                "query_text": "What are the latest developments in autonomous vehicles for freight?",
                "expected_answer": "Autonomous vehicles in freight are advancing with trials of self-driving trucks on highways, automated port operations, and last-mile delivery robots, though regulatory challenges remain.",
                "expected_article_ids": (
                    [str(articles[3].id)] if len(articles) > 3 else []
                ),
                "tags": ["autonomous", "freight", "technology"],
            },
            {
                "query_text": "How are port congestions affecting global trade?",
                "expected_answer": "Port congestions are causing significant delays in global trade, increasing shipping costs, and forcing companies to diversify their supply chains and seek alternative transportation routes.",
                "expected_article_ids": (
                    [str(articles[4].id)] if len(articles) > 4 else []
                ),
                "tags": ["ports", "congestion", "trade"],
            },
        ]

        # Convert string IDs to UUIDs
        for query in queries:
            if query.get("expected_article_ids"):
                query["expected_article_ids"] = [
                    UUID(id_str) for id_str in query["expected_article_ids"]
                ]

        # Create the golden dataset
        llm_service = get_llm_service()
        eval_service = EvaluationService(db, llm_service)

        dataset = await eval_service.create_golden_dataset(
            name="Initial Golden Dataset",
            description="Initial dataset for evaluating the news RAG system with freight and logistics queries",
            version="1.0.0",
            queries=queries,
        )

        logger.info(f"Created golden dataset: {dataset.id}")
        return dataset


async def main():
    """Main function."""
    try:
        dataset = await create_initial_golden_dataset()
        if dataset:
            logger.info(
                f"Successfully created golden dataset: {dataset.name} (ID: {dataset.id})"
            )
        else:
            logger.error("Failed to create golden dataset")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating golden dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
