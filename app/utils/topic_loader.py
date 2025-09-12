import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.topic import NewsSource, SourceType, Topic

logger = logging.getLogger(__name__)


class TopicLoader:
    """Utility class to load topics from YAML configuration files."""

    @staticmethod
    async def load_from_yaml(config_path: Path) -> int:
        """
        Load topics from a YAML configuration file.
        Returns the number of topics loaded.
        """
        if not config_path.exists():
            logger.info(f"Topic config file not found: {config_path}")
            return 0

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            if not config or "topics" not in config:
                logger.warning("No topics found in configuration file")
                return 0

            topics_loaded = 0
            async with AsyncSessionLocal() as db:
                for topic_config in config["topics"]:
                    try:
                        await TopicLoader._load_topic(db, topic_config)
                        topics_loaded += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to load topic {topic_config.get('name')}: {e}"
                        )

                await db.commit()

            logger.info(f"Loaded {topics_loaded} topics from configuration")
            return topics_loaded

        except Exception as e:
            logger.error(f"Failed to load topic configuration: {e}")
            return 0

    @staticmethod
    async def _load_topic(db: AsyncSession, config: Dict[str, Any]):
        """Load a single topic and its sources."""
        # Check if topic already exists
        stmt = select(Topic).where(Topic.slug == config["slug"])
        result = await db.execute(stmt)
        existing_topic = result.scalar_one_or_none()

        if existing_topic:
            logger.info(f"Topic {config['name']} already exists, skipping")
            return

        # Create new topic
        topic = Topic(
            name=config["name"],
            slug=config["slug"],
            description=config.get("description"),
            keywords=config.get("keywords", []),
            is_active=config.get("is_active", True),
            schedule_interval_minutes=config.get("schedule_interval_minutes", 60),
        )
        db.add(topic)
        await db.flush()  # Get the topic ID

        # Add sources
        for source_config in config.get("sources", []):
            source = NewsSource(
                topic_id=topic.id,
                name=source_config["name"],
                url=source_config["url"],
                source_type=SourceType(source_config.get("source_type", "rss")),
                config=source_config.get("config", {}),
                is_active=source_config.get("is_active", True),
            )
            db.add(source)

        logger.info(
            f"Created topic: {topic.name} with {len(config.get('sources', []))} sources"
        )


async def load_initial_topics():
    """Load initial topics from configuration file if it exists."""
    config_path = Path("config/topics.yaml")
    if config_path.exists():
        count = await TopicLoader.load_from_yaml(config_path)
        logger.info(f"Initial topic loading complete: {count} topics loaded")
    else:
        logger.info("No topics.yaml found, skipping initial topic loading")
