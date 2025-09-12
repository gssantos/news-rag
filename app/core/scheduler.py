import logging
from datetime import datetime, timezone

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.topic import Topic
from app.services.scheduled_ingestion_service import ScheduledIngestionService

logger = logging.getLogger(__name__)


class IngestionScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            jobstores={"default": MemoryJobStore()},
            executors={"default": AsyncIOExecutor()},
            job_defaults={
                "coalesce": True,  # Coalesce missed jobs into one
                "max_instances": 1,  # Only one instance of a job at a time
                "misfire_grace_time": 3600,  # Grace time for missed jobs (1 hour)
            },
            timezone="UTC",
        )
        self.ingestion_service = None

    async def start(self):
        """Start the scheduler and load all topic schedules."""
        logger.info("Starting ingestion scheduler...")

        # Initialize ingestion service
        self.ingestion_service = ScheduledIngestionService()

        # Load and schedule all active topics
        await self.reload_topic_schedules()

        # Start the scheduler
        self.scheduler.start()
        logger.info("Ingestion scheduler started successfully")

    async def reload_topic_schedules(self):
        """Reload all topic schedules from database."""
        async with AsyncSessionLocal() as db:
            try:
                # Get all active topics
                stmt = select(Topic).where(Topic.is_active)
                result = await db.execute(stmt)
                topics = result.scalars().all()

                # Remove all existing jobs
                self.scheduler.remove_all_jobs()

                # Schedule each topic
                for topic in topics:
                    self.schedule_topic(topic)

                logger.info(f"Scheduled {len(topics)} active topics for ingestion")

            except Exception as e:
                logger.error(f"Failed to reload topic schedules: {e}")
            finally:
                await db.close()

    def schedule_topic(self, topic: Topic):
        """Schedule ingestion for a specific topic."""
        job_id = f"ingest_topic_{topic.id}"

        # Remove existing job if it exists
        existing_job = self.scheduler.get_job(job_id)
        if existing_job:
            self.scheduler.remove_job(job_id)

        # Create trigger based on schedule interval
        trigger = IntervalTrigger(
            minutes=topic.schedule_interval_minutes,
            start_date=datetime.now(timezone.utc),
        )

        # Add job to scheduler
        self.scheduler.add_job(
            func=self._run_topic_ingestion,
            trigger=trigger,
            args=[str(topic.id)],
            id=job_id,
            name=f"Ingest articles for topic: {topic.name}",
            replace_existing=True,
        )

        logger.info(
            f"Scheduled topic '{topic.name}' for ingestion every "
            f"{topic.schedule_interval_minutes} minutes"
        )

    async def _run_topic_ingestion(self, topic_id: str):
        """Wrapper function to run topic ingestion with proper database session."""
        async with AsyncSessionLocal() as db:
            try:
                async with self.ingestion_service:
                    await self.ingestion_service.ingest_topic(topic_id, db)
            except Exception as e:
                logger.error(f"Error in scheduled ingestion for topic {topic_id}: {e}")
            finally:
                await db.close()

    def update_topic_schedule(self, topic_id: str, interval_minutes: int):
        """Update the schedule for a specific topic."""
        job_id = f"ingest_topic_{topic_id}"
        job = self.scheduler.get_job(job_id)

        if job:
            # Reschedule with new interval
            trigger = IntervalTrigger(
                minutes=interval_minutes, start_date=datetime.now(timezone.utc)
            )
            job.reschedule(trigger=trigger)
            logger.info(
                f"Updated schedule for topic {topic_id} to {interval_minutes} minutes"
            )

    def pause_topic(self, topic_id: str):
        """Pause scheduling for a specific topic."""
        job_id = f"ingest_topic_{topic_id}"
        job = self.scheduler.get_job(job_id)
        if job:
            job.pause()
            logger.info(f"Paused scheduling for topic {topic_id}")

    def resume_topic(self, topic_id: str):
        """Resume scheduling for a specific topic."""
        job_id = f"ingest_topic_{topic_id}"
        job = self.scheduler.get_job(job_id)
        if job:
            job.resume()
            logger.info(f"Resumed scheduling for topic {topic_id}")

    def remove_topic(self, topic_id: str):
        """Remove scheduling for a specific topic."""
        job_id = f"ingest_topic_{topic_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed scheduling for topic {topic_id}")

    async def shutdown(self):
        """Shutdown the scheduler gracefully."""
        logger.info("Shutting down ingestion scheduler...")
        self.scheduler.shutdown(wait=True)
        if self.ingestion_service:
            await self.ingestion_service.__aexit__(None, None, None)
        logger.info("Ingestion scheduler shut down successfully")


# Global scheduler instance
scheduler = IngestionScheduler()
