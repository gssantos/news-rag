"""Automation utilities for running evaluations after model updates."""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.evaluation import EvaluationType, GoldenDataset
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class EvaluationAutomation:
    """Handles automatic evaluation runs after model updates."""

    def __init__(self):
        self.engine = create_async_engine(str(settings.DATABASE_URL))
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def run_automatic_evaluation(
        self,
        trigger: str,
        dataset_id: Optional[UUID] = None,
        evaluation_types: Optional[list[EvaluationType]] = None,
    ):
        """Run automatic evaluation based on a trigger event."""
        logger.info(f"Starting automatic evaluation triggered by: {trigger}")

        async with self.SessionLocal() as db:
            llm_service = get_llm_service()
            eval_service = EvaluationService(db, llm_service)

            # Get active datasets if not specified
            if dataset_id:
                dataset_ids = [dataset_id]
            else:
                result = await db.execute(
                    select(GoldenDataset).where(GoldenDataset.is_active)
                )
                datasets = result.scalars().all()
                dataset_ids = [d.id for d in datasets]

            if not dataset_ids:
                logger.warning("No active datasets found for evaluation")
                return

            # Default to all evaluation types if not specified
            if not evaluation_types:
                evaluation_types = list(EvaluationType)

            results = []
            for dataset_id in dataset_ids:
                for eval_type in evaluation_types:
                    try:
                        logger.info(
                            f"Running {eval_type.value} evaluation on dataset {dataset_id}"
                        )

                        run = await eval_service.run_evaluation(
                            dataset_id=dataset_id,
                            evaluation_type=eval_type,
                            config={
                                "trigger": trigger,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )

                        results.append(
                            {
                                "dataset_id": dataset_id,
                                "evaluation_type": eval_type.value,
                                "status": run.status,
                                "metrics": run.metrics,
                            }
                        )

                        logger.info(f"Evaluation completed: {run.metrics}")

                    except Exception as e:
                        logger.error(
                            f"Evaluation failed for dataset {dataset_id}, type {eval_type}: {e}"
                        )
                        results.append(
                            {
                                "dataset_id": dataset_id,
                                "evaluation_type": eval_type.value,
                                "status": "failed",
                                "error": str(e),
                            }
                        )

            return results

    async def monitor_model_changes(self):
        """Monitor for model changes and trigger evaluations."""
        logger.info("Starting model change monitor")

        last_llm_model = settings.LLM_MODEL
        last_emb_model = settings.EMB_MODEL

        while True:
            await asyncio.sleep(60)  # Check every minute

            # Check if models have changed
            if (
                settings.LLM_MODEL != last_llm_model
                or settings.EMB_MODEL != last_emb_model
            ):

                logger.info(
                    f"Model change detected: LLM {last_llm_model} -> {settings.LLM_MODEL}, "
                    f"Embeddings {last_emb_model} -> {settings.EMB_MODEL}"
                )

                if settings.AUTO_EVALUATE_ON_UPDATE:
                    await self.run_automatic_evaluation(
                        trigger=f"model_update_{datetime.utcnow().isoformat()}"
                    )

                last_llm_model = settings.LLM_MODEL
                last_emb_model = settings.EMB_MODEL


async def run_scheduled_evaluation():
    """Run a scheduled evaluation (can be called from cron or scheduler)."""
    automation = EvaluationAutomation()
    results = await automation.run_automatic_evaluation(
        trigger=f"scheduled_{datetime.utcnow().isoformat()}"
    )
    return results


if __name__ == "__main__":
    # Run evaluation when script is executed directly
    asyncio.run(run_scheduled_evaluation())
