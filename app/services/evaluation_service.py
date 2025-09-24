"""Evaluation service for assessing RAG system performance using Ragas and MLFlow."""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import mlflow
import numpy as np
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from rouge_score import rouge_scorer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.article import Article
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    EvaluationType,
    GoldenDataset,
    GoldenQuery,
)
from app.services.llm_service import LLMService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating RAG system performance."""

    def __init__(self, db: AsyncSession, llm_service: LLMService):
        self.db = db
        self.llm_service = llm_service
        self.search_service = SearchService(db, llm_service)

        # Initialize MLFlow
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Initialize Ragas components
        self._init_ragas_components()

        # Initialize ROUGE scorer for summarization metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def _init_ragas_components(self):
        """Initialize Ragas evaluation components."""
        # Use the same models as the main system for consistency
        self.ragas_llm = ChatOpenAI(
            model=self.llm_service.llm_model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )

        self.ragas_embeddings = OpenAIEmbeddings(
            model=self.llm_service.emb_model_name,
            api_key=settings.OPENAI_API_KEY,
        )

    async def create_golden_dataset(
        self,
        name: str,
        description: str,
        version: str,
        queries: List[Dict[str, Any]],
    ) -> GoldenDataset:
        """Create a new golden dataset with queries."""
        logger.info(f"Creating golden dataset: {name} v{version}")

        # Create dataset
        dataset = GoldenDataset(
            name=name,
            description=description,
            version=version,
            is_active=True,
        )
        self.db.add(dataset)
        await self.db.flush()

        # Add queries
        for query_data in queries:
            query = GoldenQuery(
                dataset_id=dataset.id,
                query_text=query_data["query_text"],
                expected_answer=query_data.get("expected_answer"),
                expected_article_ids=query_data.get("expected_article_ids"),
                context=query_data.get("context"),
                metadata=query_data.get("metadata"),
                tags=query_data.get("tags"),
            )
            self.db.add(query)

        await self.db.commit()
        logger.info(f"Created golden dataset with {len(queries)} queries")
        return dataset

    async def run_evaluation(
        self,
        dataset_id: UUID,
        evaluation_type: EvaluationType,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationRun:
        """Run evaluation against a golden dataset."""
        logger.info(
            f"Starting evaluation for dataset {dataset_id}, type: {evaluation_type}"
        )

        # Get dataset and queries
        result = await self.db.execute(
            select(GoldenDataset).where(GoldenDataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Create evaluation run
        run = EvaluationRun(
            dataset_id=dataset_id,
            evaluation_type=evaluation_type,
            llm_model=self.llm_service.llm_model_name,
            embed_model=self.llm_service.emb_model_name,
            status="running",
            config=config or {},
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(run)
        await self.db.flush()

        try:
            # Start MLFlow run
            mlflow.set_experiment(f"news-rag-evaluation-{evaluation_type.value}")
            with mlflow.start_run() as mlflow_run:
                run.mlflow_run_id = mlflow_run.info.run_id
                run.mlflow_experiment_id = mlflow_run.info.experiment_id

                # Log configuration
                mlflow.log_params(
                    {
                        "dataset_name": dataset.name,
                        "dataset_version": dataset.version,
                        "evaluation_type": evaluation_type.value,
                        "llm_model": self.llm_service.llm_model_name,
                        "embed_model": self.llm_service.emb_model_name,
                    }
                )

                # Run evaluation based on type
                if evaluation_type == EvaluationType.RETRIEVAL:
                    metrics = await self._evaluate_retrieval(dataset, run)
                elif evaluation_type == EvaluationType.GENERATION:
                    metrics = await self._evaluate_generation(dataset, run)
                else:  # END_TO_END
                    metrics = await self._evaluate_end_to_end(dataset, run)

                # Log metrics to MLFlow
                mlflow.log_metrics(metrics)

                # Update run status
                run.metrics = metrics
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)

                logger.info(f"Evaluation completed. Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            raise

        finally:
            await self.db.commit()

        return run

    async def _evaluate_retrieval(
        self, dataset: GoldenDataset, run: EvaluationRun
    ) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        logger.info("Evaluating retrieval performance")

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        query_count = 0

        # Get all queries for this dataset
        result = await self.db.execute(
            select(GoldenQuery).where(GoldenQuery.dataset_id == dataset.id)
        )
        queries = result.scalars().all()

        for query in queries:
            if not query.expected_article_ids:
                continue

            start_time = time.time()

            # Perform search
            search_results = await self.search_service.search(
                query=query.query_text,
                k=min(10, len(query.expected_article_ids) * 2),
                start_date=None,
                end_date=None,
            )

            # Get retrieved article IDs
            retrieved_ids = [result.id for result in search_results]

            # Calculate metrics
            expected_set = set(query.expected_article_ids)
            retrieved_set = set(retrieved_ids[: len(query.expected_article_ids)])

            true_positives = len(expected_set & retrieved_set)
            false_positives = len(retrieved_set - expected_set)
            false_negatives = len(expected_set - retrieved_set)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Store result
            result = EvaluationResult(
                run_id=run.id,
                query_id=query.id,
                retrieved_article_ids=retrieved_ids,
                metrics={
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            self.db.add(result)

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            query_count += 1

        await self.db.flush()

        return {
            "retrieval_precision": (
                total_precision / query_count if query_count > 0 else 0
            ),
            "retrieval_recall": total_recall / query_count if query_count > 0 else 0,
            "retrieval_f1": total_f1 / query_count if query_count > 0 else 0,
            "num_queries": query_count,
        }

    async def _evaluate_generation(
        self, dataset: GoldenDataset, run: EvaluationRun
    ) -> Dict[str, float]:
        """Evaluate generation quality using ROUGE scores."""
        logger.info("Evaluating generation performance")

        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0
        query_count = 0

        # Get all queries with expected answers
        result = await self.db.execute(
            select(GoldenQuery).where(
                GoldenQuery.dataset_id == dataset.id,
                GoldenQuery.expected_answer.isnot(None),
            )
        )
        queries = result.scalars().all()

        for query in queries:
            start_time = time.time()

            # Get context from expected articles
            context = ""
            if query.expected_article_ids:
                for article_id in query.expected_article_ids[:5]:  # Limit context
                    article = await self.db.get(Article, article_id)
                    if article:
                        context += f"{article.summary}\n\n"

            # Generate answer
            generated_answer = await self.llm_service.summarization_chain.ainvoke(
                {
                    "title": "Query Response",
                    "content": context,
                    "source_domain": "evaluation",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(query.expected_answer, generated_answer)

            # Store result
            result = EvaluationResult(
                run_id=run.id,
                query_id=query.id,
                generated_answer=generated_answer,
                metrics={
                    "rouge1_fmeasure": scores["rouge1"].fmeasure,
                    "rouge2_fmeasure": scores["rouge2"].fmeasure,
                    "rougeL_fmeasure": scores["rougeL"].fmeasure,
                },
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            self.db.add(result)

            total_rouge1 += scores["rouge1"].fmeasure
            total_rouge2 += scores["rouge2"].fmeasure
            total_rougeL += scores["rougeL"].fmeasure
            query_count += 1

        await self.db.flush()

        return {
            "generation_rouge1": total_rouge1 / query_count if query_count > 0 else 0,
            "generation_rouge2": total_rouge2 / query_count if query_count > 0 else 0,
            "generation_rougeL": total_rougeL / query_count if query_count > 0 else 0,
            "num_queries": query_count,
        }

    async def _evaluate_end_to_end(
        self, dataset: GoldenDataset, run: EvaluationRun
    ) -> Dict[str, float]:
        """Evaluate end-to-end RAG performance using Ragas."""
        logger.info("Evaluating end-to-end RAG performance with Ragas")

        # Prepare data for Ragas evaluation
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        # Get all queries
        result = await self.db.execute(
            select(GoldenQuery).where(GoldenQuery.dataset_id == dataset.id)
        )
        queries = result.scalars().all()

        for query in queries:
            start_time = time.time()

            # Perform search
            search_results = await self.search_service.search(
                query=query.query_text,
                k=5,
                start_date=None,
                end_date=None,
            )

            # Get context from retrieved articles
            context_list = []
            retrieved_ids = []
            for search_result in search_results:
                article = await self.db.get(Article, search_result.id)
                if article:
                    context_list.append(article.summary)
                    retrieved_ids.append(article.id)

            # Generate answer based on retrieved context
            context_text = "\n\n".join(context_list)
            generated_answer = await self.llm_service.summarization_chain.ainvoke(
                {
                    "title": query.query_text,
                    "content": context_text,
                    "source_domain": "search results",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Store for Ragas evaluation
            questions.append(query.query_text)
            answers.append(generated_answer)
            contexts.append(context_list)
            ground_truths.append(
                [query.expected_answer] if query.expected_answer else [""]
            )

            # Store individual result
            eval_result = EvaluationResult(
                run_id=run.id,
                query_id=query.id,
                retrieved_article_ids=retrieved_ids,
                generated_answer=generated_answer,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
            self.db.add(eval_result)

        await self.db.flush()

        # Create dataset for Ragas
        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        # Run Ragas evaluation
        ragas_results: Any = evaluate(
            eval_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        # Convert Ragas results to metrics dict
        # Ragas returns arrays of scores, so we take the mean
        metrics: Dict[str, float] = {
            "context_precision": float(np.mean(ragas_results["context_precision"])),
            "context_recall": float(np.mean(ragas_results["context_recall"])),
            "faithfulness": float(np.mean(ragas_results["faithfulness"])),
            "answer_relevancy": float(np.mean(ragas_results["answer_relevancy"])),
            "num_queries": float(len(questions)),
        }

        return metrics

    async def get_evaluation_history(
        self,
        dataset_id: Optional[UUID] = None,
        evaluation_type: Optional[EvaluationType] = None,
        limit: int = 10,
    ) -> List[EvaluationRun]:
        """Get evaluation run history."""
        query = select(EvaluationRun)

        if dataset_id:
            query = query.where(EvaluationRun.dataset_id == dataset_id)
        if evaluation_type:
            query = query.where(EvaluationRun.evaluation_type == evaluation_type)

        query = query.order_by(EvaluationRun.started_at.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def compare_evaluations(self, run_ids: List[UUID]) -> Dict[str, Any]:
        """Compare metrics across multiple evaluation runs."""
        runs = []
        for run_id in run_ids:
            run = await self.db.get(EvaluationRun, run_id)
            if run:
                runs.append(run)

        if not runs:
            return {}

        comparison: Dict[str, Any] = {
            "runs": [],
            "metric_trends": {},
        }

        for run in runs:
            comparison["runs"].append(
                {
                    "id": str(run.id),
                    "dataset_id": str(run.dataset_id),
                    "evaluation_type": run.evaluation_type.value,
                    "llm_model": run.llm_model,
                    "embed_model": run.embed_model,
                    "status": run.status,
                    "started_at": run.started_at.isoformat(),
                    "metrics": run.metrics,
                }
            )

        # Calculate metric trends
        all_metrics: Dict[str, List[float]] = {}
        for run in runs:
            if run.metrics:
                for metric, value in run.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))

        for metric, values in all_metrics.items():
            comparison["metric_trends"][metric] = {
                "values": values,
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        return comparison
