"""Tests for the evaluation service."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import (
    EvaluationRun,
    EvaluationType,
    GoldenDataset,
    GoldenQuery,
)
from app.services.evaluation_service import EvaluationService


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = MagicMock()
    service.llm_model_name = "gpt-5-mini"
    service.emb_model_name = "text-embedding-3-small"
    service.emb_dim = 1536
    service.generate_summary = AsyncMock(return_value="Test summary")
    service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    service.summarization_chain = MagicMock()
    service.summarization_chain.ainvoke = AsyncMock(return_value="Generated answer")
    return service


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    return session


@pytest.fixture
def evaluation_service(mock_db_session, mock_llm_service):
    """Create evaluation service with mocked dependencies."""
    with patch("app.services.evaluation_service.mlflow"):
        service = EvaluationService(mock_db_session, mock_llm_service)
        return service


@pytest.mark.asyncio
async def test_create_golden_dataset(evaluation_service, mock_db_session):
    """Test creating a golden dataset."""
    # Prepare test data
    queries = [
        {
            "query_text": "What are the latest AI developments?",
            "expected_answer": "Recent AI developments include...",
            "expected_article_ids": [uuid.uuid4()],
            "tags": ["ai", "technology"],
        },
        {
            "query_text": "How is climate change affecting global freight?",
            "expected_answer": "Climate change impacts freight through...",
            "expected_article_ids": [uuid.uuid4(), uuid.uuid4()],
            "tags": ["climate", "logistics"],
        },
    ]

    # Create dataset
    dataset = await evaluation_service.create_golden_dataset(
        name="Test Dataset",
        description="Test dataset for evaluation",
        version="1.0.0",
        queries=queries,
    )

    # Verify dataset was created
    assert dataset.name == "Test Dataset"
    assert dataset.version == "1.0.0"
    assert dataset.is_active is True

    # Verify database operations
    assert mock_db_session.add.called
    assert mock_db_session.flush.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_evaluate_retrieval(evaluation_service, mock_db_session):
    """Test retrieval evaluation."""
    # Mock dataset and queries
    dataset = GoldenDataset(
        id=uuid.uuid4(),
        name="Test Dataset",
        version="1.0.0",
        is_active=True,
    )

    article_ids = [uuid.uuid4() for _ in range(3)]
    query = GoldenQuery(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        query_text="Test query",
        expected_article_ids=article_ids[:2],  # Expect first 2 articles
    )

    # Mock database responses
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = dataset
    mock_result.scalars.return_value.all.return_value = [query]
    mock_db_session.execute.return_value = mock_result

    # Mock search results (return 2 correct and 1 incorrect)
    mock_search_results = [
        MagicMock(id=article_ids[0]),  # Correct
        MagicMock(id=article_ids[1]),  # Correct
        MagicMock(id=article_ids[2]),  # Incorrect
    ]
    evaluation_service.search_service.search = AsyncMock(
        return_value=mock_search_results
    )

    # Run evaluation
    with patch("app.services.evaluation_service.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        run = await evaluation_service.run_evaluation(
            dataset_id=dataset.id,
            evaluation_type=EvaluationType.RETRIEVAL,
        )

    # Verify run was created
    assert run.evaluation_type == EvaluationType.RETRIEVAL
    assert run.status == "completed"
    assert run.metrics is not None

    # Check metrics calculation
    # Precision: 2/3 = 0.67, Recall: 2/2 = 1.0
    assert "retrieval_precision" in run.metrics
    assert "retrieval_recall" in run.metrics
    assert run.metrics["retrieval_recall"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_generation(evaluation_service, mock_db_session):
    """Test generation evaluation."""
    # Mock dataset and queries
    dataset = GoldenDataset(
        id=uuid.uuid4(),
        name="Test Dataset",
        version="1.0.0",
        is_active=True,
    )

    query = GoldenQuery(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        query_text="Test query",
        expected_answer="Expected answer for the test query",
        expected_article_ids=[uuid.uuid4()],
    )

    # Mock database responses
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = dataset
    mock_result.scalars.return_value.all.return_value = [query]
    mock_db_session.execute.return_value = mock_result

    # Mock article retrieval
    mock_article = MagicMock()
    mock_article.summary = "Article summary"
    mock_db_session.get.return_value = mock_article

    # Run evaluation
    with patch("app.services.evaluation_service.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        run = await evaluation_service.run_evaluation(
            dataset_id=dataset.id,
            evaluation_type=EvaluationType.GENERATION,
        )

    # Verify run was created
    assert run.evaluation_type == EvaluationType.GENERATION
    assert run.status == "completed"
    assert run.metrics is not None

    # Check ROUGE metrics exist
    assert "generation_rouge1" in run.metrics
    assert "generation_rouge2" in run.metrics
    assert "generation_rougeL" in run.metrics


@pytest.mark.asyncio
async def test_evaluate_end_to_end(evaluation_service, mock_db_session):
    """Test end-to-end RAG evaluation."""
    # Mock dataset and queries
    dataset = GoldenDataset(
        id=uuid.uuid4(),
        name="Test Dataset",
        version="1.0.0",
        is_active=True,
    )

    query = GoldenQuery(
        id=uuid.uuid4(),
        dataset_id=dataset.id,
        query_text="Test query",
        expected_answer="Expected answer",
        expected_article_ids=[uuid.uuid4()],
    )

    # Mock database responses
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = dataset
    mock_result.scalars.return_value.all.return_value = [query]
    mock_db_session.execute.return_value = mock_result

    # Mock search results
    mock_search_results = [
        MagicMock(id=uuid.uuid4()),
    ]
    evaluation_service.search_service.search = AsyncMock(
        return_value=mock_search_results
    )

    # Mock article retrieval
    mock_article = MagicMock()
    mock_article.id = mock_search_results[0].id
    mock_article.summary = "Article summary for context"
    mock_db_session.get.return_value = mock_article

    # Mock Ragas evaluation
    with patch("app.services.evaluation_service.evaluate") as mock_ragas_evaluate:
        mock_ragas_evaluate.return_value = {
            "context_precision": 0.85,
            "context_recall": 0.90,
            "faithfulness": 0.88,
            "answer_relevancy": 0.92,
        }

        with patch("app.services.evaluation_service.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock()

            run = await evaluation_service.run_evaluation(
                dataset_id=dataset.id,
                evaluation_type=EvaluationType.END_TO_END,
            )

    # Verify run was created
    assert run.evaluation_type == EvaluationType.END_TO_END
    assert run.status == "completed"
    assert run.metrics is not None

    # Check Ragas metrics exist
    assert "context_precision" in run.metrics
    assert "context_recall" in run.metrics
    assert "faithfulness" in run.metrics
    assert "answer_relevancy" in run.metrics
    assert run.metrics["context_precision"] == 0.85


@pytest.mark.asyncio
async def test_get_evaluation_history(evaluation_service, mock_db_session):
    """Test getting evaluation history."""
    # Mock evaluation runs
    runs = [
        EvaluationRun(
            id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            evaluation_type=EvaluationType.RETRIEVAL,
            llm_model="gpt-5-mini",
            embed_model="text-embedding-3-small",
            status="completed",
            started_at=datetime.now(timezone.utc),
        )
        for _ in range(3)
    ]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = runs
    mock_db_session.execute.return_value = mock_result

    # Get history
    history = await evaluation_service.get_evaluation_history(limit=10)

    # Verify results
    assert len(history) == 3
    assert all(isinstance(run, EvaluationRun) for run in history)


@pytest.mark.asyncio
async def test_compare_evaluations(evaluation_service, mock_db_session):
    """Test comparing multiple evaluation runs."""
    # Mock evaluation runs with metrics
    run1 = EvaluationRun(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        evaluation_type=EvaluationType.RETRIEVAL,
        llm_model="gpt-5-mini",
        embed_model="text-embedding-3-small",
        status="completed",
        metrics={"retrieval_precision": 0.8, "retrieval_recall": 0.9},
        started_at=datetime.now(timezone.utc),
    )

    run2 = EvaluationRun(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        evaluation_type=EvaluationType.RETRIEVAL,
        llm_model="gpt-5-mini",
        embed_model="text-embedding-3-small",
        status="completed",
        metrics={"retrieval_precision": 0.85, "retrieval_recall": 0.88},
        started_at=datetime.now(timezone.utc),
    )

    mock_db_session.get.side_effect = [run1, run2]

    # Compare runs
    comparison = await evaluation_service.compare_evaluations([run1.id, run2.id])

    # Verify comparison results
    assert len(comparison["runs"]) == 2
    assert "metric_trends" in comparison
    assert "retrieval_precision" in comparison["metric_trends"]
    assert "retrieval_recall" in comparison["metric_trends"]

    # Check metric statistics
    precision_trend = comparison["metric_trends"]["retrieval_precision"]
    assert precision_trend["mean"] == pytest.approx(0.825, rel=1e-3)
    assert precision_trend["min"] == 0.8
    assert precision_trend["max"] == 0.85
