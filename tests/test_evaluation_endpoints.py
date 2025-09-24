"""Tests for evaluation API endpoints."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from app.models.evaluation import EvaluationType


@pytest.mark.asyncio
async def test_create_golden_dataset(api_client, fake_async_session):
    """Test creating a golden dataset via API."""
    # Prepare request data
    dataset_data = {
        "name": "Test Dataset",
        "description": "Test dataset for API testing",
        "version": "1.0.0",
        "queries": [
            {
                "query_text": "Test query 1",
                "expected_answer": "Expected answer 1",
                "expected_article_ids": [str(uuid.uuid4())],
                "tags": ["test"],
            },
            {
                "query_text": "Test query 2",
                "expected_answer": "Expected answer 2",
                "expected_article_ids": [str(uuid.uuid4())],
                "tags": ["test"],
            },
        ],
    }

    # Mock the evaluation service
    with patch("app.api.evaluation_endpoints.EvaluationService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock the create_golden_dataset method
        mock_dataset = MagicMock()
        mock_dataset.id = uuid.uuid4()
        mock_dataset.name = dataset_data["name"]
        mock_dataset.description = dataset_data["description"]
        mock_dataset.version = dataset_data["version"]
        mock_dataset.is_active = True
        mock_service.create_golden_dataset = AsyncMock(return_value=mock_dataset)

        # Make request
        response = api_client.post(
            "/api/v1/evaluation/golden-datasets", json=dataset_data
        )

        # Verify response
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        assert response_data["name"] == dataset_data["name"]
        assert response_data["version"] == dataset_data["version"]
        assert response_data["is_active"] is True


@pytest.mark.asyncio
async def test_run_evaluation(api_client, fake_async_session):
    """Test running an evaluation via API."""
    # Prepare request data
    dataset_id = str(uuid.uuid4())
    request_data = {
        "dataset_id": dataset_id,
        "evaluation_type": "retrieval",
        "config": {"test": True},
    }

    # Mock the evaluation service
    with patch("app.api.evaluation_endpoints.EvaluationService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock the run_evaluation method
        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.dataset_id = uuid.UUID(dataset_id)
        mock_run.evaluation_type = EvaluationType.RETRIEVAL
        mock_run.llm_model = "gpt-5-mini"
        mock_run.embed_model = "text-embedding-3-small"
        mock_run.status = "running"
        mock_run.mlflow_run_id = "mlflow-123"
        mock_run.mlflow_experiment_id = "exp-456"
        mock_run.metrics = None
        mock_run.config = request_data["config"]
        mock_run.error_message = None
        mock_run.started_at = MagicMock()
        mock_run.completed_at = None
        mock_service.run_evaluation = AsyncMock(return_value=mock_run)

        # Make request
        response = api_client.post("/api/v1/evaluation/run", json=request_data)

        # Verify response
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["dataset_id"] == dataset_id
        assert response_data["evaluation_type"] == "retrieval"
        assert response_data["status"] == "running"


@pytest.mark.asyncio
async def test_run_evaluation_dataset_not_found(api_client, fake_async_session):
    """Test running evaluation with non-existent dataset."""
    # Prepare request data
    request_data = {
        "dataset_id": str(uuid.uuid4()),
        "evaluation_type": "retrieval",
    }

    # Mock the evaluation service to raise ValueError
    with patch("app.api.evaluation_endpoints.EvaluationService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.run_evaluation = AsyncMock(
            side_effect=ValueError("Dataset not found")
        )

        # Make request
        response = api_client.post("/api/v1/evaluation/run", json=request_data)

        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_json = response.json()
        # Handle different possible response formats
        if "detail" in response_json:
            assert "Dataset not found" in response_json["detail"]
        else:
            # May be in a different format
            assert "Dataset not found" in str(response_json)


@pytest.mark.asyncio
async def test_get_evaluation_history(api_client, fake_async_session):
    """Test getting evaluation history via API."""
    # Mock the evaluation service
    with patch("app.api.evaluation_endpoints.EvaluationService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock evaluation runs
        mock_runs = []
        for i in range(3):
            mock_run = MagicMock()
            mock_run.id = uuid.uuid4()
            mock_run.dataset_id = uuid.uuid4()
            mock_run.evaluation_type = EvaluationType.RETRIEVAL
            mock_run.llm_model = "gpt-5-mini"
            mock_run.embed_model = "text-embedding-3-small"
            mock_run.status = "completed"
            mock_run.mlflow_run_id = f"mlflow-{i}"
            mock_run.mlflow_experiment_id = f"exp-{i}"
            mock_run.metrics = {"retrieval_f1": 0.85 + i * 0.01}
            mock_run.config = {}
            mock_run.error_message = None
            mock_run.started_at = MagicMock()
            mock_run.completed_at = MagicMock()
            mock_runs.append(mock_run)

        mock_service.get_evaluation_history = AsyncMock(return_value=mock_runs)

        # Make request
        response = api_client.get("/api/v1/evaluation/history?limit=10")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data["runs"]) == 3
        assert all(run["status"] == "completed" for run in response_data["runs"])


@pytest.mark.asyncio
async def test_compare_evaluations(api_client, fake_async_session):
    """Test comparing evaluations via API."""
    # Prepare request data
    run_ids = [str(uuid.uuid4()) for _ in range(3)]

    # Mock the evaluation service
    with patch("app.api.evaluation_endpoints.EvaluationService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock comparison result
        mock_comparison = {
            "runs": [
                {
                    "id": run_id,
                    "dataset_id": str(uuid.uuid4()),
                    "evaluation_type": "retrieval",
                    "llm_model": "gpt-5-mini",
                    "embed_model": "text-embedding-3-small",
                    "status": "completed",
                    "started_at": "2025-09-12T12:00:00",
                    "metrics": {"retrieval_f1": 0.85 + i * 0.01},
                }
                for i, run_id in enumerate(run_ids)
            ],
            "metric_trends": {
                "retrieval_f1": {
                    "values": [0.85, 0.86, 0.87],
                    "mean": 0.86,
                    "std": 0.01,
                    "min": 0.85,
                    "max": 0.87,
                }
            },
        }
        mock_service.compare_evaluations = AsyncMock(return_value=mock_comparison)

        # Make request
        response = api_client.post("/api/v1/evaluation/compare", json=run_ids)

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data["runs"]) == 3
        assert "metric_trends" in response_data
        assert "retrieval_f1" in response_data["metric_trends"]
        assert response_data["metric_trends"]["retrieval_f1"]["mean"] == 0.86


@pytest.mark.asyncio
async def test_compare_evaluations_insufficient_runs(api_client, fake_async_session):
    """Test comparing evaluations with insufficient run IDs."""
    # Prepare request data with only one run ID
    run_ids = [str(uuid.uuid4())]

    # Make request
    response = api_client.post("/api/v1/evaluation/compare", json=run_ids)

    # Verify response
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    response_json = response.json()
    # Handle different possible response formats
    if "detail" in response_json:
        assert "At least 2 run IDs are required" in response_json["detail"]
    else:
        # May be in a different format
        assert "At least 2 run IDs are required" in str(response_json)
