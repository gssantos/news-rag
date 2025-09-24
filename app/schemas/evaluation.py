"""Pydantic schemas for evaluation framework."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.evaluation import EvaluationType


class GoldenQueryCreate(BaseModel):
    """Schema for creating a golden query."""

    query_text: str = Field(..., description="The query text")
    expected_answer: Optional[str] = Field(
        None, description="Expected answer for generation evaluation"
    )
    expected_article_ids: Optional[List[UUID]] = Field(
        None, description="Expected article IDs for retrieval evaluation"
    )
    context: Optional[str] = Field(None, description="Additional context for the query")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing queries")


class GoldenDatasetCreate(BaseModel):
    """Schema for creating a golden dataset."""

    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    version: str = Field(..., description="Dataset version")
    queries: List[GoldenQueryCreate] = Field(
        ..., description="List of queries in the dataset"
    )


class GoldenDatasetResponse(BaseModel):
    """Response schema for golden dataset."""

    id: UUID
    name: str
    description: Optional[str]
    version: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvaluationRunRequest(BaseModel):
    """Request schema for running an evaluation."""

    dataset_id: UUID = Field(
        ..., description="ID of the golden dataset to evaluate against"
    )
    evaluation_type: EvaluationType = Field(
        ..., description="Type of evaluation to run"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Additional configuration for the evaluation"
    )


class EvaluationRunResponse(BaseModel):
    """Response schema for evaluation run."""

    id: UUID
    dataset_id: UUID
    evaluation_type: EvaluationType
    llm_model: str
    embed_model: str
    mlflow_run_id: Optional[str]
    mlflow_experiment_id: Optional[str]
    status: str
    metrics: Optional[Dict[str, float]]
    config: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class EvaluationHistoryResponse(BaseModel):
    """Response schema for evaluation history."""

    runs: List[EvaluationRunResponse]


class MetricTrend(BaseModel):
    """Schema for metric trend analysis."""

    values: List[float]
    mean: float
    std: float
    min: float
    max: float


class CompareEvaluationsResponse(BaseModel):
    """Response schema for comparing evaluations."""

    runs: List[Dict[str, Any]]
    metric_trends: Dict[str, MetricTrend]
