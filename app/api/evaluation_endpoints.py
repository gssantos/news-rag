"""API endpoints for evaluation framework."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_api_key
from app.db.session import get_db_session
from app.models.evaluation import EvaluationType
from app.schemas.evaluation import (
    CompareEvaluationsResponse,
    EvaluationHistoryResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    GoldenDatasetCreate,
    GoldenDatasetResponse,
)
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
    dependencies=[Depends(get_api_key)],
)


def get_evaluation_service(
    db: AsyncSession = Depends(get_db_session),
    llm_service: LLMService = Depends(get_llm_service),
) -> EvaluationService:
    """Get evaluation service instance."""
    return EvaluationService(db=db, llm_service=llm_service)


@router.post(
    "/golden-datasets",
    response_model=GoldenDatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_golden_dataset(
    dataset: GoldenDatasetCreate,
    service: EvaluationService = Depends(get_evaluation_service),
):
    """Create a new golden dataset for evaluation."""
    try:
        result = await service.create_golden_dataset(
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            queries=[query.model_dump() for query in dataset.queries],
        )
        return GoldenDatasetResponse.model_validate(result)
    except Exception as e:
        logger.error(f"Failed to create golden dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create golden dataset: {str(e)}",
        )


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_evaluation(
    request: EvaluationRunRequest,
    service: EvaluationService = Depends(get_evaluation_service),
):
    """Run an evaluation against a golden dataset."""
    try:
        run = await service.run_evaluation(
            dataset_id=request.dataset_id,
            evaluation_type=request.evaluation_type,
            config=request.config,
        )
        return EvaluationRunResponse.model_validate(run)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get(
    "/history",
    response_model=EvaluationHistoryResponse,
)
async def get_evaluation_history(
    dataset_id: Optional[UUID] = Query(None, description="Filter by dataset ID"),
    evaluation_type: Optional[EvaluationType] = Query(
        None, description="Filter by evaluation type"
    ),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    service: EvaluationService = Depends(get_evaluation_service),
):
    """Get evaluation run history."""
    try:
        runs = await service.get_evaluation_history(
            dataset_id=dataset_id,
            evaluation_type=evaluation_type,
            limit=limit,
        )
        return EvaluationHistoryResponse(
            runs=[EvaluationRunResponse.model_validate(run) for run in runs]
        )
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation history: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=CompareEvaluationsResponse,
)
async def compare_evaluations(
    run_ids: List[UUID],
    service: EvaluationService = Depends(get_evaluation_service),
):
    """Compare metrics across multiple evaluation runs."""
    if len(run_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 run IDs are required for comparison",
        )

    try:
        comparison = await service.compare_evaluations(run_ids)
        return CompareEvaluationsResponse(**comparison)
    except Exception as e:
        logger.error(f"Failed to compare evaluations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare evaluations: {str(e)}",
        )
