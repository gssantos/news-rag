# News RAG Evaluation Framework Documentation

## Overview

The News RAG Evaluation Framework provides comprehensive assessment of the RAG (Retrieval-Augmented Generation) system's performance, tracking metrics for both retrieval accuracy and generation quality. The framework integrates with MLFlow for experiment tracking and uses Ragas for specialized RAG evaluation metrics.

## Architecture

### Components

1. **Evaluation Service** (`app/services/evaluation_service.py`)
   - Core evaluation logic
   - Ragas integration
   - MLFlow tracking
   - Metric calculation

2. **Database Models** (`app/models/evaluation.py`)
   - Golden datasets
   - Evaluation runs
   - Detailed results

3. **API Endpoints** (`app/api/evaluation_endpoints.py`)
   - RESTful API for evaluation operations
   - Dataset management
   - Run comparison

4. **Automation** (`app/utils/evaluation_automation.py`)
   - Automatic evaluation on model updates
   - Scheduled evaluations

## Key Features

### 1. Golden Dataset Management

Golden datasets contain ground truth queries with expected results:

```python
{
    "name": "Q4 2024 Evaluation Dataset",
    "version": "1.0.0",
    "queries": [
        {
            "query_text": "What are recent Red Sea freight developments?",
            "expected_answer": "Recent disruptions due to...",
            "expected_article_ids": ["uuid1", "uuid2"],
            "tags": ["freight", "red-sea"]
        }
    ]
}
```

### 2. Evaluation Types

#### Retrieval Evaluation
Measures how well the system retrieves relevant documents:
- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that are retrieved
- **F1 Score**: Harmonic mean of precision and recall

#### Generation Evaluation
Assesses the quality of generated summaries/answers:
- **ROUGE-1**: Unigram overlap with reference text
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

#### End-to-End RAG Evaluation
Comprehensive evaluation using Ragas metrics:
- **Context Precision**: Relevance ranking of retrieved contexts
- **Context Recall**: Coverage of ground truth by retrieved contexts
- **Faithfulness**: Factual consistency between generated answer and context
- **Answer Relevancy**: How well the answer addresses the query

## MLFlow Integration

### Tracking Server Setup

The docker-compose configuration includes:
- MLFlow tracking server on port 5000
- PostgreSQL backend for metadata
- MinIO for artifact storage

Access the MLFlow UI at: `http://localhost:5000`

### Logged Metrics

Each evaluation run logs:
- Configuration parameters
- Model versions
- All calculated metrics
- Execution times
- Error details (if any)

## API Usage

### Create Golden Dataset

```bash
curl -X POST http://localhost:8080/api/v1/evaluation/golden-datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Dataset",
    "description": "Dataset for testing",
    "version": "1.0.0",
    "queries": [...]
  }'
```

### Run Evaluation

```bash
curl -X POST http://localhost:8080/api/v1/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset-uuid",
    "evaluation_type": "end_to_end",
    "config": {}
  }'
```

### Get Evaluation History

```bash
curl http://localhost:8080/api/v1/evaluation/history?limit=10
```

### Compare Evaluations

```bash
curl -X POST http://localhost:8080/api/v1/evaluation/compare \
  -H "Content-Type: application/json" \
  -d '["run-id-1", "run-id-2", "run-id-3"]'
```

## Automation

### Automatic Evaluation on Model Updates

When `AUTO_EVALUATE_ON_UPDATE` is enabled in configuration:

1. System monitors for model changes
2. Automatically triggers evaluation
3. Logs results to MLFlow
4. Alerts on performance degradation

### Scheduled Evaluations

Run periodic evaluations using cron or scheduler:

```bash
python app/utils/evaluation_automation.py
```

## Running the Demo

### Prerequisites

1. Start all services:
```bash
docker-compose up -d
```

2. Ingest some articles:
```bash
curl -X POST http://localhost:8080/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

3. Initialize golden dataset:
```bash
python scripts/init_golden_dataset.py
```

### Run Full Demo

```bash
python scripts/evaluation_demo.py
```

The demo will:
1. Load the golden dataset
2. Run retrieval evaluation
3. Run generation evaluation
4. Run end-to-end RAG evaluation
5. Compare results
6. Display MLFlow tracking info
7. Show evaluation history

## Extending the Framework

### Adding New Metrics

1. Implement metric calculation in `EvaluationService`
2. Update the evaluation type handler
3. Log metrics to MLFlow
4. Update schemas for API response

Example:
```python
async def _calculate_custom_metric(self, query, result):
    # Your metric logic here
    return metric_value
```

### Custom Evaluation Types

1. Add new enum value to `EvaluationType`
2. Implement evaluation logic in `EvaluationService`
3. Update API endpoints
4. Add database migrations if needed

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run Evaluation
  run: |
    python scripts/init_golden_dataset.py
    curl -X POST $API_URL/evaluation/run \
      -d '{"dataset_id": "$DATASET_ID", "evaluation_type": "end_to_end"}'
```

## Interpreting Results

### Retrieval Metrics

- **Good**: F1 > 0.8
- **Acceptable**: F1 between 0.6-0.8
- **Needs Improvement**: F1 < 0.6

### Generation Metrics

- **Good**: ROUGE-L > 0.5
- **Acceptable**: ROUGE-L between 0.3-0.5
- **Needs Improvement**: ROUGE-L < 0.3

### RAG Metrics (Ragas)

- **Context Precision**: Higher is better (0-1 scale)
- **Context Recall**: Higher is better (0-1 scale)
- **Faithfulness**: Higher is better, < 0.7 indicates hallucination risk
- **Answer Relevancy**: Higher is better, < 0.7 indicates off-topic responses

## Troubleshooting

### MLFlow Connection Issues

Check MLFlow is running:
```bash
docker-compose ps mlflow
curl http://localhost:5000/health
```

### Ragas Evaluation Failures

Common causes:
- Insufficient OpenAI API quota
- Malformed golden dataset
- Network connectivity issues

Check logs:
```bash
docker-compose logs app | grep "evaluation"
```

### Performance Issues

For large datasets:
1. Increase `EVALUATION_BATCH_SIZE`
2. Run evaluations asynchronously
3. Use sampling for quick checks

## Best Practices

1. **Version Golden Datasets**: Track changes over time
2. **Regular Evaluation**: Run daily/weekly evaluations
3. **Monitor Trends**: Watch for performance degradation
4. **Diverse Queries**: Include various query types
5. **Realistic Expectations**: Set appropriate thresholds
6. **Document Changes**: Log why metrics changed

## Configuration

Key environment variables:

```env
# Evaluation Settings
ENABLE_EVALUATION=true
MLFLOW_TRACKING_URI=http://localhost:5000
EVALUATION_BATCH_SIZE=10
AUTO_EVALUATE_ON_UPDATE=true

# Model Settings (tracked in evaluations)
LLM_MODEL=gpt-5-mini
EMB_MODEL=text-embedding-3-small
```

## Future Enhancements

Planned improvements:
- A/B testing framework
- Real-time evaluation dashboard
- Automated alerting on metric degradation
- Integration with model versioning
- Custom visualization tools
- Export evaluation reports