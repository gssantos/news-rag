# Simple News RAG Service
A minimal service to ingest, summarize, and search news articles using FastAPI, LangChain, and PostgreSQL with pgvector for semantic search capabilities. Includes automatic scheduled ingestion and a comprehensive evaluation framework.

## Features
- **Article Ingestion**: Extract and process news articles from URLs using LangChain loaders.
- **AI Summarization**: Generate concise summaries using OpenAI GPT models.
- **Semantic Search**: Vector-based similarity search using pgvector and cosine distance.
- **Automatic Scheduled Ingestion**: Continuously ingest articles for configured topics using APScheduler.
- **Topic Management**: Easy configuration and management of news topics and sources (RSS support included).
- **Deduplication**: Prevent duplicate articles using URL-based deduplication.
- **Evaluation Framework**:
  - **Golden Datasets**: Manage ground truth datasets for evaluation.
  - **Multi-type Evaluation**: Assess retrieval (Precision/Recall/F1), generation (ROUGE), and end-to-end RAG performance.
  - **Ragas Integration**: Calculate advanced metrics like Faithfulness and Answer Relevancy.
  - **MLFlow Integration**: Track experiments, metrics, and model performance.
- **RESTful API**: Clean, documented API endpoints with OpenAPI/Swagger integration.
- **Error Handling & Monitoring**: Robust error handling, structured logging, and health checks.
- **Docker Support**: Containerized deployment with Docker Compose.
- **Database Migrations**: Automated schema management with Alembic.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /healthz | Health check endpoint |
| GET | /docs | Interactive API documentation |
| **Core RAG** |  |  |
| POST | /api/v1/ingest/url | Ingest article from URL |
| GET | /api/v1/search | Search articles with semantic similarity |
| GET | /api/v1/content/{id} | Get full article content by ID |
| **Topics & Ingestion** |  |  |
| POST | /api/v1/topics | Create a new topic |
| GET | /api/v1/topics | List all topics |
| GET | /api/v1/topics/{id} | Get topic details |
| PATCH | /api/v1/topics/{id} | Update a topic |
| DELETE | /api/v1/topics/{id} | Delete a topic |
| POST | /api/v1/topics/{id}/sources | Add source to topic |
| GET | /api/v1/topics/{id}/sources | List topic sources |
| DELETE | /api/v1/topics/{id}/sources/{source_id} | Delete source |
| POST | /api/v1/topics/{id}/ingest | Trigger manual ingestion |
| GET | /api/v1/topics/stats/summary | Get ingestion statistics |
| **Evaluation Framework** |  |  |
| POST | /api/v1/evaluation/golden-datasets | Create a golden dataset |
| POST | /api/v1/evaluation/run | Run an evaluation against a dataset |
| GET | /api/v1/evaluation/history | Get evaluation run history |
| POST | /api/v1/evaluation/compare | Compare metrics across multiple runs |

## Quick Start with Docker Compose

1. **Clone and configure environment**:
   ```bash
   git clone <repository-url>
   cd news-rag
   # Create a .env file for configuration
   touch .env
   ```

2. **Set required environment variables in .env**:
   
   Note: The DATABASE_URL below matches the credentials and service name defined in docker-compose.yml.
   
   ```bash
   # Connection string for the 'db' service in Docker Compose
   DATABASE_URL=postgresql+asyncpg://user:password@db:5432/newsragdb
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: If using MLFlow tracking server
   # MLFLOW_TRACKING_URI=http://localhost:5000
   ```

3. **Start the services**:
   ```bash
   docker compose up --build
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/healthz

## Development Setup

### Prerequisites
- Python 3.10+
- PostgreSQL (16+) with pgvector extension
- OpenAI API key

### Installation

This project uses Poetry for dependency management.

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**: Create a .env file in the root directory.
   
   ```bash
   # Example for local development connection
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/newsragdb
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run migrations**:
   ```bash
   poetry run alembic upgrade head
   ```

5. **Run the development server**:
   ```bash
   poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
   ```

## Usage Examples
*(Examples assume the service is running on http://localhost:8080)*

### Ingest an article
```bash
curl -X POST "http://localhost:8080/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bbc.co.uk/news/articles/cy85905dj2wo"}'
```

### Search articles
```bash
curl -X GET "http://localhost:8080/api/v1/search?query=university%20merger&k=5"
```

### Create a Topic for Automatic Ingestion
```bash
curl -X POST "http://localhost:8080/api/v1/topics" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Technology News",
    "slug": "tech-news",
    "schedule_interval_minutes": 120,
    "is_active": true
  }'
```

## Configuration
Configuration is managed via environment variables, loaded using Pydantic Settings.

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL connection string (must use asyncpg driver) | postgresql+asyncpg://user:pass@host:5432/db |
| OPENAI_API_KEY | OpenAI API key for LLM and embeddings | sk-... |

### Optional
| Variable | Default | Description |
|----------|---------|-------------|
| **General** |  |  |
| LLM_MODEL | gpt-5-mini | OpenAI model for summarization |
| EMB_MODEL | text-embedding-3-small | OpenAI model for embeddings |
| EMB_DIM | 1536 | Embedding dimension (must match model) |
| LOG_LEVEL | INFO | Logging level |
| API_KEY | None | Optional API key for endpoint protection (X-API-Key header) |
| ALLOWED_DOMAINS | None | Comma-separated list of allowed domains for ingestion |
| **Ingestion & Scheduling** |  |  |
| HTTP_FETCH_TIMEOUT_SECONDS | 15 | URL fetch timeout |
| MAX_CONTENT_CHARS | 200000 | Maximum content length to process |
| ENABLE_SCHEDULER | True | Enable automatic scheduled ingestion |
| DEFAULT_SCHEDULE_INTERVAL_MINUTES | 60 | Default ingestion interval |
| RSS_DATE_THRESHOLD_HOURS | 24 | How far back to look in RSS feeds |
| **Evaluation** |  |  |
| ENABLE_EVALUATION | True | Enable the evaluation framework |
| MLFLOW_TRACKING_URI | http://localhost:5000 | URI for the MLFlow tracking server |
| EVALUATION_BATCH_SIZE | 10 | Batch size for evaluation processing |
| AUTO_EVALUATE_ON_UPDATE | False | Automatically run evaluation when models change |

## Automatic Ingestion Setup
The service supports automatic, scheduled ingestion of articles for configured topics.

### Topic Configuration
Topics can be configured via the API or using a YAML configuration file.

#### 1. Using the API
See the "Create a Topic" usage example above. After creating a topic, add sources:

```bash
# Add an RSS source to the topic (replace {topic_id} with the actual ID)
curl -X POST "http://localhost:8080/api/v1/topics/{topic_id}/sources" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "OilPrice.com RSS",
    "url": "https://oilprice.com/rss/main",
    "source_type": "rss",
    "is_active": true
  }'
```

#### 2. Using YAML Configuration
Create a `config/topics.yaml` file. Topics defined here are loaded automatically on startup by `app/utils/topic_loader.py`.

```yaml
# config/topics.yaml
topics:
  - name: "Crude Oil Markets"
    slug: "crude-oil-markets"
    schedule_interval_minutes: 60
    is_active: true
    sources:
      - name: "OilPrice.com RSS"
        url: "https://oilprice.com/rss/main"
        source_type: "rss"
```

### Monitoring Ingestion
Monitor ingestion status and history via the API:

```bash
# Get overall statistics
curl "http://localhost:8080/api/v1/topics/stats/summary"

# Get ingestion history for a topic
curl "http://localhost:8080/api/v1/topics/{topic_id}/runs"
```

## Evaluation Framework
The service includes a comprehensive framework for evaluating the performance of the RAG pipeline using golden datasets. It integrates with Ragas for metric calculation and MLFlow for experiment tracking.

### Concepts
- **Golden Dataset**: A curated set of queries with expected answers and/or expected retrieved documents (ground truth).
- **Evaluation Run**: An execution of the RAG pipeline against a Golden Dataset.

### Evaluation Types

1. **Retrieval Evaluation** (`retrieval`): Assesses the accuracy of the vector search component.
   - **Metrics**: Precision@k, Recall@k, F1 Score.

2. **Generation Evaluation** (`generation`): Assesses the quality of the generated summaries/answers compared to ground truth answers.
   - **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L.

3. **End-to-End Evaluation** (`end_to_end`): Assesses the entire pipeline from query to final answer.
   - **Metrics** (powered by Ragas): Context Precision, Context Recall, Faithfulness, Answer Relevancy.

### Workflow

#### 1. Setup MLFlow (Optional)
To track evaluations visually, run an MLFlow server.

```bash
poetry run mlflow server --host 0.0.0.0 --port 5000
```

Ensure `MLFLOW_TRACKING_URI` in the application's `.env` points to this server.

#### 2. Create a Golden Dataset
Define your evaluation criteria and ground truth data.

```bash
curl -X POST "http://localhost:8080/api/v1/evaluation/golden-datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Q4 2025 Technology Trends Evaluation",
    "version": "1.0.0",
    "queries": [
      {
        "query_text": "What are the recent advancements in AI?",
        "expected_answer": "Recent advancements include large multimodal models...",
        "expected_article_ids": ["uuid-of-relevant-article-1"]
      }
    ]
  }'
```

**Tip**: Use the `scripts/init_golden_dataset.py` script to bootstrap an initial dataset based on existing articles.

#### 3. Run an Evaluation
Trigger an evaluation run against the dataset.

```bash
curl -X POST "http://localhost:8080/api/v1/evaluation/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-of-the-dataset",
    "evaluation_type": "end_to_end"
  }'
```

#### 4. Analyze Results
Results are stored in the database and tracked in MLFlow.

**View History**:
```bash
curl "http://localhost:8080/api/v1/evaluation/history?dataset_id=uuid-of-the-dataset"
```

**Compare Runs**:
```bash
curl -X POST "http://localhost:8080/api/v1/evaluation/compare" \
  -H "Content-Type: application/json" \
  -d '["uuid-of-run-1", "uuid-of-run-2"]'
```

**MLFlow UI**: Access the MLFlow UI (default: http://localhost:5000) to view detailed metrics and trends.

### Demo Scripts
The project includes demo scripts to showcase functionality:

- `demo_ingestion.py`: Demonstrates the automatic ingestion workflow via API calls.
- `demo_evaluation.py`: A self-contained demonstration of the evaluation logic.
- `scripts/evaluation_demo.py`: An end-to-end demo running evaluations against the service.

## Security Features
- **Optional API Key Authentication**: Protect endpoints with the `API_KEY` environment variable (use `X-API-Key` header).
- **SSRF Protection**: Validates ingested URLs to prevent access to internal networks (localhost, private IPs, reserved ranges). Implemented in `app/core/security.py`.
- **Domain Restrictions**: Limit ingestion to specific domains using the `ALLOWED_DOMAINS` environment variable.
- **Input Validation**: Comprehensive request validation with Pydantic.
- **Request ID Tracking**: Structured logging with request correlation IDs for traceability.

## Architecture

```
                                 ┌────────────────┐
                                 │   OpenAI API   │
                                 │ (LLM + Embed)  │
                                 └────────────────┘
                                        ▲
                                        │
┌──────────────┐     ┌────────────┐     │     ┌───────────┐     ┌────────────┐
│   Scheduler  ├─────▶   FastAPI  ◀─────┴─────┤ LangChain ├─────▶ PostgreSQL │
│ (APScheduler)│     │ (Async API)│           │Integration│     │ + pgvector │
└──────────────┘     └────────────┘           └───────────┘     └────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Evaluation  │
                     │(Ragas/ROUGE)│
                     └─────────────┘
                            │
                            ▼
                     ┌────────────┐
                     │   MLFlow   │
                     │  Tracking  │
                     └────────────┘
```

### Key Components:
- **FastAPI**: Modern, async web framework.
- **LangChain**: Document loading (NewsURLLoader, WebBaseLoader) and LLM integration.
- **pgvector**: PostgreSQL extension for vector similarity search.
- **Alembic**: Database migration management.
- **APScheduler**: For scheduled ingestion tasks (`app/core/scheduler.py`).
- **Ragas/ROUGE**: For evaluation metrics.
- **MLFlow**: Experiment tracking and metric logging.

## Testing
The project uses pytest. Configuration is in `pytest.ini`.

Run the test suite (assuming tests are present in the `tests` directory):

```bash
poetry run pytest
```

## Docker Deployment
The `docker-compose.yml` file defines the application and database services.

```yaml
version: '3.8'
services:
  app:
    build: .
    # ... (see docker-compose.yml for full configuration)
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:password@db:5432/newsragdb
      # ...
  db:
    # Uses PostgreSQL 17 image as defined in docker-compose.yml
    image: pgvector/pgvector:pg17
    # ...
```

## Performance Notes
- **Vector Search**: Uses cosine distance. Embeddings are normalized (L2 norm) client-side before storage.
- **Indexing**: Utilizes ivfflat indexes on the `articles.embedding` column for efficient vector search.
- **Connection Pooling**: Async SQLAlchemy connection pool configured in `app/db/session.py`.

## Troubleshooting

### Logs
View application logs:

```bash
# Docker Compose
docker compose logs -f app
```

**Enable debug logging**:
Set `LOG_LEVEL=DEBUG` in the `.env` file.