# Simple News RAG Service

A minimal service to ingest, summarize, and search news articles using FastAPI, LangChain, and PostgreSQL with pgvector for semantic search capabilities.

## Features

- **Article Ingestion**: Extract and process news articles from URLs
- **AI Summarization**: Generate concise summaries using OpenAI GPT models
- **Semantic Search**: Vector-based similarity search using pgvector and cosine distance
- **RESTful API**: Clean, documented API endpoints with OpenAPI/Swagger integration
- **Error Handling**: Robust error handling with structured logging
- **Docker Support**: Containerized deployment with Docker Compose
- **Database Migrations**: Automated schema management with Alembic
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Health check endpoint |
| `GET` | `/docs` | Interactive API documentation |
| `POST` | `/api/v1/ingest/url` | Ingest article from URL |
| `GET` | `/api/v1/search` | Search articles with semantic similarity |
| `GET` | `/api/v1/content/{id}` | Get full article content by ID |

## Quick Start with Docker Compose

1. **Clone and configure environment**:
   ```bash
   git clone <repository-url>
   cd news-rag
   cp .env.example .env
   ```

2. **Set required environment variables in `.env`**:
   ```bash
   DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/newsrag
   OPENAI_API_KEY=your_openai_api_key_here
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

- Python 3.13
- PostgreSQL with pgvector extension
- Poetry for dependency management
- OpenAI API key

### Installation

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start PostgreSQL and run migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Run the development server**:
   ```bash
   poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
   ```

## Usage Examples

### Ingest an article

```bash
curl -X POST "http://localhost:8080/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bbc.co.uk/news/articles/cy85905dj2wo"}'
```

**Response:**
```json
{
  "id": "57f0fc26-1e6d-402d-b478-fbcba838e5fd",
  "url": "https://www.bbc.co.uk/news/articles/cy85905dj2wo",
  "title": "UK's first 'super-university' to be created as two merge from 2026",
  "summary": "The universities of Kent and Greenwich will merge...",
  "created_at": "2025-09-10T14:47:10.866606Z"
}
```

### Search articles

```bash
curl -X GET "http://localhost:8080/api/v1/search?query=university%20merger&k=5"
```

**Response:**
```json
{
  "hits": [
    {
      "id": "57f0fc26-1e6d-402d-b478-fbcba838e5fd",
      "title": "UK's first 'super-university' to be created as two merge from 2026",
      "summary": "The universities of Kent and Greenwich will merge...",
      "score": 0.48149728664739655
    }
  ]
}
```

### Get article content

```bash
curl -X GET "http://localhost:8080/api/v1/content/57f0fc26-1e6d-402d-b478-fbcba838e5fd"
```

**Response:**
```json
{
  "id": "57f0fc26-1e6d-402d-b478-fbcba838e5fd",
  "url": "https://www.bbc.co.uk/news/articles/cy85905dj2wo",
  "title": "UK's first 'super-university' to be created as two merge from 2026",
  "content": "Full article text...",
  "summary": "Article summary...",
  "llm_model": "gpt-5-mini",
  "embed_model": "text-embedding-3-small"
}
```

## Configuration

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:pass@host:5432/db` |
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | `sk-...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-5-mini` | OpenAI model for summarization |
| `EMB_MODEL` | `text-embedding-3-small` | OpenAI model for embeddings |
| `EMB_DIM` | `1536` | Embedding dimension (must match model) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HTTP_FETCH_TIMEOUT_SECONDS` | `15` | URL fetch timeout |
| `MAX_CONTENT_CHARS` | `200000` | Maximum content length |
| `API_KEY` | `None` | Optional API key for endpoint protection |
| `ALLOWED_DOMAINS` | `None` | Comma-separated list of allowed domains |

## Security Features

- **Optional API Key Authentication**: Protect endpoints with API key
- **Domain Restrictions**: Limit ingestion to specific domains
- **Input Validation**: Comprehensive request validation with Pydantic
- **Error Sanitization**: Structured error responses without sensitive data
- **Non-root Container**: Docker container runs as non-root user
- **Request ID Tracking**: Structured logging with request correlation IDs

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangChain      │    │  PostgreSQL     │
│   Web Server    │────│   Integration    │────│  + pgvector     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────────────┐               │
         │              │   OpenAI API   │               │
         └──────────────│   (LLM + Embed)│───────────────┘
                        └────────────────┘
```

**Key Components:**
- **FastAPI**: Modern, async web framework with automatic documentation
- **LangChain**: Document loading and LLM integration
- **pgvector**: PostgreSQL extension for vector similarity search
- **Alembic**: Database migration management
- **Pydantic**: Data validation and settings management

## Testing

Run the test suite:
```bash
poetry run pytest
```

Run with coverage:
```bash
poetry run pytest --cov=app --cov-report=html
```

## Development Tools

### Code Formatting and Linting

```bash
# Format code with Black
poetry run black app/

# Sort imports with isort
poetry run isort app/

# Lint with Ruff
poetry run ruff check app/

# Type checking with MyPy
poetry run mypy app/
```

### Adding New Dependencies

```bash
# Add production dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Docker Deployment

**Production deployment with Docker Compose:**

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/newsrag
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
    
  db:
    image: pgvector/pgvector:pg17
    environment:
      - POSTGRES_DB=newsrag
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
volumes:
  postgres_data:
```

## Performance Notes

- **Vector Search**: Uses cosine distance for semantic similarity
- **Connection Pooling**: Async SQLAlchemy connection pool
- **Batch Processing**: Consider batching for large ingestion volumes
- **Embedding Cache**: Embeddings are stored and reused
- **Memory Usage**: Monitor memory usage with large articles
- **Database Indexing**: pgvector indexes optimize search performance

## Troubleshooting

### Common Issues

**1. Import Error: lxml.html.clean**
```bash
# Solution: Install lxml with html_clean extra
poetry add "lxml[html_clean]"
```

**2. Async Event Loop Conflicts**
```bash
# Check logs for "asyncio.run() cannot be called from a running event loop"
# Solution: Use ThreadPoolExecutor for sync operations
```

**3. Database Connection Failed**
```bash
# Verify DATABASE_URL format and pgvector extension
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**4. OpenAI API Errors**
```bash
# Check API key and rate limits
export OPENAI_API_KEY="your-key-here"
```

### Logs

**View application logs:**
```bash
# Docker Compose
docker compose logs -f app

# Direct Python
poetry run uvicorn app.main:app --log-level debug
```

**Common log patterns:**
- `Request started/finished`: HTTP request lifecycle
- `Starting ingestion`: Article processing begins
- `Successfully extracted content`: Content fetched successfully
- `Database session error`: Database operation failed
- `LLM service error`: OpenAI API issues

**Enable debug logging:**
```bash
export LOG_LEVEL=DEBUG
```