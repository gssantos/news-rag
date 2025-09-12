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
- **Automatic Scheduled Ingestion**: Continuously ingest articles for configured topics
- **Topic Management**: Easy configuration and management of news topics and sources
- **RSS Feed Support**: Automatic discovery and ingestion from RSS feeds
- **Deduplication**: Prevent duplicate articles using URL-based deduplication
- **Ingestion Monitoring**: Track ingestion status and statistics

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Health check endpoint |
| `GET` | `/docs` | Interactive API documentation |
| `POST` | `/api/v1/ingest/url` | Ingest article from URL |
| `GET` | `/api/v1/search` | Search articles with semantic similarity |
| `GET` | `/api/v1/content/{id}` | Get full article content by ID |
| **Topics** | | |
| `POST` | `/api/v1/topics` | Create a new topic |
| `GET` | `/api/v1/topics` | List all topics |
| `GET` | `/api/v1/topics/{id}` | Get topic details |
| `PATCH` | `/api/v1/topics/{id}` | Update a topic |
| `DELETE` | `/api/v1/topics/{id}` | Delete a topic |
| `POST` | `/api/v1/topics/{id}/sources` | Add source to topic |
| `GET` | `/api/v1/topics/{id}/sources` | List topic sources |
| `DELETE` | `/api/v1/topics/{id}/sources/{source_id}` | Delete source |
| `POST` | `/api/v1/topics/{id}/ingest` | Trigger manual ingestion |
| `GET` | `/api/v1/topics/{id}/runs` | Get ingestion history |
| `GET` | `/api/v1/topics/stats/summary` | Get ingestion statistics |

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
| `ENABLE_SCHEDULER` | `True` | Enable automatic scheduled ingestion |
| `SCHEDULER_MAX_INSTANCES` | `1` | Maximum concurrent ingestion tasks |
| `DEFAULT_SCHEDULE_INTERVAL_MINUTES` | `60` | Default ingestion interval |

## Automatic Ingestion Setup

The service supports automatic, scheduled ingestion of articles for configured topics. This feature enables continuous monitoring of news sources for specific domains like energy markets, shipping, and commodities.

### Topic Configuration

Topics can be configured in two ways:

#### 1. Using the API

Create topics programmatically:

```bash
# Create a topic for crude oil markets
curl -X POST "http://localhost:8080/api/v1/topics" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Crude Oil Markets",
    "slug": "crude-oil-markets",
    "description": "News about crude oil prices and markets",
    "keywords": ["crude oil", "WTI", "Brent", "OPEC"],
    "schedule_interval_minutes": 60,
    "is_active": true
  }'

# Add an RSS source to the topic
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

Create a `config/topics.yaml` file:

```yaml
topics:
  - name: "Crude Oil Markets"
    slug: "crude-oil-markets"
    description: "News about crude oil markets"
    keywords:
      - "crude oil"
      - "WTI"
      - "Brent"
    schedule_interval_minutes: 60
    is_active: true
    sources:
      - name: "OilPrice.com RSS"
        url: "https://oilprice.com/rss/main"
        source_type: "rss"
        is_active: true
```

Topics defined in `topics.yaml` are loaded automatically on startup if the file exists.

### Scheduling Options

- **Interval-based**: Topics are scheduled to run at fixed intervals (5 minutes to 1 week)
- **Manual Triggering**: Use the API to trigger immediate ingestion for any topic
- **Active/Inactive**: Topics can be paused without deletion

### Monitoring Ingestion

#### View Ingestion Status

```bash
# Get overall statistics
curl "http://localhost:8080/api/v1/topics/stats/summary"

# Get ingestion history for a topic
curl "http://localhost:8080/api/v1/topics/{topic_id}/runs"

# View logs
docker compose logs -f app | grep "ingestion"
```

#### Understanding Ingestion Runs

Each ingestion run tracks:
- **articles_discovered**: Total URLs found from all sources
- **articles_ingested**: Successfully processed new articles
- **articles_duplicates**: Articles already in database (deduplication working)
- **articles_failed**: Articles that failed to process
- **status**: `success`, `partial`, or `failed`
- **error_messages**: Detailed error information

### Deduplication

The system prevents duplicate articles through:
1. **URL-based deduplication**: Articles with the same URL are not re-ingested
2. **Database constraints**: Unique index on article URLs
3. **Topic linking**: Existing articles are linked to new topics without re-processing

### Best Practices

1. **Start with Conservative Schedules**: Begin with longer intervals (2-4 hours) to avoid rate limiting
2. **Monitor Source Health**: Check `consecutive_failures` on sources to identify problematic feeds
3. **Use Keywords Wisely**: Keywords help filter relevant content during ingestion
4. **Regular Cleanup**: Periodically review and remove inactive topics/sources
5. **API Key Protection**: Use `API_KEY` environment variable in production

### Troubleshooting

**Issue: No articles being ingested**
- Check if the topic and sources are active
- Verify RSS feed URLs are accessible
- Review logs for specific error messages

**Issue: High duplicate count**
- This is normal and indicates deduplication is working
- RSS feeds often contain the same articles for days

**Issue: Ingestion taking too long**
- Reduce the number of sources per topic
- Increase `HTTP_FETCH_TIMEOUT_SECONDS` for slow sources
- Consider increasing `schedule_interval_minutes`

**Issue: Memory/CPU usage high**
- Reduce `SCHEDULER_MAX_INSTANCES` to limit concurrent tasks
- Stagger topic schedules to avoid simultaneous runs

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