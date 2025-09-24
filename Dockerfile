# syntax=docker/dockerfile:1

# =================================================================================================
# BASE STAGE
# Define the base image (Python 3.13 as required) and common settings
# =================================================================================================
FROM python:3.13-slim AS base

# Set environment variables for Python in Docker
# PYTHONUNBUFFERED: ensures output is logged immediately (not buffered)
# PYTHONDONTWRITEBYTECODE: prevents creation of .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


# =================================================================================================
# BUILDER STAGE
# Install system build dependencies, Poetry, and Python packages
# =================================================================================================
FROM base AS builder

# Install system dependencies required for compiling lxml (used by newspaper3k) and asyncpg.
# build-essential (includes gcc) for C extensions.
# libxml2-dev and libxslt1-dev for lxml.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry to create the virtualenv inside the project directory.
# This makes it easy to locate and copy the virtual environment later.
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /build

# Copy Poetry configuration files and README
# Using the wildcard ensures the build proceeds even if poetry.lock is missing (e.g., first build).
COPY pyproject.toml poetry.lock* README.md ./

# Install dependencies using Poetry. This creates the .venv folder in the /build directory.
# --only main includes only the necessary runtime dependencies.
RUN poetry install --only main


# =================================================================================================
# RUNTIME STAGE
# Create the final, lean runtime image
# =================================================================================================
FROM base AS runtime

# Install runtime system dependencies (non-dev versions).
# These must be present for the compiled wheels (like lxml) to function.
# curl is included as a safety measure as some web scraping libraries may utilize it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libxml2 \
    libxslt1.1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Requirement 12)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the virtual environment (Python dependencies) from the builder stage
# We copy the generated .venv directory from /build into /opt/venv in the runtime image.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /build/.venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Copy the application code and migration files, ensuring correct ownership during the copy
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser alembic.ini ./

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8080

# Entrypoint: Run migrations and then start the application (Requirement 12)
# We use the exec form with "sh -c" to allow command chaining (&&).
# Workers are set to 2 as a reasonable default for an async application.
CMD ["sh", "-c", "python -m alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1 --loop asyncio"]