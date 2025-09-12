import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.endpoints import router as v1_router
from app.core.config import request_id_var, settings, setup_logging
from app.db.session import engine

# Ensure logging is configured (it runs on import in config.py, but explicit call ensures setup)
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")
    logger.info(f"LLM: {settings.LLM_MODEL}, Embeddings: {settings.EMB_MODEL}")

    # Verify DB connection on startup
    try:
        async with engine.connect():
            logger.info("Database connection verified.")
    except Exception as e:
        logger.critical(f"Database connection failed on startup: {e}")
        # In a production environment, we might want to exit here if the DB is critical.

    yield
    # Shutdown
    logger.info("Application shutting down...")
    await engine.dispose()


app = FastAPI(
    title="Simple News RAG Service",
    version="0.1.0",
    description="A minimal service to ingest, summarize, and search news articles.",
    lifespan=lifespan,
)


# --- Middleware ---
@app.middleware("http")
async def logging_and_error_middleware(request: Request, call_next):
    # Requirement 10: Structured logging with request_id
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Set the context variable using contextvars
    token = request_id_var.set(request_id)

    logger.info(f"Request started: method={request.method} path={request.url.path}")

    try:
        response = await call_next(request)
    except Exception as exc:
        # Global exception handling for unhandled exceptions only
        # HTTPException and RequestValidationError are handled by dedicated exception handlers
        if not isinstance(exc, (HTTPException, RequestValidationError)):
            # Requirement 11: Standard JSON error shape for unexpected errors (500)
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                },
            )
        else:
            # Let the exception bubble up to be handled by the dedicated exception handlers
            raise
    finally:
        # Reset the context variable
        request_id_var.reset(token)

    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"Request finished: status={response.status_code} duration={process_time:.4f}s"
    )
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with standardized error format."""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error format."""
    logger.warning(f"HTTP exception: status={exc.status_code} detail='{exc.detail}'")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"http_{exc.status_code}",
            "message": exc.detail,
            "details": {},
        },
    )


# Include the API router
# Using a version prefix is good practice, although not strictly required by the prompt.
app.include_router(v1_router, prefix="/api/v1")


# Health check endpoint (Requirement 10)
@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "News RAG API. See /docs for details."}
