import contextvars
import logging
import sys
from typing import List, Optional

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: PostgresDsn = Field(
        ...,
        description="PostgreSQL database URL (e.g., postgresql+asyncpg://user:pass@host:5432/dbname)",
    )

    # LLM Configuration
    # Requirements specify gpt-5-mini, using gpt-5-mini as the actual implementation default
    LLM_PROVIDER: str = Field(
        "openai", description="The provider for the Language Model"
    )
    LLM_MODEL: str = Field("gpt-5-mini", description="The specific LLM model name")

    # Embedding Configuration
    EMB_PROVIDER: str = Field(
        "openai", description="The provider for the Embedding Model"
    )
    EMB_MODEL: str = Field(
        "text-embedding-3-small", description="The specific embedding model name"
    )
    EMB_DIM: int = Field(
        1536, description="The dimension of the embeddings (must match EMB_MODEL)"
    )

    # OpenAI Specific
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API Key")

    # Application Settings
    LOG_LEVEL: str = Field("INFO", description="Logging level")

    # Ingestion Settings
    HTTP_FETCH_TIMEOUT_SECONDS: int = Field(
        15, description="Timeout for fetching article content"
    )
    MAX_CONTENT_CHARS: int = Field(
        200000, description="Maximum characters of content to process"
    )

    # Scheduler Settings
    ENABLE_SCHEDULER: bool = Field(
        True, description="Enable automatic scheduled ingestion"
    )
    SCHEDULER_MAX_INSTANCES: int = Field(
        1, description="Maximum concurrent ingestion tasks"
    )
    SCHEDULER_MISFIRE_GRACE_TIME: int = Field(
        3600, description="Grace time in seconds for missed jobs"
    )
    DEFAULT_SCHEDULE_INTERVAL_MINUTES: int = Field(
        60, description="Default ingestion interval in minutes"
    )
    RSS_DATE_THRESHOLD_HOURS: int = Field(
        24,
        description="RSS article age threshold in hours for ingestion filtering",
    )

    # Security
    API_KEY: Optional[str] = Field(
        None, description="Optional API key for securing endpoints"
    )
    ALLOWED_DOMAINS_STR: Optional[str] = Field(
        None,
        description="Optional comma-separated list of allowed domains",
        alias="ALLOWED_DOMAINS",
    )

    @computed_field
    def ALLOWED_DOMAINS(self) -> List[str]:
        if self.ALLOWED_DOMAINS_STR:
            return [domain.strip() for domain in self.ALLOWED_DOMAINS_STR.split(",")]
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def get_log_level(self):
        return getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)

    def validate_config(self):
        # Ensure DATABASE_URL uses the async driver
        db_url = str(self.DATABASE_URL)
        if not db_url.startswith("postgresql+asyncpg://"):
            raise ValueError(
                f"DATABASE_URL must use the 'postgresql+asyncpg' driver. Current URL: {db_url[:30]}..."
            )

        # Warn if API keys are missing for OpenAI providers (but don't fail)
        if (
            self.LLM_PROVIDER == "openai" or self.EMB_PROVIDER == "openai"
        ) and not self.OPENAI_API_KEY:
            logging.warning(
                "OPENAI_API_KEY is not set. LLM/embedding operations will fail at runtime."
            )


# Global settings instance
try:
    settings = Settings()  # type: ignore[call-arg]
    settings.validate_config()
except Exception as e:
    logging.error(f"Configuration Error: {e}")
    # Exit if configuration fails validation
    sys.exit(1)


# Context variable for request ID
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="system"
)


# Configure logging
def setup_logging():
    # Add a filter to inject request_id from contextvars
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            # Get request_id from context variable, with fallback for missing context
            try:
                record.request_id = request_id_var.get()
            except LookupError:
                # Context variable not set (e.g., during startup or in external libraries)
                record.request_id = "system"
            return True

    # Configure basic logging first
    logging.basicConfig(
        level=settings.get_log_level(),
        format="%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Apply the filter to the root logger and all its handlers
    root_logger = logging.getLogger()

    # Add filter to root logger if not already present
    if not any(isinstance(f, RequestIdFilter) for f in root_logger.filters):
        root_logger.addFilter(RequestIdFilter())

    # Also add filter to all handlers to ensure it's applied everywhere
    for handler in root_logger.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())


# Initialize logging immediately
setup_logging()
