import logging
from functools import lru_cache
from typing import NoReturn

import numpy as np
from fastapi import HTTPException, status
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential_jitter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Define the retry strategy (Requirement 10: exponential backoff with jitter, max 3 attempts)
llm_retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    reraise=True,
)


class LLMService:
    def __init__(self):
        # Configuration is validated in config.py, so we assume providers and keys are valid here.
        self.llm_model_name = settings.LLM_MODEL
        self.emb_model_name = settings.EMB_MODEL
        self.emb_dim = settings.EMB_DIM

        self._initialize_models(settings.OPENAI_API_KEY)
        self._define_summarization_chain()

    def _initialize_models(self, api_key: str):
        logger.info(
            f"Initializing LLM: {self.llm_model_name}, Embeddings: {self.emb_model_name} ({self.emb_dim} dim)"
        )

        # Check for OpenAI API key if using OpenAI providers
        if (
            settings.LLM_PROVIDER == "openai" or settings.EMB_PROVIDER == "openai"
        ) and not api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI providers")

        # Initialize Chat Model
        self.chat_model = ChatOpenAI(
            model=self.llm_model_name,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0.0,  # Factual summarization requires low temperature
        )

        # Initialize Embedding Model
        self.embedding_model = OpenAIEmbeddings(
            model=self.emb_model_name,
            api_key=SecretStr(api_key) if api_key else None,
            dimensions=self.emb_dim,  # Pass the configured dimension
        )

    def _define_summarization_chain(self):
        # Requirement 6: System prompt for concise, neutral summary (<= 5 sentences)
        system_prompt = (
            "You are a concise, neutral news summarizer. Your goal is to extract key facts objectively without speculation or opinion. "
            "Include what happened (key events), who (key entities), where, and when. "
            "Your summary MUST be 5 sentences or fewer. "
            "Explicitly mention the publication date and the source domain provided in the context."
        )
        # Requirement 6: User content includes article text, title, date, and source.
        user_prompt = (
            "Summarize the following news article.\n\n"
            "Source Domain: {source_domain}\n"
            "Published Date: {published_at}\n"
            "Title: {title}\n\n"
            "Article Content:\n{content}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        # Requirement 6: Chain: prompt -> LLM -> text output parser
        self.summarization_chain = prompt_template | self.chat_model | StrOutputParser()

    @llm_retry_decorator
    async def generate_summary(
        self, title: str, content: str, source_domain: str, published_at: str | None
    ) -> str:
        logger.info("Generating summary...")
        try:
            # Truncate content if it exceeds max characters
            truncated_content = content[: settings.MAX_CONTENT_CHARS]

            # Use async invoke (ainvoke)
            summary = await self.summarization_chain.ainvoke(
                {
                    "title": title or "N/A",
                    "content": truncated_content,
                    "source_domain": source_domain or "N/A",
                    "published_at": published_at or "Unknown",
                }
            )
            logger.info("Summary generation complete.")
            return summary
        except Exception as e:
            self._handle_provider_error(e, "LLM")

    @llm_retry_decorator
    async def generate_embedding(self, text: str) -> list[float]:
        logger.info("Generating embedding...")
        try:
            # Use async embedding generation (aembed_query)
            embedding = await self.embedding_model.aembed_query(text)

            # Requirement 6: Normalize embeddings client-side before storing if using cosine.
            normalized_embedding = self._normalize_l2(embedding)

            logger.info("Embedding generation complete.")
            return normalized_embedding
        except Exception as e:
            self._handle_provider_error(e, "Embedding")

    def _normalize_l2(self, embedding: list[float]) -> list[float]:
        """Normalize the embedding vector to unit length (L2 norm)."""
        vector = np.array(embedding)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return embedding
        return (vector / norm).tolist()

    def compose_embedding_input(self, title: str, summary: str, content: str) -> str:
        # Requirement 6: Input to embedding: title + newline + summary + newline + content
        combined = f"{title or ''}\n{summary}\n{content}"
        # Truncate combined text if it exceeds max characters
        return combined[: settings.MAX_CONTENT_CHARS]

    def _handle_provider_error(
        self, exception: Exception, provider_type: str
    ) -> NoReturn:
        # Check if the error occurred after retries (tenacity wraps the original exception)
        if isinstance(exception, RetryError):
            logger.error(
                f"{provider_type} provider error after retries: {exception.last_attempt.exception()}"
            )
        else:
            logger.error(
                f"Unexpected {provider_type} provider error: {exception}", exc_info=True
            )

        # Requirement 3: 502 LLM or embeddings provider error
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error communicating with {provider_type} provider.",
        )


@lru_cache()
def get_llm_service() -> LLMService:
    # Initialize once at startup (Singleton pattern via lru_cache)
    return LLMService()
