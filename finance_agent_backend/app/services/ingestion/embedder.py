import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger("finance_agent.ingestion.embedder")


class SECBatchEmbedder:
    """Generates batch embeddings using OpenAI text-embedding-3-small (1536 dims)."""

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = settings.OPENAI_API_KEY

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set in settings/environment. Real embeddings cannot be generated.")
            self.client = None
        else:
            self.client = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
            )

    def embed_texts(self, texts: List[str], on_batch_progress=None) -> List[List[float]]:
        """Computes embeddings in batches for a list of strings."""
        if not texts:
            return []

        if not self.client:
            logger.warning("No OpenAI client available. Returning zero-vectors for offline/dry-run mode.")
            return [[0.0] * 1536 for _ in texts]

        all_embeddings: List[List[float]] = []
        total = len(texts)
        total_batches = (total + self.batch_size - 1) // self.batch_size

        for i in range(0, total, self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = texts[i : i + self.batch_size]
            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            try:
                batch_embeddings = self.client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                if on_batch_progress:
                    on_batch_progress(batch_num, total_batches)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch starting at {i}: {e}")
                raise e

        return all_embeddings
