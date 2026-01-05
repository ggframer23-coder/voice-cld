"""Generate embeddings for FAISS vector search."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding generator.

        Args:
            model_name: sentence-transformers model name
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (N x dimension)
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text.

        Args:
            text: Text string

        Returns:
            Numpy array embedding (dimension,)
        """
        return self.encode([text])[0]
