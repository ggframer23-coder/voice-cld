"""Generate embeddings for FAISS vector search."""

from typing import List
import numpy as np


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
        # TODO: Implement model loading
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (N x dimension)
        """
        # TODO: Implement encoding
        # For now, return dummy embeddings
        return np.zeros((len(texts), 384))

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text.

        Args:
            text: Text string

        Returns:
            Numpy array embedding (dimension,)
        """
        return self.encode([text])[0]
