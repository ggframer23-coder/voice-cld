"""FAISS vector store for semantic search of transcripts."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from src.utils.embeddings import EmbeddingGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for transcript search."""

    def __init__(self, index_path: Path, metadata_dir: Path):
        """Initialize vector store.

        Args:
            index_path: Path to FAISS index file
            metadata_dir: Directory containing metadata files
        """
        self.index_path = index_path
        self.metadata_dir = metadata_dir
        self.id_mapping_path = metadata_dir / "id_mapping.json"
        self.index = None
        self.id_mapping = {}
        self.embedding_gen = EmbeddingGenerator()

    def load_index(self):
        """Load FAISS index from disk."""
        if self.index_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))

            # Load ID mapping
            if self.id_mapping_path.exists():
                with open(self.id_mapping_path, "r") as f:
                    # Convert string keys to int
                    self.id_mapping = {
                        int(k): v for k, v in json.load(f).items()
                    }

            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        else:
            logger.warning(f"Index file not found: {self.index_path}")

    def build_index(self, transcripts_dir: Path):
        """Build FAISS index from transcripts.

        Args:
            transcripts_dir: Directory containing transcript JSON files
        """
        # Find all JSON transcripts
        transcript_files = list(transcripts_dir.rglob("*.json"))

        if not transcript_files:
            logger.warning(f"No transcript files found in {transcripts_dir}")
            return

        logger.info(f"Found {len(transcript_files)} transcript files")

        # Load embedding model
        self.embedding_gen.load_model()

        # Collect texts and metadata
        texts = []
        file_paths = []

        for transcript_file in transcript_files:
            try:
                with open(transcript_file, "r") as f:
                    data = json.load(f)
                    text = data.get("text", "")
                    if text:
                        texts.append(text)
                        file_paths.append(str(transcript_file.relative_to(transcripts_dir.parent)))
            except Exception as e:
                logger.warning(f"Failed to read {transcript_file}: {e}")

        if not texts:
            logger.warning("No valid transcripts found")
            return

        logger.info(f"Generating embeddings for {len(texts)} transcripts...")
        embeddings = self.embedding_gen.encode(texts)

        # Create FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index (dimension={dimension})...")

        # Use IndexFlatL2 for exact search (good for <10k vectors)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

        # Create ID mapping
        self.id_mapping = {i: file_paths[i] for i in range(len(file_paths))}

        # Save index and mapping
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.id_mapping_path, "w") as f:
            json.dump(self.id_mapping, f, indent=2)

        logger.info(f"Index saved: {self.index.ntotal} vectors")
        logger.info(f"Index file: {self.index_path}")
        logger.info(f"ID mapping: {self.id_mapping_path}")

    def search(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar transcripts.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of matching transcript metadata
        """
        if self.index is None:
            self.load_index()

        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not loaded")
            return []

        # Generate query embedding
        query_embedding = self.embedding_gen.encode_single(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.id_mapping:
                results.append(
                    {
                        "file_path": self.id_mapping[idx],
                        "distance": float(dist),
                        "similarity": 1.0 / (1.0 + float(dist)),
                    }
                )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary of statistics
        """
        if self.index is None:
            self.load_index()

        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else 0,
            "index_type": type(self.index).__name__ if self.index else "None",
            "index_file": str(self.index_path),
            "mapping_file": str(self.id_mapping_path),
        }

        return stats


def main():
    """Main vectorstore management entry point."""
    parser = argparse.ArgumentParser(
        description="Manage FAISS vector index for transcripts"
    )
    parser.add_argument(
        "command", choices=["rebuild", "stats"], help="Command to run"
    )
    parser.add_argument(
        "transcripts_dir",
        type=Path,
        nargs="?",
        default=Path("./transcripts"),
        help="Transcripts directory (default: ./transcripts)",
    )

    args = parser.parse_args()

    # Initialize vector store
    metadata_dir = args.transcripts_dir / "metadata"
    index_path = metadata_dir / "faiss_index.bin"
    store = VectorStore(index_path, metadata_dir)

    if args.command == "rebuild":
        logger.info("Rebuilding FAISS index...")
        store.build_index(args.transcripts_dir / "json")
        logger.info("Done!")

    elif args.command == "stats":
        stats = store.get_stats()
        logger.info("FAISS Index Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
