"""Search interface for transcripts using FAISS."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys

from src.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def search_transcripts(
    query: str,
    transcripts_dir: Path = Path("./transcripts"),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Search transcripts using semantic similarity.

    Args:
        query: Search query
        transcripts_dir: Directory containing transcripts
        date_from: Filter results from this date
        date_to: Filter results until this date
        k: Number of results to return

    Returns:
        List of search results with metadata
    """
    # Initialize vector store
    metadata_dir = transcripts_dir / "metadata"
    index_path = metadata_dir / "faiss_index.bin"
    store = VectorStore(index_path, metadata_dir)

    # Perform search
    logger.info(f"Searching for: {query}")
    results = store.search(query, k=k * 2)  # Get more for filtering

    if not results:
        logger.info("No results found")
        return []

    # Load transcript metadata and filter by date
    filtered_results = []

    for result in results:
        transcript_path = transcripts_dir.parent / result["file_path"]

        if not transcript_path.exists():
            logger.warning(f"Transcript not found: {transcript_path}")
            continue

        try:
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)

            # Check date filter
            if date_from or date_to:
                recording_dt_str = transcript_data.get("recording_datetime")
                if recording_dt_str:
                    recording_dt = datetime.fromisoformat(recording_dt_str)

                    if date_from and recording_dt < date_from:
                        continue
                    if date_to and recording_dt > date_to:
                        continue

            # Add full metadata to result
            result["transcript_data"] = transcript_data
            filtered_results.append(result)

            if len(filtered_results) >= k:
                break

        except Exception as e:
            logger.warning(f"Error loading {transcript_path}: {e}")

    return filtered_results


def print_results(results: List[Dict[str, Any]]):
    """Print search results in a readable format.

    Args:
        results: List of search results
    """
    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        data = result["transcript_data"]
        similarity = result["similarity"]

        print(f"[{i}] Similarity: {similarity:.2%}")
        print(f"    File: {result['file_path']}")

        if data.get("recording_datetime"):
            dt = datetime.fromisoformat(data["recording_datetime"])
            print(f"    Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        if data.get("language"):
            print(f"    Language: {data['language']}")

        text = data.get("text", "")
        if len(text) > 200:
            print(f"    Preview: {text[:200]}...")
        else:
            print(f"    Text: {text}")

        print()


def main():
    """Main search entry point."""
    parser = argparse.ArgumentParser(
        description="Search transcripts using semantic similarity"
    )
    parser.add_argument("query", type=str, nargs="?", help="Search query")
    parser.add_argument(
        "--date-from", type=str, help="Filter from date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date-to", type=str, help="Filter to date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--similar-to", type=Path, help="Find transcripts similar to this file"
    )
    parser.add_argument(
        "-k", type=int, default=5, help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("./transcripts"),
        help="Transcripts directory (default: ./transcripts)",
    )

    args = parser.parse_args()

    # Parse date filters
    date_from = None
    date_to = None

    if args.date_from:
        try:
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date_from} (use YYYY-MM-DD)")
            sys.exit(1)

    if args.date_to:
        try:
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date_to} (use YYYY-MM-DD)")
            sys.exit(1)

    # Determine query
    query = None

    if args.similar_to:
        # Load transcript and use its text as query
        if not args.similar_to.exists():
            logger.error(f"File not found: {args.similar_to}")
            sys.exit(1)

        try:
            with open(args.similar_to, "r") as f:
                data = json.load(f)
                query = data.get("text", "")

            if not query:
                logger.error(f"No text found in {args.similar_to}")
                sys.exit(1)

            logger.info(f"Finding transcripts similar to: {args.similar_to}")

        except Exception as e:
            logger.error(f"Error reading {args.similar_to}: {e}")
            sys.exit(1)

    elif args.query:
        query = args.query

    else:
        logger.error("Either query or --similar-to must be specified")
        sys.exit(1)

    # Perform search
    try:
        results = search_transcripts(
            query=query,
            transcripts_dir=args.transcripts_dir,
            date_from=date_from,
            date_to=date_to,
            k=args.k,
        )

        print_results(results)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
