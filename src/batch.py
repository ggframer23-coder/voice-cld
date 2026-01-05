"""Batch processing for multiple audio files."""

import argparse
from pathlib import Path
from typing import List


def process_directory(
    input_dir: Path,
    output_dir: Path,
    model: str = "small",
    language: str = None,
    only_new: bool = False,
) -> List[Path]:
    """Process all audio files in a directory.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory for output transcripts
        model: Model size to use
        language: Language code (None for auto-detect)
        only_new: Only process files not yet transcribed

    Returns:
        List of processed file paths
    """
    # TODO: Implement batch processing
    return []


def main():
    """Main batch processing entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process audio files for transcription"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing audio files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./transcripts"),
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--model", type=str, default="small", help="Model size to use"
    )
    parser.add_argument("--language", type=str, help="Language code")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only process files not yet transcribed",
    )

    args = parser.parse_args()

    # TODO: Implement batch processing
    print(f"Batch processing {args.input_dir}...")
    print("Not yet implemented - coming soon!")


if __name__ == "__main__":
    main()
