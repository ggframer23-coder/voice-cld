"""Batch processing for multiple audio files."""

import argparse
import logging
from pathlib import Path
from typing import List
from datetime import datetime
import sys

from src.transcriber import Transcriber
from src.utils.audio import is_audio_file, validate_audio_file
from src.utils.metadata import extract_metadata
from src.utils.formats import save_transcription

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input directory: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and is_audio_file(file_path):
            audio_files.append(file_path)

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return []

    logger.info(f"Found {len(audio_files)} audio files")

    # Initialize transcriber (load model once for all files)
    transcriber = Transcriber(model_size=model, device="cpu", language=language)
    transcriber.load_model()

    processed_files = []
    failed_files = []

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        try:
            # Validate file
            is_valid, error_msg = validate_audio_file(audio_file)
            if not is_valid:
                logger.error(f"Skipping invalid file: {error_msg}")
                failed_files.append((audio_file, error_msg))
                continue

            # Extract metadata
            metadata = extract_metadata(audio_file)

            # Generate output filename
            recording_dt = metadata.get("recording_datetime", datetime.now())
            timestamp = recording_dt.strftime("%Y-%m-%d_%H%M%S")
            output_file = output_dir / f"transcript_{timestamp}.json"

            # Skip if already processed (only_new mode)
            if only_new and output_file.exists():
                logger.info(f"Skipping (already transcribed): {output_file.name}")
                continue

            # Transcribe
            transcription = transcriber.transcribe(audio_file)

            # Save
            save_transcription(transcription, metadata, output_file, format="json")

            logger.info(
                f"Saved: {output_file.name} "
                f"({len(transcription['segments'])} segments, "
                f"{transcription['language']})"
            )
            processed_files.append(audio_file)

        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            failed_files.append((audio_file, str(e)))
            continue

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing complete!")
    logger.info(f"  Processed: {len(processed_files)}/{len(audio_files)}")
    if failed_files:
        logger.info(f"  Failed: {len(failed_files)}")
        for file_path, error in failed_files:
            logger.info(f"    - {file_path.name}: {error}")
    logger.info(f"{'='*60}")

    return processed_files


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
        help="Output directory for transcripts (default: ./transcripts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Model size to use (default: small)",
    )
    parser.add_argument("--language", type=str, help="Language code (auto-detect if not specified)")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only process files not yet transcribed",
    )

    args = parser.parse_args()

    try:
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            language=args.language,
            only_new=args.only_new,
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
