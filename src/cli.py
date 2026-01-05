"""Command-line interface for audio transcription."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

from src.transcriber import Transcriber
from src.utils.audio import validate_audio_file
from src.utils.metadata import extract_metadata
from src.utils.formats import save_transcription

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to text using faster-whisper"
    )
    parser.add_argument("audio_file", type=Path, help="Audio file to transcribe")
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Model size to use (default: small)",
    )
    parser.add_argument("--output", type=Path, help="Output file path (optional)")
    parser.add_argument(
        "--language", type=str, help="Language code (auto-detect if not specified)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Quantization type (default: int8)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json", "srt"],
        help="Output format (default: txt)",
    )

    args = parser.parse_args()

    # Validate audio file
    logger.info(f"Validating audio file: {args.audio_file}")
    is_valid, error_msg = validate_audio_file(args.audio_file)
    if not is_valid:
        logger.error(f"Invalid audio file: {error_msg}")
        sys.exit(1)

    # Extract metadata
    logger.info("Extracting metadata...")
    metadata = extract_metadata(args.audio_file)
    logger.info(f"Recording date: {metadata.get('recording_datetime')}")
    if metadata.get("duration"):
        logger.info(f"Duration: {metadata['duration']:.1f}s")

    # Initialize transcriber
    logger.info(f"Initializing transcriber (model={args.model})")
    transcriber = Transcriber(
        model_size=args.model,
        device="cpu",
        compute_type=args.quantization,
        language=args.language,
    )

    # Perform transcription
    try:
        transcription = transcriber.transcribe(args.audio_file)

        logger.info(f"Transcription complete!")
        logger.info(f"Language detected: {transcription['language']}")
        logger.info(f"Number of segments: {len(transcription['segments'])}")

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Generate output filename based on input
            recording_dt = metadata.get("recording_datetime", datetime.now())
            timestamp = recording_dt.strftime("%Y-%m-%d_%H%M%S")
            output_path = Path(f"transcript_{timestamp}.{args.format}")

        # Save transcription
        logger.info(f"Saving transcription to: {output_path}")
        save_transcription(transcription, metadata, output_path, format=args.format)

        # Print preview
        text = transcription["text"]
        if len(text) > 200:
            print(f"\nPreview: {text[:200]}...")
        else:
            print(f"\nTranscription: {text}")

        logger.info("Done!")

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
