"""Directory watcher for automatic transcription of new files."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import time
import sys

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from src.transcriber import Transcriber
from src.utils.audio import is_audio_file, validate_audio_file
from src.utils.metadata import extract_metadata
from src.utils.formats import save_transcription

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioFileHandler(FileSystemEventHandler):
    """Handler for new audio file events."""

    def __init__(self, output_dir: Path, transcriber: Transcriber):
        """Initialize handler.

        Args:
            output_dir: Directory for output transcripts
            transcriber: Transcriber instance
        """
        self.output_dir = output_dir
        self.transcriber = transcriber
        self.processing = set()  # Track files being processed

    def on_created(self, event: FileSystemEvent):
        """Handle file creation event.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's an audio file
        if not is_audio_file(file_path):
            return

        # Avoid processing the same file twice
        if file_path in self.processing:
            return

        self.processing.add(file_path)

        try:
            # Wait a bit to ensure file is fully written
            time.sleep(1)

            logger.info(f"\n{'='*60}")
            logger.info(f"New audio file detected: {file_path.name}")

            # Validate
            is_valid, error_msg = validate_audio_file(file_path)
            if not is_valid:
                logger.error(f"Invalid file: {error_msg}")
                return

            # Extract metadata
            metadata = extract_metadata(file_path)
            logger.info(f"Recording date: {metadata.get('recording_datetime')}")

            # Transcribe
            logger.info("Starting transcription...")
            transcription = self.transcriber.transcribe(file_path)

            # Save
            recording_dt = metadata.get("recording_datetime", datetime.now())
            timestamp = recording_dt.strftime("%Y-%m-%d_%H%M%S")
            output_file = self.output_dir / f"transcript_{timestamp}.json"

            save_transcription(transcription, metadata, output_file, format="json")

            logger.info(f"Transcription saved: {output_file.name}")
            logger.info(
                f"Language: {transcription['language']}, "
                f"Segments: {len(transcription['segments'])}"
            )
            logger.info(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

        finally:
            self.processing.discard(file_path)


def watch_directory(
    watch_dir: Path,
    output_dir: Path,
    model: str = "small",
    language: str = None,
):
    """Watch directory for new audio files and transcribe them.

    Args:
        watch_dir: Directory to monitor
        output_dir: Directory for output transcripts
        model: Model size to use
        language: Language code (None for auto-detect)
    """
    if not watch_dir.exists() or not watch_dir.is_dir():
        raise ValueError(f"Invalid watch directory: {watch_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize transcriber (load model once)
    logger.info(f"Initializing transcriber (model={model})...")
    transcriber = Transcriber(model_size=model, device="cpu", language=language)
    transcriber.load_model()

    # Set up file watcher
    event_handler = AudioFileHandler(output_dir, transcriber)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)

    logger.info(f"\nWatching directory: {watch_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Press Ctrl+C to stop...\n")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping watcher...")
        observer.stop()

    observer.join()
    logger.info("Watcher stopped.")


def main():
    """Main watcher entry point."""
    parser = argparse.ArgumentParser(
        description="Watch directory for new audio files to transcribe"
    )
    parser.add_argument("watch_dir", type=Path, help="Directory to monitor")
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

    args = parser.parse_args()

    try:
        watch_directory(
            watch_dir=args.watch_dir,
            output_dir=args.output_dir,
            model=args.model,
            language=args.language,
        )
    except Exception as e:
        logger.error(f"Watcher failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
