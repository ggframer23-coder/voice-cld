"""Directory watcher for automatic transcription of new files."""

import argparse
from pathlib import Path


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
    # TODO: Implement directory watching with watchdog
    pass


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
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--model", type=str, default="small", help="Model size to use"
    )
    parser.add_argument("--language", type=str, help="Language code")

    args = parser.parse_args()

    # TODO: Implement watcher
    print(f"Watching {args.watch_dir} for new files...")
    print("Not yet implemented - coming soon!")


if __name__ == "__main__":
    main()
