"""Command-line interface for audio transcription."""

import argparse
from pathlib import Path
from typing import Optional


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
        help="Model size to use",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file path (optional)"
    )
    parser.add_argument(
        "--language", type=str, help="Language code (auto-detect if not specified)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Quantization type",
    )

    args = parser.parse_args()

    # TODO: Implement transcription
    print(f"Transcribing {args.audio_file} with model {args.model}...")
    print("Not yet implemented - coming soon!")


if __name__ == "__main__":
    main()
