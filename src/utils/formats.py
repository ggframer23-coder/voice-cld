"""Output formatters for transcription results."""

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json


def format_as_text(
    transcription: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Format transcription as plain text.

    Args:
        transcription: Transcription results
        metadata: Audio file metadata

    Returns:
        Formatted plain text
    """
    lines = []

    # Add metadata header
    if metadata.get("recording_datetime"):
        dt = metadata["recording_datetime"]
        lines.append(f"Recording Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if metadata.get("filename"):
        lines.append(f"Original File: {metadata['filename']}")

    if metadata.get("duration"):
        duration = int(metadata["duration"])
        mins, secs = divmod(duration, 60)
        lines.append(f"Duration: {mins}:{secs:02d}")

    lines.append(f"Language: {transcription.get('language', 'unknown')}")
    lines.append("---")
    lines.append("")
    lines.append(transcription.get("text", ""))

    return "\n".join(lines)


def format_as_json(
    transcription: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """Format transcription as JSON.

    Args:
        transcription: Transcription results
        metadata: Audio file metadata

    Returns:
        Formatted JSON string
    """
    output = {
        "recording_datetime": metadata.get("recording_datetime").isoformat()
        if metadata.get("recording_datetime")
        else None,
        "original_filename": metadata.get("filename"),
        "transcription_datetime": datetime.now().isoformat(),
        "audio_duration_seconds": metadata.get("duration"),
        "language": transcription.get("language", "unknown"),
        "text": transcription.get("text", ""),
        "segments": transcription.get("segments", []),
    }

    return json.dumps(output, indent=2, ensure_ascii=False)


def format_as_srt(transcription: Dict[str, Any]) -> str:
    """Format transcription as SRT subtitles.

    Args:
        transcription: Transcription results with segments

    Returns:
        Formatted SRT string
    """
    # TODO: Implement SRT formatting
    return ""


def save_transcription(
    transcription: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path,
    format: str = "json",
):
    """Save transcription to file.

    Args:
        transcription: Transcription results
        metadata: Audio file metadata
        output_path: Output file path
        format: Output format (txt/json/srt)
    """
    if format == "txt":
        content = format_as_text(transcription, metadata)
    elif format == "json":
        content = format_as_json(transcription, metadata)
    elif format == "srt":
        content = format_as_srt(transcription)
    else:
        raise ValueError(f"Unsupported format: {format}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
