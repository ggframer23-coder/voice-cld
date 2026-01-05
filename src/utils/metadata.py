"""Extract recording date/time and metadata from audio files."""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
import re
import os
import logging

logger = logging.getLogger(__name__)


def extract_recording_datetime(file_path: Path) -> Optional[datetime]:
    """Extract recording date/time from audio file metadata.

    Priority order:
    1. Audio file metadata (ID3, MP4, FLAC tags)
    2. File creation timestamp
    3. Filename parsing

    Args:
        file_path: Path to audio file

    Returns:
        Recording datetime, or None if not found
    """
    # Try audio metadata first
    try:
        audio = MutagenFile(file_path)

        if audio is not None:
            # Try different metadata tag formats
            date_tags = [
                "recording_date",
                "creation_time",
                "date",
                "TDRC",  # ID3 recording time
                "TDOR",  # ID3 original release time
                "\xa9day",  # MP4 date
            ]

            for tag in date_tags:
                if tag in audio:
                    date_str = str(audio[tag][0] if isinstance(audio[tag], list) else audio[tag])
                    # Try parsing the date string
                    dt = _parse_date_string(date_str)
                    if dt:
                        return dt

    except Exception as e:
        logger.debug(f"Could not extract metadata from {file_path}: {e}")

    # Try parsing filename for date
    filename_date = _parse_filename_date(file_path.name)
    if filename_date:
        return filename_date

    # Fallback to file creation time
    try:
        ctime = os.path.getctime(file_path)
        return datetime.fromtimestamp(ctime)
    except Exception as e:
        logger.warning(f"Could not get creation time for {file_path}: {e}")
        return None


def _parse_date_string(date_str: str) -> Optional[datetime]:
    """Try to parse various date formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def _parse_filename_date(filename: str) -> Optional[datetime]:
    """Try to extract date from filename patterns."""
    patterns = [
        r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})(\d{2})(\d{2})",  # YYYY-MM-DD_HHMMSS
        r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})",  # YYYYMMDD_HHMMSS
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 6:  # With time
                    return datetime(
                        int(groups[0]),
                        int(groups[1]),
                        int(groups[2]),
                        int(groups[3]),
                        int(groups[4]),
                        int(groups[5]),
                    )
                elif len(groups) == 3:  # Date only
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            except ValueError:
                continue

    return None


def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract all metadata from audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary of metadata
    """
    from .audio import get_audio_duration

    return {
        "filename": file_path.name,
        "recording_datetime": extract_recording_datetime(file_path),
        "duration": get_audio_duration(file_path),
        "format": file_path.suffix,
        "file_size": file_path.stat().st_size,
    }
