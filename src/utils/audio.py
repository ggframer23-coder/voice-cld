"""Audio file handling and validation utilities."""

from pathlib import Path
from typing import Optional, Tuple
from mutagen import File as MutagenFile
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"}


def is_audio_file(file_path: Path) -> bool:
    """Check if file is a supported audio format.

    Args:
        file_path: Path to file

    Returns:
        True if file is supported audio format
    """
    return file_path.suffix.lower() in SUPPORTED_FORMATS


def get_audio_duration(file_path: Path) -> Optional[float]:
    """Get audio file duration in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds, or None if error
    """
    try:
        audio = MutagenFile(file_path)
        if audio is not None and hasattr(audio.info, "length"):
            return float(audio.info.length)
    except Exception as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")

    return None


def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
    """Validate audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, "File does not exist"

    if not is_audio_file(file_path):
        return False, f"Unsupported format: {file_path.suffix}"

    # Check if file is readable
    if not file_path.is_file():
        return False, "Path is not a file"

    # Check file size (should be > 0)
    if file_path.stat().st_size == 0:
        return False, "File is empty"

    # Try to read metadata
    try:
        audio = MutagenFile(file_path)
        if audio is None:
            return False, "Could not read audio file metadata"
    except Exception as e:
        return False, f"Corrupted or invalid audio file: {e}"

    return True, ""
