"""Audio file handling and validation utilities."""

from pathlib import Path
from typing import Optional, Tuple


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
    # TODO: Implement using mutagen or similar
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

    # TODO: Add more validation (file size, corruption check, etc.)

    return True, ""
