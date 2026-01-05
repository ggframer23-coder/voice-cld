"""Tests for transcriber module."""

import pytest
from pathlib import Path
from src.transcriber import Transcriber


class TestTranscriber:
    """Test cases for Transcriber class."""

    def test_transcriber_init(self):
        """Test transcriber initialization."""
        transcriber = Transcriber(model_size="small", device="cpu")
        assert transcriber.model_size == "small"
        assert transcriber.device == "cpu"

    def test_transcriber_with_language(self):
        """Test transcriber with specific language."""
        transcriber = Transcriber(model_size="base", language="en")
        assert transcriber.language == "en"

    def test_transcriber_default_values(self):
        """Test transcriber default values."""
        transcriber = Transcriber()
        assert transcriber.model_size == "small"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert transcriber.language is None


class TestUtils:
    """Test cases for utility modules."""

    def test_audio_validation(self):
        """Test audio file validation."""
        from src.utils.audio import is_audio_file

        assert is_audio_file(Path("test.mp3")) is True
        assert is_audio_file(Path("test.wav")) is True
        assert is_audio_file(Path("test.txt")) is False

    def test_supported_formats(self):
        """Test all supported audio formats."""
        from src.utils.audio import SUPPORTED_FORMATS

        assert ".mp3" in SUPPORTED_FORMATS
        assert ".wav" in SUPPORTED_FORMATS
        assert ".m4a" in SUPPORTED_FORMATS
