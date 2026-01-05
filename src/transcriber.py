"""Core transcription engine using faster-whisper."""

from typing import Optional, Dict, List, Any
from pathlib import Path
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)


class Transcriber:
    """Handles audio transcription using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ):
        """Initialize the transcriber.

        Args:
            model_size: Model size (tiny/base/small/medium/large-v2)
            device: Device to use (cpu/cuda)
            compute_type: Compute type (int8/float16/float32)
            language: Language code (None for auto-detect)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None

    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            logger.info(
                f"Loading {self.model_size} model (device={self.device}, "
                f"compute_type={self.compute_type})"
            )
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Model loaded successfully")

    def transcribe(
        self, audio_file: Path, output_format: str = "json"
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_file: Path to audio file
            output_format: Output format (json/txt/srt)

        Returns:
            Dictionary containing transcription results
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing: {audio_file}")

        # Perform transcription
        segments, info = self.model.transcribe(
            str(audio_file),
            language=self.language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection
        )

        # Process segments
        full_text = []
        segment_list = []

        for segment in segments:
            full_text.append(segment.text)
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

        result = {
            "text": " ".join(full_text).strip(),
            "segments": segment_list,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        logger.info(
            f"Transcription complete: {len(segment_list)} segments, "
            f"{info.duration:.1f}s, language={info.language}"
        )

        return result

    def transcribe_file(self, audio_file: Path) -> str:
        """Transcribe audio file and return text.

        Args:
            audio_file: Path to audio file

        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_file)
        return result.get("text", "")
