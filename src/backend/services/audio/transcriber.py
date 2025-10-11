"""Speech transcription utilities."""
from __future__ import annotations

import io
import logging

import speech_recognition as sr


logger = logging.getLogger(__name__)


class SpeechTranscriber:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio = self.recognizer.record(source)
        except Exception as exc:  # pragma: no cover - defensive catch
            logger.warning("Speech transcription failed to read audio: %s", exc)
            return ""

        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except (sr.RequestError, OSError) as exc:
            logger.warning("Speech transcription failed: %s", exc)
            return ""
