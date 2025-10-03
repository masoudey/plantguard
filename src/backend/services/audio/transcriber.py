"""Speech transcription utilities."""
from __future__ import annotations

import io
import speech_recognition as sr


class SpeechTranscriber:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_bytes: bytes) -> str:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
