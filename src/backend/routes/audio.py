"""Endpoints for audio-based symptom descriptions."""
from fastapi import APIRouter, File, UploadFile, status

from ..services import models, preprocess

router = APIRouter()


@router.post("/classify", status_code=status.HTTP_200_OK)
async def classify_audio(file: UploadFile = File(...)) -> dict:
    """Transcribe speech input and return predicted symptom class."""
    audio_bytes = await file.read()
    audio_model = models.get_audio_model()
    sample_rate = getattr(audio_model, "sample_rate", 16000)
    max_length = getattr(audio_model, "max_length", None)
    waveforms, lengths = preprocess.transform_audio_bytes(
        audio_bytes,
        sample_rate=sample_rate,
        max_length=max_length,
        filename=file.filename,
        content_type=file.content_type,
    )
    probs = audio_model.predict(waveforms, lengths)
    transcript = models.get_transcriber().transcribe(audio_bytes)
    return {
        "transcript": transcript,
        "predictions": models.format_predictions(probs, labels=audio_model.class_names),
    }
