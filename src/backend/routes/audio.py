"""Endpoints for audio-based symptom descriptions."""
from fastapi import APIRouter, File, UploadFile, status

from ..services import models, preprocess

router = APIRouter()


@router.post("/classify", status_code=status.HTTP_200_OK)
async def classify_audio(file: UploadFile = File(...)) -> dict:
    """Transcribe speech input and return predicted symptom class."""
    audio_bytes = await file.read()
    audio_model = models.get_audio_model()
    pad_to = getattr(audio_model, "pad_to", None)
    features = preprocess.transform_audio_bytes(audio_bytes, pad_to=pad_to)
    probs = audio_model.predict(features)
    transcript = models.get_transcriber().transcribe(audio_bytes)
    return {
        "transcript": transcript,
        "predictions": models.format_predictions(probs, labels=audio_model.class_names),
    }
