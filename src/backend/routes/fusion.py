"""Endpoints for multimodal diagnosis."""
from fastapi import APIRouter, UploadFile, File, status

from ..services import fusion as fusion_service

router = APIRouter()


@router.post("/diagnose", status_code=status.HTTP_200_OK)
async def run_multimodal_diagnosis(
    image: UploadFile = File(...),
    audio: UploadFile | None = File(default=None),
    question: str | None = None,
) -> dict:
    """Combine modalities into a single diagnosis with treatment guidance."""
    image_bytes = await image.read()
    audio_bytes = await audio.read() if audio else None
    result = fusion_service.run_fusion(
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        question=question,
    )
    return result
