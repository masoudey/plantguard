"""Endpoints for image-based inference."""
from fastapi import APIRouter, File, UploadFile, status

from ..services import models, preprocess

router = APIRouter()


@router.post("/diagnose", status_code=status.HTTP_200_OK)
async def diagnose_leaf(file: UploadFile = File(...)) -> dict:
    """Run the vision classifier on an uploaded image and return predictions."""
    image_bytes = await file.read()
    tensor = preprocess.transform_image_bytes(image_bytes)
    vision_model = models.get_vision_model()
    probs = vision_model.predict(tensor)
    return models.format_predictions(probs, labels=vision_model.class_names)
