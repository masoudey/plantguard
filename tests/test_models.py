"""Smoke tests ensuring models forward pass successfully."""
from __future__ import annotations

import torch

from plantguard.src.backend.services import models


def test_vision_model_predict():
    tensor = torch.randn(1, 3, 224, 224)
    vision_model = models.get_vision_model()
    preds = vision_model.predict(tensor)
    assert preds.shape[0] == len(vision_model.class_names)
