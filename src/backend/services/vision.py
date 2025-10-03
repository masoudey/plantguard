"""Vision classifier wrapper around a fine-tuned ResNet."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torchvision import models


class VisionClassifier(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        class_names: Sequence[str] | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.model.fc.in_features
        self._class_names: List[str] = list(class_names) if class_names else [
            "healthy",
            "powdery_mildew",
            "early_blight",
            "leaf_spot",
        ]
        self.model.fc = torch.nn.Linear(num_features, len(self._class_names))

        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            state_dict = state.get("state_dict", state)
            checkpoint_classes = state.get("class_names")
            if checkpoint_classes:
                self._class_names = list(checkpoint_classes)
                self.model.fc = torch.nn.Linear(num_features, len(self._class_names))
                self.model.fc.to(device)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(device)
        self.eval()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor.to(self.device))

    def predict(self, tensor: torch.Tensor) -> np.ndarray:
        logits = self.forward(tensor)
        return logits.squeeze(0).detach().cpu().numpy()

    @property
    def class_names(self) -> List[str]:
        return self._class_names
