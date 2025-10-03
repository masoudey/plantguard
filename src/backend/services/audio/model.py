"""CNN-LSTM audio classifier placeholder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        class_names: list[str] | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.class_names = class_names or [f"class_{idx}" for idx in range(num_classes)]
        self.pad_to: int | None = None
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(16 * 20, 64, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)
        self.to(self.device)

        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            state_dict = state.get("state_dict", state)
            checkpoint_classes = state.get("class_names")
            if checkpoint_classes:
                self.class_names = list(checkpoint_classes)
                self.classifier = nn.Linear(64, len(self.class_names))
                self.classifier.to(self.device)
            self.pad_to = state.get("pad_to")
            self.load_state_dict(state_dict, strict=False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, w, c * h)
        _, (h_n, _) = self.lstm(x)
        logits = self.classifier(h_n[-1])
        return logits

    def predict(self, features: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.forward(features)
            return logits.squeeze(0).cpu().numpy()
