"""Multimodal fusion head combining vision, text, and audio embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


class MultimodalFusionHead(nn.Module):
    def __init__(
        self,
        device: torch.device,
        image_dim: int = 4,
        text_dim: int = 4,
        audio_dim: int | None = None,
        num_classes: int = 4,
        class_names: Sequence[str] | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.num_classes = num_classes
        self.class_names: List[str] = list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]

        fused_dim = image_dim + text_dim + (audio_dim or 0)
        hidden_dim = max(128, fused_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        ).to(device)

        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            metadata = state.get("metadata", {})
            self.class_names = metadata.get("class_names", self.class_names)
            self.image_dim = metadata.get("image_dim", self.image_dim)
            self.text_dim = metadata.get("text_dim", self.text_dim)
            self.audio_dim = metadata.get("audio_dim", self.audio_dim)
            self.num_classes = metadata.get("num_classes", self.num_classes)
            self.mlp.load_state_dict(state.get("state_dict", state))

        self.eval()

    def forward(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
        audio_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [image_embed, text_embed]
        if audio_embed is not None:
            parts.append(audio_embed)
        fused = torch.cat(parts, dim=-1).to(self.device)
        return self.mlp(fused)

    def predict(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
        audio_embed: torch.Tensor | None = None,
    ) -> np.ndarray:
        logits = self.forward(image_embed, text_embed, audio_embed)
        return logits.squeeze(0).detach().cpu().numpy()
