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
        metadata: dict[str, object] = {}
        state_dict: dict[str, torch.Tensor] | None = None
        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            metadata = state.get("metadata", {})
            raw_state = state.get("state_dict", state)
            if isinstance(raw_state, dict):
                state_dict = raw_state

        self.class_names: List[str] = list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
        self.image_dim = metadata.get("image_dim", image_dim)  # type: ignore[arg-type]
        self.text_dim = metadata.get("text_dim", text_dim)  # type: ignore[arg-type]
        self.audio_dim = metadata.get("audio_dim", audio_dim)  # type: ignore[arg-type]
        self.num_classes = metadata.get("num_classes", num_classes)  # type: ignore[arg-type]
        if metadata.get("class_names"):
            self.class_names = list(metadata["class_names"])  # type: ignore[index]

        fused_dim = self.image_dim + self.text_dim + (self.audio_dim or 0)
        hidden_dim = max(128, fused_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, self.num_classes),
        ).to(device)

        if state_dict:
            if any(key.startswith("mlp.") for key in state_dict.keys()):
                self.load_state_dict(state_dict, strict=False)
            else:
                self.mlp.load_state_dict(state_dict)  # type: ignore[arg-type]

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
