"""CNN-LSTM audio classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """Speech classifier with configurable convolutional front-end and LSTM head."""

    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        class_names: list[str] | None = None,
        checkpoint: Path | None = None,
        *,
        conv_channels: Sequence[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        n_mfcc: int = 40,
    ) -> None:
        super().__init__()
        self.device = device
        self.class_names = class_names or [f"class_{idx}" for idx in range(num_classes)]

        arch_config: Dict[str, object] = {
            "conv_channels": tuple(conv_channels or (32, 64)),
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "n_mfcc": n_mfcc,
        }

        state = None
        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            arch_config.update(state.get("arch", {}))
            checkpoint_classes = state.get("class_names")
            if checkpoint_classes:
                self.class_names = list(checkpoint_classes)
                num_classes = len(self.class_names)
            self.pad_to = state.get("pad_to")
        else:
            self.pad_to: int | None = None

        self._build_layers(num_classes, arch_config)
        self.to(self.device)

        if state is not None:
            state_dict = state.get("state_dict", state)
            self.load_state_dict(state_dict, strict=False)
        self.eval()

    def _build_layers(self, num_classes: int, arch: Dict[str, object]) -> None:
        self.conv_channels = tuple(arch.get("conv_channels", (32, 64)))
        self.lstm_hidden = int(arch.get("lstm_hidden", 128))
        self.lstm_layers = int(arch.get("lstm_layers", 2))
        self.bidirectional = bool(arch.get("bidirectional", True))
        self.dropout_rate = float(arch.get("dropout", 0.3))
        self.n_mfcc = int(arch.get("n_mfcc", 40))

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in self.conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        lstm_input = self.conv_channels[-1] * self.n_mfcc
        self.lstm = nn.LSTM(
            lstm_input,
            self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.lstm_layers > 1 else 0.0,
        )

        classifier_in = self.lstm_hidden * (2 if self.bidirectional else 1)
        classifier_layers: list[nn.Module] = [nn.Linear(classifier_in, self.lstm_hidden), nn.ReLU()]
        if self.dropout_rate > 0:
            classifier_layers.append(nn.Dropout(self.dropout_rate))
        classifier_layers.append(nn.Linear(self.lstm_hidden, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.post_lstm_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        b, t, c, h = x.shape
        x = x.view(b, t, c * h)
        outputs, _ = self.lstm(x)
        last_hidden = outputs[:, -1]
        last_hidden = self.post_lstm_dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits

    def predict(self, features: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.forward(features)
            return logits.squeeze(0).cpu().numpy()
