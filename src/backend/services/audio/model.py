"""Transformer-based audio classifier."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float, batch_first: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return self.dropout(x + self.pe[:, : x.size(1)])
        return self.dropout(x + self.pe[: x.size(0)])


class AudioTransformerClassifier(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        class_names: Optional[list[str]] = None,
        checkpoint: Optional[Path] = None,
        *,
        sample_rate: int = 16000,
        max_length: int = 96000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.device = device
        self.class_names = class_names or [f"class_{idx}" for idx in range(num_classes)]
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
            normalized=False,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
        self.proj = nn.Conv1d(n_mels, embed_dim, kernel_size=1)

        max_frames = math.ceil(max_length / hop_length) + 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.positional = PositionalEncoding(embed_dim, max_frames, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.arch: Dict[str, object] = {
            "sample_rate": sample_rate,
            "max_length": max_length,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
        }

        if checkpoint and checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            state_dict = state.get("state_dict", state)
            checkpoint_classes = state.get("class_names")
            if checkpoint_classes:
                self.class_names = list(checkpoint_classes)
            self.load_state_dict(state_dict, strict=False)
        self.to(self.device)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms.to(self.device)
        lengths = lengths.to(self.device)

        spec = self.melspec(waveforms)
        spec = self.db_transform(spec.clamp(min=1e-10))
        spec = (spec - spec.mean(dim=(-2, -1), keepdim=True)) / (spec.std(dim=(-2, -1), keepdim=True) + 1e-5)
        emb = self.proj(spec).transpose(1, 2)  # (batch, time, embed)
        emb = self.positional(emb)

        frame_lengths = ((lengths + self.hop_length - 1) // self.hop_length).clamp(min=1)
        max_frames = emb.size(1)
        mask = torch.arange(max_frames, device=self.device).expand(len(frame_lengths), max_frames) >= frame_lengths.unsqueeze(1)

        encoded = self.transformer(emb, src_key_padding_mask=mask)
        mask_invert = (~mask).unsqueeze(-1).float()
        pooled = (encoded * mask_invert).sum(dim=1) / mask_invert.sum(dim=1).clamp(min=1e-6)
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits

    def predict(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> np.ndarray:
        """Convenience helper that mirrors the legacy interface."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(waveforms, lengths)
        return logits.squeeze(0).cpu().numpy()


# Backward-compatible alias used across the codebase.
AudioClassifier = AudioTransformerClassifier


__all__ = ["AudioTransformerClassifier", "AudioClassifier"]
