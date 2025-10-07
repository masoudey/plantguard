"""Audio waveform loading and preprocessing utilities for transformer models."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import soundfile as sf
import torch
import torchaudio


def load_waveform(path: str | Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load an audio file and resample/convert it to mono."""
    waveform, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    return waveform.contiguous(), sr


def load_waveform_from_bytes(audio_bytes: bytes, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load an audio file from in-memory bytes and resample if needed."""
    waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    return waveform.contiguous(), sr


def pad_or_trim_waveform(waveform: torch.Tensor, max_length: int) -> Tuple[torch.Tensor, int]:
    """Pad or trim the waveform to ``max_length`` samples.

    Returns the padded/trimmed waveform and the original unclipped length.
    """
    original_length = min(waveform.numel(), max_length)
    if waveform.numel() < max_length:
        waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.numel()))
    else:
        waveform = waveform[:max_length]
    return waveform.contiguous(), original_length


def apply_waveform_augmentation(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Apply lightweight augmentations (gain, noise, time shift)."""
    gain = torch.empty(1).uniform_(0.9, 1.1).item()
    waveform = waveform * gain

    if torch.rand(1).item() < 0.5:
        shift = int(torch.empty(1).uniform_(-0.02, 0.02).item() * sr)
        waveform = torch.roll(waveform, shifts=shift)

    if torch.rand(1).item() < 0.5:
        noise_scale = torch.empty(1).uniform_(0.001, 0.003).item()
        waveform = waveform + noise_scale * torch.randn_like(waveform)

    return waveform.clamp_(-1.0, 1.0)
