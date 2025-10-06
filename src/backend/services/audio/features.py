"""Audio feature extraction helpers leveraging torchaudio."""
from __future__ import annotations

import io
import os
import random
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T


def _to_mono_tensor(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 2:
        # shape (channels, samples)
        waveform = waveform.mean(dim=0)
    return waveform.float().contiguous()


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    waveform = torchaudio.functional.resample(waveform.unsqueeze(0), orig_sr, target_sr)
    return waveform.squeeze(0)


def load_waveform(path: str | os.PathLike[str], sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load a waveform from disk and resample to the requested rate."""
    data, orig_sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform
    else:
        waveform = waveform.transpose(0, 1)  # (channels, samples)
    waveform = _to_mono_tensor(waveform)
    waveform = _resample(waveform, orig_sr, sr)
    return waveform, sr


def load_waveform_from_bytes(audio_bytes: bytes, sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load waveform from raw audio bytes."""
    data, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform
    else:
        waveform = waveform.transpose(0, 1)
    waveform = _to_mono_tensor(waveform)
    waveform = _resample(waveform, orig_sr, sr)
    return waveform, sr


def mfcc_from_waveform(waveform: torch.Tensor, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    """Compute MFCC features directly from a waveform tensor."""
    transform = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc)
    mfcc = transform(waveform.unsqueeze(0))  # (1, n_mfcc, time)
    return mfcc.squeeze(0).numpy()


def mfcc_from_bytes(audio_bytes: bytes, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    waveform, _ = load_waveform_from_bytes(audio_bytes, sr=sr)
    return mfcc_from_waveform(waveform, sr=sr, n_mfcc=n_mfcc)


def mfcc_from_file(path: str | os.PathLike[str], sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    waveform, _ = load_waveform(path, sr=sr)
    return mfcc_from_waveform(waveform, sr=sr, n_mfcc=n_mfcc)


def apply_augmentations(
    waveform: torch.Tensor,
    sr: int,
    *,
    time_stretch_range: Tuple[float, float] = (0.9, 1.1),
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
    noise_scale: float = 0.004,
) -> torch.Tensor:
    """Apply random speed, pitch, and noise augmentations."""

    augmented = waveform.clone()

    if time_stretch_range[0] < 1.0 or time_stretch_range[1] > 1.0:
        if random.random() < 0.5:
            speed = random.uniform(*time_stretch_range)
            new_sr = max(int(sr * speed), 1)
            augmented = _resample(augmented, sr, new_sr)
            augmented = _resample(augmented, new_sr, sr)

    if pitch_shift_range[0] != 0.0 or pitch_shift_range[1] != 0.0:
        if random.random() < 0.5:
            steps = random.uniform(*pitch_shift_range)
            factor = 2.0 ** (steps / 12.0)
            new_sr = max(int(sr * factor), 1)
            augmented = _resample(augmented, sr, new_sr)
            augmented = _resample(augmented, new_sr, sr)

    if noise_scale > 0 and random.random() < 0.5:
        noise = noise_scale * torch.randn_like(augmented)
        augmented = augmented + noise

    return torch.clamp(augmented, -1.0, 1.0)


def pad_features(features: np.ndarray, max_length: int = 200) -> np.ndarray:
    """Pad or truncate MFCC features to a consistent length."""
    if features.shape[1] > max_length:
        return features[:, :max_length]
    if features.shape[1] < max_length:
        pad_width = max_length - features.shape[1]
        return np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    return features
