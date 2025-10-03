"""Audio feature extraction helpers."""
from __future__ import annotations

import io
import os
from typing import Tuple

import librosa
import numpy as np


def mfcc_from_bytes(audio_bytes: bytes, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    """Convert raw audio bytes into MFCC feature matrix."""
    waveform, orig_sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    if orig_sr != sr:
        waveform = librosa.resample(waveform, orig_sr, sr)
    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)


def mfcc_from_file(path: str | os.PathLike[str], sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    """Load audio from ``path`` and compute MFCC features."""
    waveform, orig_sr = librosa.load(path, sr=None)
    if orig_sr != sr:
        waveform = librosa.resample(waveform, orig_sr, sr)
    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)


def pad_features(features: np.ndarray, max_length: int = 200) -> np.ndarray:
    """Pad or truncate MFCC features to a consistent length."""
    if features.shape[1] > max_length:
        return features[:, :max_length]
    if features.shape[1] < max_length:
        pad_width = max_length - features.shape[1]
        return np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    return features
