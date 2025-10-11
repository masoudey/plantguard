"""Audio waveform loading and preprocessing utilities for transformer models."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
import mimetypes

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


def _resolve_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        ext = Path(filename).suffix
        if ext:
            return ext
    if content_type:
        ext = mimetypes.guess_extension(content_type, strict=False)
        if ext:
            return ext
    return ".tmp"


def _decode_with_ffmpeg(audio_bytes: bytes, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Decode arbitrary audio bytes using ffmpeg CLI to ensure broad format support."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "1",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
    waveform_np, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
    waveform = torch.from_numpy(waveform_np)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    return waveform.contiguous(), sr


def load_waveform_from_bytes(
    audio_bytes: bytes,
    target_sr: int,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file from in-memory bytes and resample if needed."""
    buffer = io.BytesIO(audio_bytes)
    try:
        waveform_np, sr = sf.read(buffer, dtype="float32")
        waveform = torch.from_numpy(waveform_np)
    except RuntimeError:
        buffer.seek(0)
        suffix = _resolve_suffix(filename, content_type)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(buffer.read())
            tmp_path = tmp.name
        try:
            try:
                waveform, sr = torchaudio.load(tmp_path)
            except (RuntimeError, OSError, sf.LibsndfileError):
                waveform, sr = _decode_with_ffmpeg(audio_bytes, target_sr)
            else:
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=0)
                else:
                    waveform = waveform.squeeze(0)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

    waveform = waveform.contiguous()
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

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
