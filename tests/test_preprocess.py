"""Unit tests for preprocessing utilities."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch

from plantguard.src.backend.services import preprocess


def test_transform_image_bytes(tmp_path):
    from PIL import Image

    path = tmp_path / "image.jpg"
    Image.new("RGB", (256, 256), color="green").save(path)
    tensor = preprocess.transform_image_bytes(path.read_bytes())
    assert tensor.shape[-2:] == (224, 224)


def test_transform_audio_bytes(tmp_path):
    waveform = np.zeros(16000, dtype=np.float32)
    path = tmp_path / "audio.wav"
    sf.write(path, waveform, samplerate=16000)
    tensor = preprocess.transform_audio_bytes(path.read_bytes())
    assert tensor.ndim == 4
    assert tensor.dtype == torch.float32
