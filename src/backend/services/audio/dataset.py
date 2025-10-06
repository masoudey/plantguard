"""Dataset wrapper for symptom speech descriptions stored on disk."""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import features


@dataclass
class AudioSample:
    file_path: Path
    label: int


class SymptomSpeechDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Loads MFCC features for speech clips defined in a manifest file.

    The dataset expects a JSON manifest at ``root/{split}.json`` containing records
    with ``{"file": "relative/path.wav", "label": "class_name"}``. Labels are mapped
    to indices alphabetically so the same mapping can be reused at inference time.
    """

    def __init__(
        self,
        root: str = "data/processed/audio",
        split: str = "train",
        pad_to: int = 200,
        label_to_index: Dict[str, int] | None = None,
        *,
        augment: bool = False,
        augment_prob: float = 0.4,
        n_mfcc: int = 40,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.pad_to = pad_to
        self.augment = augment
        self.augment_prob = augment_prob
        self.n_mfcc = n_mfcc

        manifest = self.root / f"{split}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest '{manifest}' was not found. Generate it before training.")

        entries: List[Dict[str, str]] = json.loads(manifest.read_text())
        if not entries:
            raise ValueError(f"Manifest '{manifest}' is empty.")

        if label_to_index is None:
            labels = sorted({item["label"] for item in entries})
            self.label_to_index = {name: idx for idx, name in enumerate(labels)}
        else:
            self.label_to_index = label_to_index
        self._index_to_label = [name for name, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]
        self.samples = [
            AudioSample(file_path=self.root / item["file"], label=self.label_to_index[item["label"]])
            for item in entries
        ]
        self._label_counts = Counter([item["label"] for item in entries])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        if not sample.file_path.exists():
            raise FileNotFoundError(f"Audio file '{sample.file_path}' not found")

        waveform, sr = features.load_waveform(sample.file_path)
        if self.augment and random.random() < self.augment_prob:
            waveform = features.apply_augmentations(waveform, sr)

        mfcc = features.mfcc_from_waveform(waveform, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = features.pad_features(mfcc, max_length=self.pad_to)
        tensor = torch.from_numpy(np.expand_dims(mfcc, axis=0)).float()
        return tensor, sample.label

    @property
    def class_names(self) -> List[str]:
        return list(self._index_to_label)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    @property
    def label_counts(self) -> Dict[str, int]:
        return dict(self._label_counts)

    def sample_weights(self) -> List[float]:
        counts = self.label_counts
        weights = []
        for sample in self.samples:
            label_name = self._index_to_label[sample.label]
            weights.append(1.0 / counts[label_name])
        return weights
