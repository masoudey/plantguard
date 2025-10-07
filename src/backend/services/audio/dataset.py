"""Waveform dataset for audio transformer training."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from . import features


@dataclass
class AudioSample:
    file_path: Path
    label: int


class SymptomSpeechDataset(Dataset):
    """Loads waveforms and metadata for symptom speech classification."""

    def __init__(
        self,
        root: str = "data/processed/audio",
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 6.0,
        augment: bool = False,
        augment_prob: float = 0.2,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(sample_rate * max_duration)
        self.augment = augment and split == "train"
        self.augment_prob = augment_prob

        manifest = self.root / f"{split}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest '{manifest}' was not found. Generate it before training.")

        entries: List[Dict[str, str]] = json.loads(manifest.read_text())
        if not entries:
            raise ValueError(f"Manifest '{manifest}' is empty.")

        labels = sorted({item["label"] for item in entries})
        self.label_to_index = {name: idx for idx, name in enumerate(labels)}
        self.index_to_label = [name for name, _ in sorted(self.label_to_index.items(), key=lambda x: x[1])]
        self.samples = [
            AudioSample(file_path=self.root / item["file"], label=self.label_to_index[item["label"]])
            for item in entries
        ]

        counts = {name: 0 for name in labels}
        for sample in self.samples:
            label_name = self.index_to_label[sample.label]
            counts[label_name] += 1
        self._label_counts = counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        if not sample.file_path.exists():
            raise FileNotFoundError(f"Audio file '{sample.file_path}' not found")

        waveform, _ = features.load_waveform(sample.file_path, self.sample_rate)
        waveform, original_length = features.pad_or_trim_waveform(waveform, self.max_length)

        if self.augment and random.random() < self.augment_prob:
            waveform = features.apply_waveform_augmentation(waveform, self.sample_rate)

        return waveform, original_length, sample.label

    @property
    def class_names(self) -> List[str]:
        return list(self.index_to_label)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    @property
    def label_counts(self) -> Dict[str, int]:
        return dict(self._label_counts)

    def sample_weights(self) -> List[float]:
        weights = []
        for sample in self.samples:
            label_name = self.index_to_label[sample.label]
            weights.append(1.0 / self._label_counts[label_name])
        return weights
