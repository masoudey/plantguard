"""Dataset wrapper for PlantVillage images."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset[Tuple[Image.Image, int]]):
    def __init__(self, root: str = "data/processed/plantvillage", split: str = "train", transform=None) -> None:
        self.root = Path(root) / split
        self.transform = transform
        self.samples = list(self.root.glob("*/*.jpg")) if self.root.exists() else []
        self._class_names = sorted({path.parent.name for path in self.samples}) or [
            "healthy",
            "powdery_mildew",
            "early_blight",
            "leaf_spot",
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = self.class_names.index(image_path.parent.name)
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)
