"""Dataset wrapper for PlantVillage images."""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset[Tuple[Image.Image, int]]):
    def __init__(self, root: str = "data/processed/plantvillage", split: str = "train", transform=None) -> None:
        self.base_root = Path(root)
        self.split = split
        self.transform = transform
        self._class_names: List[str] = []
        self._label_lookup: dict[str, int] = {}
        self._parquet_mode = False

        parquet_files = self._discover_parquet_files()
        if parquet_files:
            self._load_parquet(parquet_files)
        else:
            self.root = self.base_root / split
            self.samples = list(self.root.glob("*/*.jpg")) if self.root.exists() else []
            if not self.samples:
                raise FileNotFoundError(
                    f"No images found under {self.root}. Provide PlantVillage images or parquet files in {self.base_root}."
                )
            self._class_names = sorted({path.parent.name for path in self.samples})
            self._label_lookup = {name: idx for idx, name in enumerate(self._class_names)}

    def _discover_parquet_files(self) -> list[Path]:
        candidates: list[Path] = []
        if self.base_root.is_file() and self.base_root.suffix == ".parquet":
            candidates.append(self.base_root)
        elif self.base_root.is_dir():
            pattern = f"{self.split}*.parquet"
            candidates.extend(sorted(self.base_root.glob(pattern)))
        return candidates

    def _load_parquet(self, files: list[Path]) -> None:
        frames = [pd.read_parquet(path) for path in files]
        df = pd.concat(frames, ignore_index=True)
        if "split" in df.columns:
            df = df[df["split"].str.lower() == self.split.lower()].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for split '{self.split}' in parquet files under {self.base_root}")

        if "caption" not in df.columns:
            raise ValueError("Expected 'caption' column with class labels in parquet dataset")

        self._parquet_mode = True
        self.df = df
        self._class_names = sorted(df["caption"].unique().tolist())
        self._label_lookup = {name: idx for idx, name in enumerate(self._class_names)}

    def __len__(self) -> int:
        if self._parquet_mode:
            return len(self.df)
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        if self._parquet_mode:
            row = self.df.iloc[index]
            image_info = row["image"]
            if isinstance(image_info, dict) and "bytes" in image_info:
                image_bytes = image_info["bytes"]
            else:
                raise ValueError("Expected parquet 'image' column to contain byte dicts")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            label_name = row["caption"]
            label = self._label_lookup[label_name]
        else:
            image_path = self.samples[index]
            image = Image.open(image_path).convert("RGB")
            label = self._label_lookup[image_path.parent.name]

        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)
