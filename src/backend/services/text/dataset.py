"""Text dataset for plant-care FAQs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset


@dataclass
class PlantFaqDataset:
    path: str = "data/processed/faq"

    def load(self) -> DatasetDict:
        root = Path(self.path)
        if root.exists():
            data_files = {}
            for split in ("train", "validation", "test"):
                file = root / f"{split}.json"
                if file.exists():
                    data_files[split] = str(file)
            if not data_files:
                raise FileNotFoundError(
                    f"No JSON files found in {root}. Expected train.json, validation.json, or test.json"
                )
            return load_dataset("json", data_files=data_files)
        return load_dataset("squad")
