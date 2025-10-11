"""Text dataset for plant-care FAQs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset


def _flatten_squad(ds: DatasetDict) -> DatasetDict:
    def _extract(example: dict[str, object]) -> dict[str, object]:
        data = example.get("data") or {}
        paragraphs = data.get("paragraphs") if isinstance(data, dict) else None
        if paragraphs:
            paragraph = paragraphs[0]
            context = paragraph.get("context", "")
            qas = paragraph.get("qas") or []
            qa = qas[0] if qas else {}
        else:
            context = ""
            qa = {}

        answers = qa.get("answers") or []
        texts = []
        starts = []
        for ans in answers:
            if isinstance(ans, dict):
                texts.append(ans.get("text", ""))
                starts.append(ans.get("answer_start", 0))

        return {
            "id": qa.get("id", data.get("title")) if isinstance(data, dict) else qa.get("id"),
            "question": qa.get("question", ""),
            "context": context,
            "answers": {
                "text": texts,
                "answer_start": starts,
            },
        }

    column_names = None
    for split in ds.keys():
        if "data" in ds[split].column_names:
            column_names = ds[split].column_names
            break

    if column_names and "data" in column_names:
        ds = ds.map(_extract, remove_columns=column_names)
    return ds


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
            ds = load_dataset("json", data_files=data_files)
            print(f"Loaded dataset from {root} with splits: {list(ds.keys())}")
            return _flatten_squad(ds)
        ds = load_dataset("squad")
        print("your dataset wasnt found, Loaded SQuAD dataset as fallback.")
        return ds
