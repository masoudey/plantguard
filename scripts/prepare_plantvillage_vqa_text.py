#!/usr/bin/env python3
"""Convert PlantVillage VQA CSV into SQuAD-style FAQ splits."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/plantvillage_vqa"
CSV_PATH = RAW_DIR / "PlantVillageVQA.csv"
OUTPUT_DIR = ROOT / "data/processed/faq"


def _build_squad(split_rows: list[dict[str, str]]) -> dict:
    data = []
    for idx, row in enumerate(split_rows):
        question = (row.get("question") or "").strip()
        answer_text = (row.get("answer") or "").strip() or "No answer provided."
        context = f"Question: {question}\nAnswer: {answer_text}"
        answer_start = context.index(answer_text)
        qas = [
            {
                "id": f"{row.get('image_id','unknown')}::{idx}",
                "question": question,
                "answers": [
                    {
                        "text": answer_text,
                        "answer_start": answer_start,
                    }
                ],
                "is_impossible": False,
            }
        ]
        data.append(
            {
                "title": row.get("image_id") or f"entry_{idx}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": qas,
                    }
                ],
            }
        )
    return {"version": "plantvillage-vqa", "data": data}


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find PlantVillageVQA CSV at {CSV_PATH}")

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = (row.get("split") or "train").lower()
            rows_by_split[split].append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = rows_by_split.get("train") or []
    eval_rows = rows_by_split.get("validation") or rows_by_split.get("val") or []
    test_rows = rows_by_split.get("test") or []

    if not eval_rows and test_rows:
        eval_rows = test_rows

    datasets = {
        "train.json": train_rows,
        "validation.json": eval_rows,
    }

    for filename, rows in datasets.items():
        if not rows:
            continue
        squad = _build_squad(rows)
        out_path = OUTPUT_DIR / filename
        out_path.write_text(json.dumps(squad, indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} records to {out_path}")

    if test_rows and eval_rows is not test_rows:
        test_path = OUTPUT_DIR / "test.json"
        squad = _build_squad(test_rows)
        test_path.write_text(json.dumps(squad, indent=2), encoding="utf-8")
        print(f"Wrote {len(test_rows)} records to {test_path}")


if __name__ == "__main__":
    main()
