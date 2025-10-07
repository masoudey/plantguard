#!/usr/bin/env python3
"""Build a manifest of aligned multimodal samples for fusion training."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "plantguard/data/processed/audio"
VISION_DIR = ROOT / "plantguard/data/processed/plantvillage"
TTS_MANIFEST = ROOT / "plantguard/data/audio/Expanded_Plant_Symptom_TTS_Manifest.csv"
OUTPUT_DIR = ROOT / "plantguard/data/processed/fusion"

# Mapping between audio label and the corresponding PlantVillage class directory.
LABEL_TO_VISION = {
    "late_blight": "Tomato_Late_blight",
    "powdery_mildew": "Tomato_Powdery_mildew",
    "spider_mites": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "mosaic_virus": "Tomato__Tomato_mosaic_virus",
    "leaf_spot": "Tomato_Septoria_leaf_spot",
    "tylcv": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "bacterial_blight": "Tomato_Bacterial_spot",
}

SUPPORTED_SPLITS = ("train", "val")
RNG = random.Random(42)


def _load_tts_manifest() -> dict[str, str]:
    if not TTS_MANIFEST.exists():
        raise FileNotFoundError(f"Missing TTS manifest at {TTS_MANIFEST}")
    mapping: dict[str, str] = {}
    with TTS_MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel_path = row.get("rel_audio_path")
            text = row.get("text")
            if not rel_path or not text:
                continue
            mapping[rel_path.strip()] = text.strip()
    return mapping


def _collect_images(split: str) -> dict[str, list[Path]]:
    images: dict[str, list[Path]] = {}
    split_dir = VISION_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing vision split directory: {split_dir}")
    for audio_label, vision_class in LABEL_TO_VISION.items():
        class_dir = split_dir / vision_class
        if not class_dir.exists():
            continue
        paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))
        if paths:
            images[audio_label] = paths
    return images


def _build_rows(split: str, tts_text: dict[str, str]) -> list[dict[str, str]]:
    manifest_path = AUDIO_DIR / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing audio manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        samples = json.load(handle)

    available_images = _collect_images(split)
    rows: list[dict[str, str]] = []
    for sample in samples:
        label = sample.get("label")
        rel_audio_path = sample.get("file")
        if not label or not rel_audio_path:
            continue
        if label not in LABEL_TO_VISION:
            continue
        if label not in available_images:
            continue
        text = tts_text.get(rel_audio_path)
        if not text:
            continue
        image_path = RNG.choice(available_images[label])
        audio_path = AUDIO_DIR / rel_audio_path
        if not audio_path.exists():
            continue
        rows.append(
            {
                "split": split,
                "label": label,
                "image_path": str(image_path.relative_to(ROOT)),
                "audio_path": str(audio_path.relative_to(ROOT)),
                "text": text,
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tts_text = _load_tts_manifest()

    for split in SUPPORTED_SPLITS:
        rows = _build_rows(split, tts_text)
        if not rows:
            raise RuntimeError(f"No rows generated for split '{split}'.")
        out_path = OUTPUT_DIR / f"fusion_manifest_{split}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "label", "image_path", "audio_path", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
