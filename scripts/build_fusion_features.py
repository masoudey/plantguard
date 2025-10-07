#!/usr/bin/env python3
"""Generate fused modality feature packs from the curated manifest."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
plantguard_root = ROOT / "plantguard"
if str(plantguard_root) not in sys.path:
    sys.path.insert(0, str(plantguard_root))

from src.backend.services.preprocess import transform_image_bytes  # noqa: E402
from src.backend.services.vision import VisionClassifier  # noqa: E402
from src.backend.services.audio import features as audio_features  # noqa: E402
from src.backend.services.audio.model import AudioTransformerClassifier  # noqa: E402

FUSION_DIR = ROOT / "plantguard/data/processed/fusion"
VISION_CKPT = ROOT / "plantguard/models/vision/plantguard_resnet50.pt"
AUDIO_CKPT = ROOT / "plantguard/models/audio/plantguard_transformer.pt"
MANIFEST_TEMPLATE = "fusion_manifest_{split}.csv"
OUTPUT_TEMPLATE = "{split}.pt"
MAX_VOCAB_SIZE = 1024


def _load_manifest(split: str) -> list[dict[str, str]]:
    manifest_path = FUSION_DIR / MANIFEST_TEMPLATE.format(split=split)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Manifest {manifest_path} is empty")
    return rows


def _build_label_mapping(rows: Iterable[dict[str, str]]) -> list[str]:
    labels = sorted({row["label"] for row in rows})
    if not labels:
        raise RuntimeError("No labels discovered across manifests")
    return labels


def _build_vocabulary(rows: Iterable[dict[str, str]], max_features: int = MAX_VOCAB_SIZE) -> list[str]:
    token_counts: Counter[str] = Counter()
    for row in rows:
        tokens = _tokenize(row["text"])
        token_counts.update(tokens)
    if not token_counts:
        raise RuntimeError("No tokens found while building vocabulary")
    most_common = token_counts.most_common(max_features)
    return [token for token, _ in most_common]


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'_-]*")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _encode_texts(rows: list[dict[str, str]], vocabulary: list[str]) -> torch.Tensor:
    vocab_index = {token: idx for idx, token in enumerate(vocabulary)}
    matrix = torch.zeros((len(rows), len(vocabulary)), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        tokens = _tokenize(row["text"])
        if not tokens:
            continue
        counts = Counter(tokens)
        for token, count in counts.items():
            token_idx = vocab_index.get(token)
            if token_idx is not None:
                matrix[row_idx, token_idx] = float(count)
    return matrix


def _init_models(device: torch.device) -> tuple[VisionClassifier, AudioTransformerClassifier]:
    if not VISION_CKPT.exists():
        raise FileNotFoundError(f"Vision checkpoint missing: {VISION_CKPT}")
    if not AUDIO_CKPT.exists():
        raise FileNotFoundError(f"Audio checkpoint missing: {AUDIO_CKPT}")

    vision_model = VisionClassifier(device=device, checkpoint=VISION_CKPT)

    audio_state = torch.load(AUDIO_CKPT, map_location=device)
    class_names = audio_state.get("class_names")
    num_classes = len(class_names) if class_names else 0
    if num_classes == 0:
        classifier_weight = audio_state["state_dict"].get("classifier.1.weight")
        if classifier_weight is None:
            raise RuntimeError("Cannot determine number of audio classes from checkpoint")
        num_classes = classifier_weight.shape[0]
    arch = audio_state.get("architecture", {})
    audio_model = AudioTransformerClassifier(
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        checkpoint=AUDIO_CKPT,
        sample_rate=int(arch.get("sample_rate", 16000)),
        max_length=int(arch.get("max_length", 96000)),
        n_mels=int(arch.get("n_mels", 64)),
        n_fft=int(arch.get("n_fft", 1024)),
        hop_length=int(arch.get("hop_length", 256)),
        embed_dim=int(arch.get("embed_dim", 256)),
        num_heads=int(arch.get("num_heads", 4)),
        num_layers=int(arch.get("num_layers", 4)),
        ffn_dim=int(arch.get("ffn_dim", 512)),
        dropout=float(arch.get("dropout", 0.2)),
    )
    return vision_model, audio_model


def _extract_image_embedding(model: VisionClassifier, path: Path) -> torch.Tensor:
    image_bytes = path.read_bytes()
    tensor = transform_image_bytes(image_bytes)
    with torch.no_grad():
        logits = model.forward(tensor)
    return logits.squeeze(0).cpu().float()


def _extract_audio_embedding(
    model: AudioTransformerClassifier,
    path: Path,
    sample_rate: int,
    max_length: int,
) -> torch.Tensor:
    waveform, _ = audio_features.load_waveform(path, target_sr=sample_rate)
    padded, original_length = audio_features.pad_or_trim_waveform(waveform, max_length)
    waveforms = padded.unsqueeze(0)
    lengths = torch.tensor([original_length], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(waveforms, lengths)
    return logits.squeeze(0).cpu().float()


def _save_feature_pack(
    split: str,
    rows: list[dict[str, str]],
    class_names: list[str],
    vision_model: VisionClassifier,
    audio_model: AudioTransformerClassifier,
    vocabulary: list[str],
) -> None:
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    text_embeddings = _encode_texts(rows, vocabulary)
    image_embeddings: list[torch.Tensor] = []
    audio_embeddings: list[torch.Tensor] = []
    labels: list[int] = []

    audio_arch = getattr(audio_model, "arch", None)
    if audio_arch:
        sample_rate = int(audio_arch.get("sample_rate", 16000))
        max_length = int(audio_arch.get("max_length", 96000))
    else:
        sample_rate = getattr(audio_model, "sample_rate", 16000)
        max_length = getattr(audio_model, "max_length", 96000)

    for idx, row in enumerate(rows):
        image_path = ROOT / row["image_path"]
        audio_path = ROOT / row["audio_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        image_embeddings.append(_extract_image_embedding(vision_model, image_path))
        audio_embeddings.append(_extract_audio_embedding(audio_model, audio_path, sample_rate, max_length))
        labels.append(label_to_index[row["label"]])

        if idx % 50 == 0:
            print(f"Processed {idx + 1}/{len(rows)} samples for split '{split}'")

    feature_pack = {
        "image": torch.stack(image_embeddings),
        "audio": torch.stack(audio_embeddings),
        "text": text_embeddings,
        "labels": torch.tensor(labels, dtype=torch.long),
        "class_names": class_names,
        "text_vocabulary": vocabulary,
        "manifest": rows,
    }

    output_path = FUSION_DIR / OUTPUT_TEMPLATE.format(split=split)
    torch.save(feature_pack, output_path)
    print(f"Saved feature pack to {output_path.relative_to(ROOT)}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_rows = _load_manifest("train")
    val_rows = _load_manifest("val")
    class_names = _build_label_mapping(train_rows + val_rows)
    vocabulary = _build_vocabulary(train_rows + val_rows)

    vision_model, audio_model = _init_models(device)

    _save_feature_pack("train", train_rows, class_names, vision_model, audio_model, vocabulary)
    _save_feature_pack("val", val_rows, class_names, vision_model, audio_model, vocabulary)

    summary = {
        "class_names": class_names,
        "text_vocabulary_size": len(vocabulary),
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
        },
    }
    summary_path = FUSION_DIR / "fusion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
