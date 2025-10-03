"""Organise the PlantVillage dataset into train/val/test splits for PlantGuard."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from sklearn.model_selection import train_test_split

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


def iter_class_images(root: Path) -> Iterable[Tuple[str, Path]]:
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_dir.name
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in SUPPORTED_EXTS:
                yield label, image_path


def copy_files(pairs: List[Tuple[str, Path]], destination: Path) -> None:
    for label, src_path in pairs:
        dest_dir = destination / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        if dest_path.exists():
            # Avoid collisions by prefixing with random token
            dest_path = dest_dir / f"{random.randint(0, 99999)}_{src_path.name}"
        shutil.copy2(src_path, dest_path)


def organise_dataset(raw_dir: Path, processed_dir: Path, val_size: float, test_size: float, seed: int) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist. Download and extract PlantVillage first.")

    all_samples = list(iter_class_images(raw_dir))
    if not all_samples:
        raise RuntimeError(
            "No images discovered. Ensure raw_dir contains class sub-folders such as 'Potato___Early_blight/'."
        )

    labels = [label for label, _ in all_samples]
    train_samples, temp_samples = train_test_split(
        all_samples,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )

    temp_labels = [label for label, _ in temp_samples]
    val_ratio = val_size / (val_size + test_size)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=1 - val_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    copy_files(train_samples, processed_dir / "train")
    copy_files(val_samples, processed_dir / "val")
    copy_files(test_samples, processed_dir / "test")

    print(f"Processed dataset stored in {processed_dir}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage dataset for PlantGuard training")
    parser.add_argument("--raw-dir", default="data/raw/plantvillage/extracted", help="Directory with class folders from PlantVillage")
    parser.add_argument("--output", default="data/processed/plantvillage", help="Destination for organised dataset")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    organise_dataset(raw_dir, output_dir, args.val_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
