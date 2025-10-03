"""Train the PlantGuard speech classifier using MFCC features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split

from src.backend.services.audio.dataset import SymptomSpeechDataset
from src.backend.services.audio.model import AudioClassifier
from src.backend.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the audio modality classifier")
    parser.add_argument("--data-dir", default="data/processed/audio", help="Directory containing <split>.json manifests and audio files")
    parser.add_argument("--output", default="models/audio/plantguard_cnn_lstm.pt", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data used for validation if val manifest missing")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--pad-to", type=int, default=200, help="Number of frames to pad MFCC features to")
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, list[str]]:
    base_train = SymptomSpeechDataset(root=args.data_dir, split="train", pad_to=args.pad_to)
    val_manifest = Path(args.data_dir) / "val.json"

    if val_manifest.exists():
        train_dataset = base_train
        val_dataset = SymptomSpeechDataset(
            root=args.data_dir,
            split="val",
            pad_to=args.pad_to,
            label_to_index=base_train.label_to_index,
        )
    else:
        val_len = max(1, int(len(base_train) * args.val_split))
        train_len = len(base_train) - val_len
        train_dataset, val_dataset = random_split(base_train, [train_len, val_len])

    class_names = [name for name, _ in sorted(base_train.label_to_index.items(), key=lambda x: x[1])]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names


def train() -> None:
    args = parse_args()
    log = logger.get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device", device=str(device))

    train_loader, val_loader, class_names = build_dataloaders(args)
    num_classes = len(class_names)

    model = AudioClassifier(device=device, num_classes=num_classes, class_names=class_names)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        log.info(
            "Epoch summary",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "pad_to": args.pad_to,
            }, output_path)
            log.info("Saved new best audio checkpoint", path=str(output_path), accuracy=round(best_acc, 4))

    log.info("Audio training complete", best_accuracy=round(best_acc, 4))


@torch.no_grad()
def evaluate(model: AudioClassifier, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += features.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


if __name__ == "__main__":
    train()
