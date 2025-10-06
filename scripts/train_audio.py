"""Train the PlantGuard speech classifier using MFCC features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None  # type: ignore

from src.backend.services.audio.dataset import SymptomSpeechDataset
from src.backend.services.audio.model import AudioClassifier
from src.backend.utils import logger


def _parse_conv_channels(value: str) -> List[int]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--conv-channels must contain at least one integer")
    try:
        return [int(chunk) for chunk in parts]
    except ValueError as exc:  # pragma: no cover - input validation
        raise argparse.ArgumentTypeError("--conv-channels expects a comma-separated list of integers") from exc


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
    parser.add_argument("--n-mfcc", type=int, default=40, help="Number of MFCC coefficients to compute")
    parser.add_argument("--augment", action="store_true", help="Enable on-the-fly waveform augmentation for training data")
    parser.add_argument("--augment-prob", type=float, default=0.5, help="Probability of applying augmentation to an example")
    parser.add_argument("--class-weighting", action="store_true", help="Apply class-balanced loss weights derived from training manifest")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Enable WeightedRandomSampler to rebalance batches")
    parser.add_argument("--conv-channels", default="32,64", help="Comma-separated convolution channel sizes (e.g., 32,64,128)")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for LSTM layers")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional LSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability applied after LSTM")
    parser.add_argument("--log-batches", type=int, default=0, help="Log every N training batches (set 0 to disable)")
    parser.add_argument("--progress", action="store_true", help="Display tqdm progress bars during training")
    parser.add_argument("--log-dir", default=None, help="Optional TensorBoard log directory")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON file to write best validation metrics")
    return parser.parse_args()


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, List[str], Optional[SymptomSpeechDataset]]:
    val_manifest = Path(args.data_dir) / "val.json"

    weighted_sampler: Optional[WeightedRandomSampler] = None
    train_manifest_dataset: Optional[SymptomSpeechDataset] = None

    if val_manifest.exists():
        train_dataset = SymptomSpeechDataset(
            root=args.data_dir,
            split="train",
            pad_to=args.pad_to,
            augment=args.augment,
            augment_prob=args.augment_prob,
            n_mfcc=args.n_mfcc,
        )
        val_dataset = SymptomSpeechDataset(
            root=args.data_dir,
            split="val",
            pad_to=args.pad_to,
            label_to_index=train_dataset.label_to_index,
            augment=False,
            n_mfcc=args.n_mfcc,
        )
        train_manifest_dataset = train_dataset

        if args.use_weighted_sampler:
            sample_weights = train_dataset.sample_weights()
            weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        base_dataset = SymptomSpeechDataset(
            root=args.data_dir,
            split="train",
            pad_to=args.pad_to,
            augment=False,
            n_mfcc=args.n_mfcc,
        )
        val_len = max(1, int(len(base_dataset) * args.val_split))
        train_len = len(base_dataset) - val_len
        train_dataset, val_dataset = random_split(base_dataset, [train_len, val_len])

    if isinstance(train_dataset, Subset):
        base_dataset = train_dataset.dataset  # type: ignore[attr-defined]
        class_names = base_dataset.class_names  # type: ignore[has-type]
    else:
        class_names = train_dataset.class_names

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=weighted_sampler is None,
        sampler=weighted_sampler,
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

    return train_loader, val_loader, class_names, train_manifest_dataset


def train() -> None:
    args = parse_args()
    args.conv_channels = _parse_conv_channels(args.conv_channels)
    log = logger.get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device", device=str(device))

    if args.progress and tqdm is None:
        log.warning("tqdm is not installed; install tqdm or omit --progress to disable progress bars")

    train_loader, val_loader, class_names, train_manifest_dataset = build_dataloaders(args)
    num_classes = len(class_names)

    arch = {
        "conv_channels": tuple(args.conv_channels),
        "lstm_hidden": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "bidirectional": not args.no_bidirectional,
        "dropout": args.dropout,
        "n_mfcc": args.n_mfcc,
    }

    model = AudioClassifier(
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        conv_channels=arch["conv_channels"],
        lstm_hidden=args.hidden_size,
        lstm_layers=args.lstm_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
        n_mfcc=args.n_mfcc,
    )

    class_weights_tensor = None
    if args.class_weighting and train_manifest_dataset is not None:
        counts = train_manifest_dataset.label_counts
        total = sum(counts.values())
        weights = [total / (counts[name] * len(counts)) for name in class_names]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    writer: SummaryWriter | None = None
    if args.log_dir and SummaryWriter is not None:
        writer = SummaryWriter(args.log_dir)

    best_macro_f1 = -1.0
    best_metrics: Dict[str, object] = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        if args.progress and tqdm is not None:
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
        else:
            batch_iterator = train_loader

        for batch_idx, (features, labels) in enumerate(batch_iterator, 1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)

            if args.progress and tqdm is not None:
                batch_iterator.set_postfix(loss=float(loss.item()))
            elif args.log_batches and batch_idx % args.log_batches == 0:
                total_batches = len(train_loader)
                log.info(
                    "Batch progress",
                    epoch=epoch,
                    batch=batch_idx,
                    total_batches=total_batches,
                    loss=round(float(loss.item()), 4),
                )

        if args.progress and tqdm is not None:
            batch_iterator.close()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, class_names)
        val_acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        scheduler.step(val_loss)

        log.info(
            "Epoch summary",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
            macro_f1=round(macro_f1, 4),
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/val_accuracy", val_acc, epoch)
            writer.add_scalar("Metrics/val_macro_f1", macro_f1, epoch)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_metrics = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "pad_to": args.pad_to,
                    "metrics": metrics,
                    "arch": arch,
                },
                output_path,
            )
            log.info(
                "Saved new best audio checkpoint",
                path=str(output_path),
                accuracy=round(val_acc, 4),
                macro_f1=round(macro_f1, 4),
            )

    if writer:
        writer.close()

    if args.metrics_output and best_metrics:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as fh:
            json.dump(best_metrics, fh, indent=2)
        log.info("Wrote best metrics", path=str(metrics_path))

    log.info(
        "Audio training complete",
        best_macro_f1=round(best_macro_f1, 4) if best_macro_f1 >= 0 else None,
    )


@torch.no_grad()
def evaluate(
    model: AudioClassifier,
    loader: DataLoader,
    criterion,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, Dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += features.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy),
        "macro_f1": 0.0,
        "per_class": {},
    }

    if total_samples:
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        per_class: Dict[str, Dict[str, float]] = {}
        f1_total = 0.0
        for idx, name in enumerate(class_names):
            tp = float(np.sum((preds_arr == idx) & (labels_arr == idx)))
            fp = float(np.sum((preds_arr == idx) & (labels_arr != idx)))
            fn = float(np.sum((preds_arr != idx) & (labels_arr == idx)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            support = int(np.sum(labels_arr == idx))
            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
            f1_total += f1
        metrics["per_class"] = per_class
        metrics["macro_f1"] = float(f1_total / max(len(class_names), 1))

    return avg_loss, metrics


if __name__ == "__main__":
    train()
