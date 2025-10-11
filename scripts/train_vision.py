"""Train the PlantGuard vision classifier on PlantVillage style data."""
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
from torchvision import models, transforms

from src.backend.services.datasets.plant_village import PlantVillageDataset
from src.backend.utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ResNet50 on plant disease images")
    parser.add_argument(
        "--data-dir",
        default="data/vision",
        help="Root directory containing PlantVillage data (directories of images or parquet files)",
    )
    parser.add_argument("--output", default="models/vision/plantguard_resnet50.pt", help="Where to store the trained weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data to use for validation if no dedicated split exists")
    parser.add_argument("--num-workers", type=int, default=4, help="PyTorch DataLoader workers")
    parser.add_argument("--freeze-epochs", type=int, default=1, help="Epochs to keep the backbone frozen at the start")
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    augment = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PlantVillageDataset(root=args.data_dir, split="train", transform=augment)
    val_root = Path(args.data_dir) / "val"

    if val_root.exists():
        val_dataset = PlantVillageDataset(root=args.data_dir, split="val", transform=eval_transform)
    else:
        if len(train_dataset) == 0:
            raise FileNotFoundError(f"No training images found under {train_dataset.root}")
        val_len = max(1, int(len(train_dataset) * args.val_split))
        train_len = len(train_dataset) - val_len
        train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])
        train_dataset.dataset.transform = augment  # type: ignore[attr-defined]
        val_dataset.dataset.transform = eval_transform  # type: ignore[attr-defined]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def train() -> None:
    args = parse_args()
    log = logger.get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device", device=str(device))

    train_loader, val_loader = build_dataloaders(args)
    class_names = _extract_class_names(train_loader.dataset)
    num_classes = len(class_names)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        if epoch <= args.freeze_epochs:
            for name, param in model.named_parameters():
                if name.startswith("layer"):
                    param.requires_grad = False
        elif epoch == args.freeze_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True

        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log.info(
            "Epoch summary",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "class_names": class_names}, output_path)
            log.info("Saved new best checkpoint", path=str(output_path), accuracy=round(best_acc, 4))

    log.info("Training complete", best_accuracy=round(best_acc, 4))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def _extract_class_names(dataset) -> list[str]:  # type: ignore[typing-arg]
    if hasattr(dataset, "class_names"):
        return dataset.class_names  # type: ignore[return-value]
    if hasattr(dataset, "dataset"):
        return _extract_class_names(dataset.dataset)  # type: ignore[attr-defined]
    raise AttributeError("Unable to resolve class names from dataset")


if __name__ == "__main__":
    train()
