"""Train the multimodal fusion head on pre-computed embeddings."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src.backend.services.fusion_model import MultimodalFusionHead
from src.backend.utils import logger


@dataclass
class FeaturePack:
    image: torch.Tensor
    text: torch.Tensor
    audio: Optional[torch.Tensor]
    labels: torch.Tensor
    class_names: list[str]


class FusionTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]):
    def __init__(self, pack: FeaturePack) -> None:
        self.pack = pack

    def __len__(self) -> int:
        return self.pack.labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audio = self.pack.audio[index] if self.pack.audio is not None else None
        return (
            self.pack.image[index],
            self.pack.text[index],
            audio,
            self.pack.labels[index],
        )


def collate_batch(batch):
    images, texts, audios, labels = zip(*batch)
    images = torch.stack(images)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    if audios[0] is None:
        audio_tensor = None
    else:
        audio_tensor = torch.stack([a for a in audios if a is not None])
    return images, texts, audio_tensor, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PlantGuard fusion head")
    parser.add_argument("--train", required=True, help="Path to a torch.save() feature pack (train)")
    parser.add_argument("--val", required=True, help="Path to a torch.save() feature pack (validation)")
    parser.add_argument("--output", default="models/fusion/plantguard_mlp.pt", help="Where to store the trained fusion head")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def load_feature_pack(path: Path) -> FeaturePack:
    payload = torch.load(path)
    image = _ensure_tensor(payload["image"], dtype=torch.float32)
    text = _ensure_tensor(payload["text"], dtype=torch.float32)
    audio = payload.get("audio")
    audio_tensor = _ensure_tensor(audio, dtype=torch.float32) if audio is not None else None
    labels = _ensure_tensor(payload["labels"], dtype=torch.long)
    class_names = payload.get("class_names") or [f"class_{i}" for i in range(labels.max().item() + 1)]
    return FeaturePack(image=image, text=text, audio=audio_tensor, labels=labels, class_names=class_names)


def _ensure_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype)


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = logger.get_logger(__name__)
    log.info("Using device", device=str(device))

    train_pack = load_feature_pack(Path(args.train))
    val_pack = load_feature_pack(Path(args.val))

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        FusionTensorDataset(train_pack),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        FusionTensorDataset(val_pack),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )

    image_dim = train_pack.image.shape[1]
    text_dim = train_pack.text.shape[1]
    audio_dim = train_pack.audio.shape[1] if train_pack.audio is not None else None
    num_classes = len(train_pack.class_names)

    model = MultimodalFusionHead(
        device=device,
        image_dim=image_dim,
        text_dim=text_dim,
        audio_dim=audio_dim,
        num_classes=num_classes,
        class_names=train_pack.class_names,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for image_embed, text_embed, audio_embed, labels in train_loader:
            image_embed = image_embed.to(device)
            text_embed = text_embed.to(device)
            labels = labels.to(device)
            audio_embed = audio_embed.to(device) if audio_embed is not None else None

            optimizer.zero_grad()
            logits = model(image_embed, text_embed, audio_embed)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image_embed.size(0)

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
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "class_names": train_pack.class_names,
                        "image_dim": image_dim,
                        "text_dim": text_dim,
                        "audio_dim": audio_dim,
                        "num_classes": num_classes,
                    },
                },
                output_path,
            )
            log.info("Saved new best fusion checkpoint", path=str(output_path), accuracy=round(best_acc, 4))

    log.info("Fusion training complete", best_accuracy=round(best_acc, 4))


@torch.no_grad()
def evaluate(model: MultimodalFusionHead, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for image_embed, text_embed, audio_embed, labels in loader:
        image_embed = image_embed.to(device)
        text_embed = text_embed.to(device)
        labels = labels.to(device)
        audio_embed = audio_embed.to(device) if audio_embed is not None else None

        logits = model(image_embed, text_embed, audio_embed)
        loss = criterion(logits, labels)

        total_loss += loss.item() * image_embed.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += image_embed.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


if __name__ == "__main__":
    train()
