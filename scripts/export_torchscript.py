"""Export trained models to TorchScript for deployment."""
from __future__ import annotations

import torch

from plantguard.src.backend.services.vision import VisionClassifier


def main() -> None:
    model = VisionClassifier(device=torch.device("cpu"))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, dummy_input)
    traced.save("models/vision/plantguard_resnet50.ts")


if __name__ == "__main__":
    main()
