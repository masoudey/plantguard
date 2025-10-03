"""Evaluation script for PlantGuard models."""
from __future__ import annotations

import json
from pathlib import Path

from plantguard.src.backend.utils.metrics import classification_report


def main() -> None:
    y_true = [0, 1, 2]
    y_pred = [0, 2, 2]
    metrics = classification_report(y_true, y_pred)
    output_path = Path("reports/metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
