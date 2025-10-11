#!/usr/bin/env python3
"""Quick helper to query the fine‑tuned text QA model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.services.text.qa_pipeline import answer_question as qa_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single question through the text QA head.")
    parser.add_argument("--question", required=True, help="Question to ask the model.")
    parser.add_argument(
        "--context",
        default="",
        help="Optional supporting context. If omitted, only the question is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = qa_answer(args.question, args.context)
    print("Question:", args.question)
    print("Context:", args.context if args.context else "<empty>")
    print("Answer :", result.get("answer", ""))
    print("Confidence:", round(result.get("confidence", 0.0), 4))
    print("Span:", (result.get("start"), result.get("end")))


if __name__ == "__main__":
    main()
