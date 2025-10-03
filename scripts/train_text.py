"""Fine-tune a QA head on the PlantGuard FAQ dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.backend.services.text.dataset import PlantFaqDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a QA model for PlantGuard")
    parser.add_argument("--model", default="bert-base-uncased", help="Base transformer model to fine-tune")
    parser.add_argument("--data-dir", default="data/processed/faq", help="Directory containing SQuAD-style JSON files")
    parser.add_argument("--output", default="models/text/plantguard_qa_head", help="Where to store trained model + tokenizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Maximum sequence length for tokenized inputs")
    parser.add_argument("--doc-stride", type=int, default=128, help="Stride when splitting long contexts")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional number of training examples to keep for quick experiments")
    return parser.parse_args()


def prepare_features_factory(tokenizer, max_length: int, doc_stride: int):
    def prepare_features(examples: Dict[str, List]) -> Dict[str, List]:
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions: List[int] = []
        end_positions: List[int] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            sequence_ids = tokenized_examples.sequence_ids(i)

            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if token_start_index > token_end_index or not (
                offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char
            ):
                start_positions.append(token_start_index)
                end_positions.append(token_start_index)
                continue

            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    return prepare_features


def train() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = PlantFaqDataset(path=args.data_dir).load()

    if args.sample_size:
        size = min(args.sample_size, len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(size))
        if "validation" in dataset:
            val_size = max(1, min(args.sample_size // 5, len(dataset["validation"])))
            dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = "epoch" if "validation" in dataset else "no"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=eval_strategy,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_total_limit=1,
        remove_unused_columns=True,
    )

    prepare_features = prepare_features_factory(tokenizer, args.max_length, args.doc_stride)

    remove_columns = dataset["train"].column_names
    tokenized_train = dataset["train"].map(
        prepare_features,
        batched=True,
        remove_columns=remove_columns,
    )

    tokenized_eval = None
    if "validation" in dataset:
        tokenized_eval = dataset["validation"].map(
            prepare_features,
            batched=True,
            remove_columns=dataset["validation"].column_names,
        )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    train()
