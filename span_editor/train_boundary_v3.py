"""Train a RoBERTa classifier for evidence-conditioned edit boundaries."""

import argparse
import json
from pathlib import Path

from span_editor.boundary_v3 import build_examples
from span_editor.common import ranked, sha256, write_json
from span_editor.v2_data import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--negatives", type=int, default=6)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-span-chars", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/revproof_boundary_v3_cache"))
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_length, args.negatives, args.radius,
           args.max_span_chars, args.epochs, args.logging_steps) < 1:
        parser.error("training limits must be positive")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("training/development IDs overlap")

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                               cache_dir=str(args.cache_dir))
    examples, positives, skipped = build_examples(
        tokenizer, train, args.max_length, args.negatives, args.radius,
        args.max_span_chars)
    if positives < 100 or len(examples) <= positives:
        raise ValueError("insufficient positive and negative boundary examples")
    protocol = {"role": "evidence_conditioned_boundary_ranker_v3",
                "model": args.model, "train_sha256": sha256(args.train),
                "dev_sha256": sha256(args.dev),
                "train_ids": [row["id"] for row in train],
                "selected_train": len(train), "positive": positives,
                "skipped": skipped, "examples": len(examples),
                "max_length": args.max_length,
                "negatives": args.negatives, "radius": args.radius,
                "max_span_chars": args.max_span_chars,
                "epochs": args.epochs, "seed": args.seed,
                "label_source": "FRUIT_silver_character_span"}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / "protocol.json"
    if protocol_file.exists() and json.loads(
            protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory has a different boundary protocol")
    write_json(protocol_file, protocol)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, cache_dir=str(args.cache_dir))
    settings = TrainingArguments(
        output_dir=str(args.output / "trainer_state"),
        num_train_epochs=args.epochs, per_device_train_batch_size=16,
        gradient_accumulation_steps=1, learning_rate=2e-5,
        weight_decay=0.01, warmup_ratio=0.05,
        logging_steps=args.logging_steps, save_strategy="no", report_to="none",
        remove_unused_columns=False, seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported())
    trainer = Trainer(model=model, args=settings, train_dataset=examples,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      processing_class=tokenizer)
    print(f"START BOUNDARY V3: {len(examples)} examples", flush=True)
    metrics = trainer.train().metrics
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
