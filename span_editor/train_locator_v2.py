"""Train QA span locator on individual referenced evidence items."""

import argparse
import json
from pathlib import Path

from span_editor.common import ranked, sha256, write_json
from span_editor.train_locator import build_examples
from span_editor.v2_data import load


def reference_views(rows, max_references=2):
    views = []
    for row in rows:
        for index in row["reference_ids"][:max_references]:
            views.append({**row, "evidence": row["evidence_items"][index]})
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/revproof_locator_v2_cache"))
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_length, args.epochs, args.logging_steps) < 1:
        parser.error("invalid positive limit")
    if args.negatives < 0:
        parser.error("negative count must be nonnegative")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/development ID overlap")

    import torch
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                               cache_dir=str(args.cache_dir))
    if not tokenizer.is_fast or tokenizer.cls_token_id is None:
        raise ValueError("locator needs a fast tokenizer with CLS token")
    views = reference_views(train)
    examples, positive, skipped = build_examples(tokenizer, views,
                                                   args.max_length, args.negatives)
    if not examples:
        raise ValueError("no token-eligible reference views")
    protocol = {"role": "evidence_item_span_locator_v2", "model": args.model,
                "train_sha256": sha256(args.train), "dev_sha256": sha256(args.dev),
                "train_ids": [x["id"] for x in train],
                "selected_train": len(train), "reference_views": len(views),
                "eligible_positive": positive, "skipped_positive": skipped,
                "training_examples": len(examples), "max_length": args.max_length,
                "negatives": args.negatives, "epochs": args.epochs,
                "seed": args.seed, "question_version": 2}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory has a different locator protocol")
    write_json(path, protocol)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model, cache_dir=str(args.cache_dir))
    settings = TrainingArguments(
        output_dir=str(args.output / "trainer_state"),
        num_train_epochs=args.epochs, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=3e-5,
        weight_decay=0.01, warmup_ratio=0.05,
        logging_steps=args.logging_steps, save_strategy="no", report_to="none",
        remove_unused_columns=False, seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported())
    trainer = Trainer(model=model, args=settings, train_dataset=examples,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      processing_class=tokenizer)
    print(f"START LOCATOR V2: {len(examples)} examples", flush=True)
    metrics = trainer.train().metrics
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
