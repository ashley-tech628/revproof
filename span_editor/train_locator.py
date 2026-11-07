"""Train a RoBERTa QA locator with gold spans and hard negative sentences."""

import argparse
import json
from pathlib import Path

from fruit_span import load
from span_editor.common import ranked, sha256, write_json
from span_editor.locator import candidate_sentences, encode_pair


def build_examples(tokenizer, rows, max_length, negatives):
    examples = []
    kept_positive = skipped_positive = 0
    for count, row in enumerate(rows, 1):
        positive = encode_pair(tokenizer, row, row["edit_index"], max_length, True)
        if positive is None:
            skipped_positive += 1
            continue
        examples.append(positive[0])
        kept_positive += 1
        for index in candidate_sentences(row, negatives)[1:]:
            negative = encode_pair(tokenizer, row, index, max_length, True)
            if negative is not None:
                examples.append(negative[0])
        if count == 1 or count % 100 == 0 or count == len(rows):
            print(f"ENCODE LOCATOR: {count}/{len(rows)}; positive {kept_positive}; "
                  f"total {len(examples)}", flush=True)
    return examples, kept_positive, skipped_positive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--max-train-rows", type=int, default=1100)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/revproof_locator_cache"))
    args = ap.parse_args()
    if min(args.max_train_rows, args.max_length, args.epochs, args.logging_steps) < 1 or args.negatives < 0:
        ap.error("invalid numeric limit")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/dev ID overlap")

    import torch
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                               cache_dir=str(args.cache_dir))
    if not tokenizer.is_fast or tokenizer.cls_token_id is None:
        raise ValueError("locator requires a fast tokenizer with CLS token")
    examples, positive, skipped = build_examples(tokenizer, train,
                                                   args.max_length, args.negatives)
    if not examples:
        raise ValueError("no locator examples")
    protocol = {"role": "evidence_conditioned_span_locator",
                "model": args.model, "train_sha256": sha256(args.train),
                "dev_sha256": sha256(args.dev), "train_ids": [x["id"] for x in train],
                "selected_train": len(train), "eligible_positive": positive,
                "skipped_positive": skipped, "training_examples": len(examples),
                "max_length": args.max_length, "negatives": args.negatives,
                "epochs": args.epochs, "seed": args.seed, "question_version": 1}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory contains a different locator protocol")
    write_json(path, protocol)
    print(json.dumps({k: v for k, v in protocol.items() if k != "train_ids"}), flush=True)
    print(f"LOAD LOCATOR: {args.model}", flush=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model,
                                                           cache_dir=str(args.cache_dir))
    settings = TrainingArguments(
        output_dir=str(args.output / "trainer_state"),
        num_train_epochs=args.epochs, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=3e-5,
        weight_decay=0.01, warmup_ratio=0.05, logging_steps=args.logging_steps,
        save_strategy="no", report_to="none", remove_unused_columns=False,
        seed=args.seed, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported())
    trainer = Trainer(model=model, args=settings, train_dataset=examples,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      processing_class=tokenizer)
    print(f"START LOCATOR TRAINING: {len(examples)} examples, {args.epochs} epochs",
          flush=True)
    metrics = trainer.train().metrics
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
