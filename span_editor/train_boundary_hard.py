"""Train the boundary ranker against actual selector/QA training proposals."""

import argparse
import json
from pathlib import Path

from span_editor.boundary_v3 import encode_candidate
from span_editor.common import ranked, sha256, write_json
from span_editor.v2_data import load


def build_examples(tokenizer, rows, mined, max_length=384):
    examples = []
    positives = skipped = negatives = 0
    for n, (row, mine) in enumerate(zip(rows, mined), 1):
        if row["id"] != mine["id"]:
            raise ValueError("mined proposal ID order differs from training rows")
        refs = [i for i in row["reference_ids"] if row["evidence_items"][i]]
        if not refs:
            skipped += 1
            continue
        sentence = row["source_sentences"][row["edit_index"]]
        a, b = row["span_start"], row["span_end"]
        positive = encode_candidate(tokenizer, sentence,
                                    row["evidence_items"][refs[0]],
                                    a, b, max_length)
        positive["labels"] = 1
        examples.append(positive)
        positives += 1
        for item in mine["negatives"]:
            sentence_index = item["sentence_index"]
            evidence_id = item["evidence_id"]
            if not 0 <= sentence_index < len(row["source_sentences"]):
                raise ValueError("mined sentence index outside training article")
            if not 0 <= evidence_id < len(row["evidence_items"]):
                raise ValueError("mined evidence index outside training row")
            source = row["source_sentences"][sentence_index]
            start, end = item["start"], item["end"]
            if not 0 <= start < end <= len(source):
                raise ValueError("invalid mined character span")
            if (sentence_index == row["edit_index"] and
                    start == a and end == b and evidence_id in refs):
                raise ValueError("positive training candidate marked negative")
            encoded = encode_candidate(tokenizer, source,
                                       row["evidence_items"][evidence_id],
                                       start, end, max_length)
            encoded["labels"] = 0
            examples.append(encoded)
            negatives += 1
        if n == 1 or n % 1000 == 0 or n == len(rows):
            print(f"ENCODE HARD BOUNDARY: {n}/{len(rows)}, positive={positives}, "
                  f"negative={negatives}", flush=True)
    return examples, positives, negatives, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--mine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/revproof_boundary_hard_cache"))
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_length, args.epochs,
           args.logging_steps) < 1:
        parser.error("training limits must be positive")
    rows = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in rows} & {x["id"] for x in dev}:
        raise ValueError("training/development IDs overlap")
    mine_protocol_path = args.mine.parent / "protocol.json"
    if not mine_protocol_path.is_file():
        parser.error("missing mining protocol")
    mine_protocol = json.loads(mine_protocol_path.read_text(encoding="utf-8"))
    if (mine_protocol.get("role") != "training_only_qa_proposal_hard_negatives" or
            mine_protocol.get("train_sha256") != sha256(args.train) or
            mine_protocol.get("dev_sha256") != sha256(args.dev) or
            mine_protocol.get("train_ids") != [x["id"] for x in rows]):
        parser.error("mined proposals do not match this training data")
    mined = [json.loads(line) for line in args.mine.open(encoding="utf-8")]
    if len(mined) != len(rows):
        parser.error("mined proposal row count mismatch")

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                               cache_dir=str(args.cache_dir))
    examples, positives, negatives, skipped = build_examples(
        tokenizer, rows, mined, args.max_length)
    if positives < 100 or negatives < positives:
        raise ValueError("insufficient hard-negative training examples")
    protocol = {"role": "evidence_conditioned_boundary_ranker_v3",
                "training_method": "selector_qa_hard_mined_candidates",
                "model": args.model, "train_sha256": sha256(args.train),
                "dev_sha256": sha256(args.dev),
                "mine_sha256": sha256(args.mine),
                "mine_protocol_sha256": sha256(mine_protocol_path),
                "train_ids": [row["id"] for row in rows],
                "selected_train": len(rows), "positive": positives,
                "negative": negatives, "skipped": skipped,
                "examples": len(examples), "max_length": args.max_length,
                "radius": mine_protocol["settings"]["radius"],
                "max_span_chars": mine_protocol["settings"]["max_span_chars"],
                "epochs": args.epochs, "seed": args.seed,
                "label_source": "FRUIT_silver_span_and_reference"}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / "protocol.json"
    if protocol_file.is_file() and json.loads(
            protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("training output has a different protocol")
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
    print(f"START HARD BOUNDARY: {len(examples)} examples", flush=True)
    metrics = trainer.train().metrics
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
