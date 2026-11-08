"""Train a sentence/evidence-item classifier on FRUIT reference labels."""

import argparse
import json
from pathlib import Path

from span_editor.common import ranked, sha256, write_json
from span_editor.locator import lexical_overlap
from span_editor.v2_data import load


def sentence_shortlist(row, limit=4):
    return sorted(range(len(row["source_sentences"])),
                  key=lambda i: (-lexical_overlap(row["source_sentences"][i],
                                                   row["evidence"]), i))[:limit]


def training_pairs(row, negatives=3, wrong_sentences=2, max_references=2):
    """(sentence_index, evidence_index, label); references never enter inference."""
    refs = [i for i in row["reference_ids"] if row["evidence_items"][i]][:max_references]
    if not refs:
        return []
    gold = row["edit_index"]
    pairs = [(gold, i, 1) for i in refs]
    unused = [i for i, item in enumerate(row["evidence_items"])
              if item and i not in row["reference_ids"]]
    unused.sort(key=lambda i: (-lexical_overlap(row["source_sentences"][gold],
                                                row["evidence_items"][i]), i))
    pairs.extend((gold, i, 0) for i in unused[:negatives])
    others = [i for i in sentence_shortlist(row, len(row["source_sentences"]))
              if i != gold][:wrong_sentences]
    pairs.extend((i, refs[0], 0) for i in others)
    return pairs


def encode_pair(tokenizer, sentence, evidence, max_length=384):
    return tokenizer(sentence, evidence, truncation=True, max_length=max_length)


def build_examples(tokenizer, rows, max_length, negatives, wrong_sentences):
    examples = []
    positive = negative = 0
    for n, row in enumerate(rows, 1):
        for sentence_id, evidence_id, label in training_pairs(
                row, negatives, wrong_sentences):
            encoded = encode_pair(tokenizer, row["source_sentences"][sentence_id],
                                  row["evidence_items"][evidence_id], max_length)
            encoded["labels"] = label
            examples.append(encoded)
            positive += label
            negative += not label
        if n == 1 or n % 1000 == 0 or n == len(rows):
            print(f"ENCODE SELECTOR: {n}/{len(rows)}, positive={positive}, "
                  f"negative={negative}", flush=True)
    return examples, positive, negative


def rank_pairs(model, tokenizer, row, torch, top_sentences=4, top_pairs=6,
               max_length=384, batch_size=12):
    """Inference uses only old source sentences and numbered evidence items."""
    sentence_ids = sentence_shortlist(row, top_sentences)
    keys = [(s, e) for s in sentence_ids
            for e, item in enumerate(row["evidence_items"]) if item]
    scored = []
    device = next(model.parameters()).device
    model.eval()
    for offset in range(0, len(keys), batch_size):
        chunk = keys[offset:offset + batch_size]
        fields = tokenizer([row["source_sentences"][s] for s, _ in chunk],
                           [row["evidence_items"][e] for _, e in chunk],
                           padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt")
        fields = {key: value.to(device) for key, value in fields.items()}
        with torch.inference_mode():
            logits = model(**fields).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
        for (sentence_id, evidence_id), probability in zip(chunk, probs):
            scored.append({"sentence_index": sentence_id, "evidence_id": evidence_id,
                           "selector_probability": probability})
    return sorted(scored, key=lambda x: (-x["selector_probability"],
                                          x["sentence_index"], x["evidence_id"]))[:top_pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--wrong-sentences", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/revproof_selector_cache"))
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_length, args.epochs, args.logging_steps) < 1:
        parser.error("invalid positive limit")
    if args.negatives < 0 or args.wrong_sentences < 0:
        parser.error("negative counts must be nonnegative")
    rows = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in rows} & {x["id"] for x in dev}:
        raise ValueError("train/development ID overlap")

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                               cache_dir=str(args.cache_dir))
    examples, positive, negative = build_examples(tokenizer, rows, args.max_length,
                                                   args.negatives, args.wrong_sentences)
    if not positive or not negative:
        raise ValueError("selector needs positive and negative evidence pairs")
    protocol = {"role": "reference_supervised_sentence_evidence_selector",
                "model": args.model, "train_sha256": sha256(args.train),
                "dev_sha256": sha256(args.dev), "train_ids": [x["id"] for x in rows],
                "selected_train": len(rows), "positive": positive, "negative": negative,
                "max_length": args.max_length, "negatives": args.negatives,
                "wrong_sentences": args.wrong_sentences, "epochs": args.epochs,
                "seed": args.seed, "label_source": "heuristic_FRUIT_target_references"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory has a different selector protocol")
    write_json(path, protocol)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, cache_dir=str(args.cache_dir))
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
    print(f"START SELECTOR: {len(examples)} pairs", flush=True)
    metrics = trainer.train().metrics
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
