"""FRUIT-WIKI sentence-copy editor pilot with a 3B T5-family model.

Train only on official FRUIT training shards and choose hyperparameters on
official development shard 00009. Do not use gold_test for model selection.
Outputs are the publisher's EDIT5-style sentence-copy/evidence-reference
format, not free-form character offsets. GPU required for model execution.
"""

import argparse
import hashlib
import json
from pathlib import Path

from fruit_data import normalize_fruit


def load_rows(paths, split):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream)
    if not rows or any(row.get("source_dataset") != "FRUIT-WIKI" or
                       row.get("split") != split for row in rows):
        raise ValueError(f"requires nonempty FRUIT-WIKI {split} rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {split} IDs")
    return rows


def choose_rows(rows, limit):
    if limit is None or limit >= len(rows):
        return rows
    return sorted(rows, key=lambda x: hashlib.sha256(str(x["id"]).encode()).digest())[:limit]


def encode(tokenizer, row, max_input, max_target):
    src = tokenizer(row["input"], add_special_tokens=True, truncation=False).input_ids
    tgt = tokenizer(row["target"], add_special_tokens=True, truncation=False).input_ids
    if len(src) > max_input or len(tgt) > max_target:
        return None
    return {"input_ids": src, "attention_mask": [1] * len(src), "labels": tgt}


def encode_rows(tokenizer, rows, max_input, max_target, label, progress_every):
    pairs = []
    for index, row in enumerate(rows, 1):
        encoded = encode(tokenizer, row, max_input, max_target)
        if encoded is not None:
            pairs.append((row, encoded))
        if index == 1 or index % progress_every == 0 or index == len(rows):
            print(f"ENCODE {label}: {index}/{len(rows)} rows, "
                  f"{len(pairs)} within token limits", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", nargs="+", required=True, type=Path)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--max-train-rows", type=int)
    ap.add_argument("--max-dev-rows", type=int, default=200)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--resume", type=Path, help="resume a same-protocol Trainer checkpoint")
    args = ap.parse_args()
    if min(args.max_steps, args.max_input_length, args.max_target_length,
           args.logging_steps, args.progress_every) < 1:
        ap.error("steps, progress interval, and token limits must be positive")
    if args.max_train_rows is not None and args.max_train_rows < 1:
        ap.error("max-train-rows must be positive")
    if args.max_dev_rows is not None and args.max_dev_rows < 1:
        ap.error("max-dev-rows must be positive")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    train = load_rows(args.train, "train")
    dev = load_rows([args.dev], "dev")
    if set(x["id"] for x in train) & set(x["id"] for x in dev):
        raise ValueError("train/dev IDs overlap")
    train = choose_rows(train, args.max_train_rows)
    dev = choose_rows(dev, args.max_dev_rows)
    if args.positive_control:
        train = sorted(train, key=lambda x: len(x["input"]) + len(x["target"]))[:8]
        dev = train
    print(f"LOAD TOKENIZER: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_pairs = encode_rows(tokenizer, train, args.max_input_length,
                             args.max_target_length, "train",
                             args.progress_every)
    dev_pairs = encode_rows(tokenizer, dev, args.max_input_length,
                           args.max_target_length, "dev",
                           args.progress_every)
    if not train_pairs or not dev_pairs:
        raise ValueError("token limits exclude every training or development row")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*args.train, args.dev]}
    protocol = {"model": args.model, "training_role": "FRUIT_WIKI_silver",
                "positive_control": args.positive_control,
                "train_ids_overlap_dev": args.positive_control,
                "file_sha256": hashes, "train_count": len(train_pairs),
                "dev_count": len(dev_pairs), "max_steps": args.max_steps,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "logging_steps": args.logging_steps, "seed": args.seed}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory already has a different protocol")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"selected_train": len(train), "eligible_train": len(train_pairs),
                      "selected_dev": len(dev), "eligible_dev": len(dev_pairs)}),
          flush=True)

    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    print(f"LOAD MODEL: {args.model} in 4-bit; first download can take time",
          flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  quantization_config=quant,
                                                  device_map="auto")
    print(f"MODEL READY: CUDA allocated "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    if args.stage in {"train", "all"}:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))
        model.config.use_cache = False
        settings = TrainingArguments(
            output_dir=str(output / "checkpoints"), max_steps=args.max_steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-4, logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=min(500, args.max_steps), save_total_limit=2,
            report_to="none", remove_unused_columns=False, seed=args.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit")
        trainer = Trainer(
            model=model, args=settings,
            train_dataset=[encoded for _, encoded in train_pairs],
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer)
        print(f"START TRAINING: {args.max_steps} steps; "
              f"loss every {args.logging_steps} steps; "
              f"checkpoint every {min(500, args.max_steps)} steps",
              flush=True)
        metrics = trainer.train(
            resume_from_checkpoint=str(args.resume) if args.resume else None).metrics
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(str(output / "adapter"))
        (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                   encoding="utf-8")
    else:
        model = PeftModel.from_pretrained(model, str(output / "adapter"))
    if args.stage == "train":
        return
    model.eval()
    predictions = []
    for row, encoded in dev_pairs:
        inputs = tokenizer(row["input"], return_tensors="pt", add_special_tokens=True)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(**inputs,
                                       max_new_tokens=args.max_target_length,
                                       do_sample=False, num_beams=1)
        raw = tokenizer.decode(generated[0], skip_special_tokens=True)
        try:
            _, prediction = normalize_fruit(row["input"], raw)
            error = None
        except ValueError as exc:
            prediction, error = None, str(exc)
        predictions.append({"id": row["id"], "raw": raw,
                            "target": row["target"], "prediction": prediction,
                            "gold": row["gold"], "source": row["source"],
                            "valid_copy": error is None, "error": error,
                            "exact": prediction == row["gold"]})
    (output / "dev_predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n",
        encoding="utf-8")
    summary = {**protocol, "dev_exact": sum(x["exact"] for x in predictions),
               "dev_n": len(predictions),
               "dev_valid_copy": sum(x["valid_copy"] for x in predictions),
               "copy_source_exact": sum(x["source"] == x["gold"] for x in predictions),
               "warning": "Exact match is not a factual-support metric; audit edits."}
    (output / "dev_summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
