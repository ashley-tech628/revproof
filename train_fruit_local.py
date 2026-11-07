"""Train a 4-bit FLAN-T5-XL oracle-locus local replacement diagnostic.

Run --positive-control first. It intentionally trains and evaluates on the
same eight rows to check that serialization, gradients and decoding work.
Never report that control as held-out generalization or an end-to-end editor.
"""

import argparse
import hashlib
import json
from pathlib import Path

from fruit_local import choose, load, prompt


def encode(tokenizer, row, max_input, max_target):
    source = tokenizer(prompt(row), add_special_tokens=True, truncation=False).input_ids
    target = tokenizer(row["replacement"], add_special_tokens=True, truncation=False).input_ids
    if len(source) > max_input or len(target) > max_target:
        return None
    return {"input_ids": source, "attention_mask": [1] * len(source), "labels": target}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--max-train-rows", type=int, default=2000)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    if min(args.max_train_rows, args.max_input_length, args.max_target_length,
           args.max_steps, args.logging_steps) < 1:
        ap.error("row, token and step limits must be positive")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    train = choose(load(args.train, "train"), args.max_train_rows)
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/dev overlap")
    print(f"LOAD TOKENIZER: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = []
    for index, row in enumerate(train, 1):
        encoded = encode(tokenizer, row, args.max_input_length, args.max_target_length)
        if encoded is not None:
            pairs.append((row, encoded))
        if index == 1 or index % 100 == 0 or index == len(train):
            print(f"ENCODE: {index}/{len(train)}, eligible {len(pairs)}", flush=True)
    if args.positive_control:
        pairs = sorted(pairs, key=lambda p: (len(p[1]["input_ids"]) +
                                           len(p[1]["labels"]), p[0]["id"]))[:8]
    if not pairs:
        raise ValueError("no token-eligible training rows")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"role": "oracle_locus_silver_diagnostic",
                "positive_control": args.positive_control,
                "train_equals_eval": args.positive_control,
                "model": args.model, "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                "selected_train": len(train), "eligible_train": len(pairs),
                "train_ids": [x["id"] for x, _ in pairs],
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "max_steps": args.max_steps, "seed": args.seed,
                "prompt_version": 1}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory already contains another protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in protocol.items() if k != "train_ids"}), flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    print(f"LOAD MODEL: {args.model}; first download may take time", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  quantization_config=quant,
                                                  device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    settings = TrainingArguments(
        output_dir=str(args.output / "checkpoints"), max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=min(500, args.max_steps),
        save_total_limit=2, report_to="none", remove_unused_columns=False,
        seed=args.seed, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=settings,
                      train_dataset=[encoded for _, encoded in pairs],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                      processing_class=tokenizer)
    print(f"START TRAINING: {args.max_steps} steps; loss every {args.logging_steps}", flush=True)
    metrics = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None).metrics
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    (args.output / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
