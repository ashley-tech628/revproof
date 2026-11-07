"""Fine-tune FLAN-T5-XL to emit only the missing short span (oracle training labels)."""

import argparse
import json
from pathlib import Path

from fruit_span import load
from span_editor.common import encode_infiller, ranked, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--max-train-rows", type=int, default=1100)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=96)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    if min(args.max_train_rows, args.max_input_length, args.max_target_length,
           args.max_steps, args.logging_steps, args.save_steps) < 1:
        ap.error("all numeric limits must be positive")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/dev ID overlap")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    print(f"LOAD TOKENIZER: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = []
    for i, row in enumerate(train, 1):
        item = encode_infiller(tokenizer, row, args.max_input_length, args.max_target_length)
        if item is not None:
            pairs.append((row, item))
        if i == 1 or i % 100 == 0 or i == len(train):
            print(f"ENCODE: {i}/{len(train)}; eligible {len(pairs)}", flush=True)
    if args.positive_control:
        pairs = sorted(pairs, key=lambda p: (len(p[1]["input_ids"]) + len(p[1]["labels"]),
                                             p[0]["id"]))[:8]
    if not pairs:
        raise ValueError("no token-eligible training rows")
    protocol = {"role": "oracle_span_infiller", "positive_control": args.positive_control,
                "train_equals_eval": args.positive_control, "model": args.model,
                "train_sha256": sha256(args.train), "dev_sha256": sha256(args.dev),
                "selected_train": len(train), "eligible_train": len(pairs),
                "train_ids": [x["id"] for x, _ in pairs],
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "max_steps": args.max_steps, "seed": args.seed, "prompt_version": 1}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory contains a different training protocol")
    write_json(path, protocol)
    print(json.dumps({k: v for k, v in protocol.items() if k != "train_ids"}), flush=True)

    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    print(f"LOAD MODEL: {args.model}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  quantization_config=quant, device_map="auto")
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
        save_strategy="steps", save_steps=min(args.save_steps, args.max_steps),
        save_total_limit=2, report_to="none", remove_unused_columns=False,
        seed=args.seed, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=settings,
                      train_dataset=[item for _, item in pairs],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                      processing_class=tokenizer)
    print(f"START TRAINING: {args.max_steps} steps; loss every {args.logging_steps}", flush=True)
    metrics = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None).metrics
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
