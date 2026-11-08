"""Fine-tune FLAN-T5-XL on reference-selected FRUIT span replacements."""

import argparse
import json
from pathlib import Path

from span_editor.common import encode_infiller, ranked, sha256, write_json
from span_editor.v2_data import load


def referenced_view(row, max_items=3):
    references = row["reference_ids"][:max_items]
    if not references:
        return None
    evidence = " ".join(f"({i}) {row['evidence_items'][i]}" for i in references)
    return {**row, "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-reference-items", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_input_length, args.max_target_length,
           args.max_reference_items, args.max_steps, args.logging_steps,
           args.save_steps) < 1:
        parser.error("all numeric limits must be positive")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/development ID overlap")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = []
    missing_refs = 0
    for n, row in enumerate(train, 1):
        view = referenced_view(row, args.max_reference_items)
        if view is None:
            missing_refs += 1
            continue
        encoded = encode_infiller(tokenizer, view, args.max_input_length,
                                  args.max_target_length)
        if encoded is not None:
            pairs.append((row["id"], encoded))
        if n == 1 or n % 1000 == 0 or n == len(train):
            print(f"ENCODE INFILLER V2: {n}/{len(train)}, eligible={len(pairs)}",
                  flush=True)
    if not pairs:
        raise ValueError("no token-eligible reference-conditioned rows")
    protocol = {"role": "reference_conditioned_span_infiller_v2",
                "model": args.model, "train_sha256": sha256(args.train),
                "dev_sha256": sha256(args.dev),
                "train_ids": [identifier for identifier, _ in pairs],
                "selected_train": len(train), "eligible_train": len(pairs),
                "missing_references": missing_refs,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "max_reference_items": args.max_reference_items,
                "max_steps": args.max_steps, "seed": args.seed,
                "prompt_version": 2}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory has a different infiller protocol")
    write_json(path, protocol)
    print(json.dumps({k: v for k, v in protocol.items() if k != "train_ids"}), flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    print(f"LOAD INFILLER V2: {args.model}", flush=True)
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
        save_strategy="steps", save_steps=min(args.save_steps, args.max_steps),
        save_total_limit=2, report_to="none", remove_unused_columns=False,
        seed=args.seed, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=settings,
                      train_dataset=[encoded for _, encoded in pairs],
                      data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                      processing_class=tokenizer)
    print(f"START INFILLER V2: {args.max_steps} steps; loss every "
          f"{args.logging_steps}", flush=True)
    metrics = trainer.train(resume_from_checkpoint=str(args.resume)
                            if args.resume else None).metrics
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    write_json(args.output / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
