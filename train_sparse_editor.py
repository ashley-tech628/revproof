"""LoRA SFT for exact-copy edit scripts: Qwen causal or FLAN-T5 seq2seq."""
import argparse
import hashlib
import json
from pathlib import Path

from revproof_data import read_records
from sparse_edit import edit_prompt, gold_script, oracle_infill_prompt, script_text


def encode_row(tokenizer, row, backend, max_input_length, max_target_length,
               objective="edit_script"):
    script = gold_script(row["old_span"], row["new_span"])
    if objective == "oracle_infill":
        source = oracle_infill_prompt(row)
        target = script["new"]
    elif objective == "edit_script":
        source = edit_prompt(row)
        target = script_text(script)
    else:
        raise ValueError(f"unknown objective: {objective}")
    if backend == "causal":
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": source}], tokenize=False,
            add_generation_prompt=True)
        source_ids = tokenizer(source, add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if len(source_ids) > max_input_length or len(target_ids) > max_target_length:
            raise ValueError(f"{row['id']}: source/target tokens {len(source_ids)}/{len(target_ids)} "
                             f"exceed limits {max_input_length}/{max_target_length}")
        return {"input_ids": source_ids + target_ids,
                "attention_mask": [1] * (len(source_ids) + len(target_ids)),
                "labels": [-100] * len(source_ids) + target_ids,
                "source_tokens": len(source_ids), "target_tokens": len(target_ids)}
    source_ids = tokenizer(source, add_special_tokens=True).input_ids
    target_ids = tokenizer(target, add_special_tokens=True).input_ids
    if len(source_ids) > max_input_length or len(target_ids) > max_target_length:
        raise ValueError(f"{row['id']}: source/target tokens {len(source_ids)}/{len(target_ids)} "
                         f"exceed limits {max_input_length}/{max_target_length}")
    return {"input_ids": source_ids, "attention_mask": [1] * len(source_ids),
            "labels": target_ids, "source_tokens": len(source_ids),
            "target_tokens": len(target_ids)}


def collate(batch, pad_id):
    import torch
    input_length = max(len(x["input_ids"]) for x in batch)
    label_length = max(len(x["labels"]) for x in batch)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (input_length-len(x["input_ids"]))
                                   for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (input_length-len(x["attention_mask"]))
                                        for x in batch]),
        "labels": torch.tensor([x["labels"] + [-100] * (label_length-len(x["labels"]))
                               for x in batch]),
    }


class EncodedDataset:
    def __init__(self, rows, tokenizer, backend, max_input_length, max_target_length,
                 objective="edit_script"):
        self.items = [encode_row(tokenizer, row, backend, max_input_length,
                                 max_target_length, objective)
                      for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["causal", "seq2seq"], required=True)
    ap.add_argument("--objective", choices=["edit_script", "oracle_infill"],
                    default="edit_script")
    ap.add_argument("--model")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or args.gradient_accumulation < 1:
        ap.error("epochs, lr, and gradient-accumulation must be positive")
    if args.max_input_length < 1 or args.max_target_length < 1:
        ap.error("token limits must be positive")
    model_name = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if args.backend == "causal"
                                else "google/flan-t5-large")

    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA")
    set_seed(args.seed)
    rows = read_records(args.data)
    if any(r["source_type"] != "real" or r.get("prompt_version") != "v2_fact_card"
           for r in rows):
        raise ValueError("sparse research training requires real v2 fact-card rows")
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if not train or not dev:
        raise ValueError("nonempty train and dev splits are required")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.backend == "causal":
        tokenizer.pad_token = tokenizer.eos_token
    train_set = EncodedDataset(train, tokenizer, args.backend,
                               args.max_input_length, args.max_target_length,
                               args.objective)
    dev_set = EncodedDataset(dev, tokenizer, args.backend,
                             args.max_input_length, args.max_target_length,
                             args.objective)
    lengths = {split: {"max_source": max(x["source_tokens"] for x in dataset.items),
                       "max_target": max(x["target_tokens"] for x in dataset.items)}
               for split, dataset in (("train", train_set), ("dev", dev_set))}
    print("Encoded lengths:", lengths, flush=True)

    bf16 = torch.cuda.is_bf16_supported()
    if args.backend == "causal":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        task_type, optimizer = "CAUSAL_LM", "paged_adamw_8bit"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if bf16 else torch.float32)
        target_modules = ["q", "v"]
        task_type, optimizer = "SEQ_2_SEQ_LM", "adamw_torch"
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=target_modules, bias="none", task_type=task_type))
    model.print_trainable_parameters()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, logging_steps=1, report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=args.backend == "causal" and not bf16,
        optim=optimizer, remove_unused_columns=False, seed=args.seed)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_set,
                      eval_dataset=dev_set,
                      data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
                      processing_class=tokenizer)
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    metadata = {
        "model": model_name, "backend": args.backend,
        "train_count": len(train), "dev_count": len(dev),
        "train_groups": len({r["group"] for r in train}),
        "dev_groups": len({r["group"] for r in dev}),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_metrics": result.metrics, "dev_metrics": metrics,
        "lengths": lengths, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "gradient_accumulation": args.gradient_accumulation,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "objective": args.objective,
        "oracle_location": args.objective == "oracle_infill",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
