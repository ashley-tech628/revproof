"""QLoRA SFT of a local replacement-span generator. Run on Colab GPU."""
import argparse
import hashlib
import json
from pathlib import Path

from revproof_data import prompt, read_records


def encode_row(tokenizer, row, max_length):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt(row)}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(row["new_span"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if len(prefix_ids) + len(answer_ids) > max_length:
        raise ValueError(f"{row['id']}: {len(prefix_ids)+len(answer_ids)} tokens exceed {max_length}")
    return {"input_ids": prefix_ids + answer_ids,
            "attention_mask": [1] * (len(prefix_ids) + len(answer_ids)),
            "labels": [-100] * len(prefix_ids) + answer_ids}


class EditDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items = [encode_row(tokenizer, row, max_length) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad_id):
    import torch
    length = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (length-len(x["input_ids"])) for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (length-len(x["attention_mask"])) for x in batch]),
        "labels": torch.tensor([x["labels"] + [-100] * (length-len(x["labels"])) for x in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gradient-accumulation", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-synthetic-smoke", action="store_true")
    args = ap.parse_args()

    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU runtime")
    rows = read_records(args.data)
    if not args.allow_synthetic_smoke and any(r["source_type"] != "real" for r in rows):
        raise ValueError("toy examples cannot be used for a research run")
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if not train or not dev:
        raise ValueError("need nonempty train and dev splits")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if len({r["group"] for r in train}) < 3 and not args.allow_synthetic_smoke:
        print("Warning: fewer than three independent train revisions; this is a pipeline pilot only.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.lr,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=1, report_to="none", gradient_checkpointing=True,
        fp16=not bf16, bf16=bf16, optim="paged_adamw_8bit",
        remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(model=model, args=train_args,
                      train_dataset=EditDataset(train, tokenizer, args.max_length),
                      eval_dataset=EditDataset(dev, tokenizer, args.max_length),
                      data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
                      processing_class=tokenizer)
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    (output / "run.json").write_text(json.dumps({
        "model": args.model, "train_count": len(train), "dev_count": len(dev),
        "train_groups": len({r["group"] for r in train}),
        "dev_groups": len({r["group"] for r in dev}),
        "data": args.data, "train_metrics": result.metrics,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "dev_metrics": metrics, "toy_smoke": args.allow_synthetic_smoke,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "gradient_accumulation": args.gradient_accumulation,
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
