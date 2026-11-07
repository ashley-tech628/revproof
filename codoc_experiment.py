"""Qwen QLoRA pilot on CoDocBench silver code/docstring revisions.

This is code-conditioned editing, not audited version-grounded updating.
Modes: full-text baseline or multi-span edit-plan generation. Positive control
intentionally evaluates the same three training examples to check learnability.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multispan_edit import apply_plan, gold_plan


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or any(row.get("source_dataset") != "kunpai/codocbench" or
                       row.get("support_audited") is not False or
                       row.get("training_role") != "domain_silver_candidate" for row in rows):
        raise ValueError("requires nonempty, unaudited CoDocBench silver rows")
    return rows


def project_split(rows, positive_control=False):
    if positive_control:
        selected = sorted(rows, key=lambda r: len(r["old_text"]) +
                          len(r["new_text"]) + len(r["evidence_text"]))[:3]
        return selected, selected
    projects = sorted({r["group"] for r in rows},
                      key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    n_dev = max(1, len(projects) // 10)
    dev_projects = set(projects[:n_dev])
    train = [r for r in rows if r["group"] not in dev_projects]
    dev = [r for r in rows if r["group"] in dev_projects]
    if not train or not dev or {r["group"] for r in train} & dev_projects:
        raise ValueError("invalid project split")
    return train, dev


def prompt(row, objective):
    instruction = ("Output a JSON array of source-offset edits, each with exactly "
                   "start, end, new. Offsets refer to the original old docstring; "
                   "copy everything outside the edits exactly. No explanation."
                   if objective == "plan" else
                   "Update the old Python docstring using the code change. "
                   "Output only the complete new docstring, with no explanation.")
    return (f"{instruction}\nFunction: {row['id']}\n"
            f"Old docstring:\n{row['old_text']}\n"
            f"Code diff (silver evidence, not audited):\n{row['evidence_text']}")


def target(row, objective):
    return (json.dumps(gold_plan(row["old_text"], row["new_text"]),
                       ensure_ascii=False, separators=(",", ":"))
            if objective == "plan" else row["new_text"])


def encode(tokenizer, row, objective, max_input, max_target):
    source = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt(row, objective)}],
        tokenize=False, add_generation_prompt=True)
    source_ids = tokenizer(source, add_special_tokens=False).input_ids
    target_ids = tokenizer(target(row, objective), add_special_tokens=False).input_ids
    if len(source_ids) > max_input or len(target_ids) + 1 > max_target:
        return None
    target_ids += [tokenizer.eos_token_id]
    return {"input_ids": source_ids + target_ids,
            "attention_mask": [1] * (len(source_ids) + len(target_ids)),
            "labels": [-100] * len(source_ids) + target_ids}


def matched_encode(tokenizer, rows, objective, max_input, max_target):
    """Keep exactly the same eligible rows for full and plan comparisons."""
    other = "plan" if objective == "full" else "full"
    selected = []
    for row in rows:
        item = encode(tokenizer, row, objective, max_input, max_target)
        if item is not None and encode(tokenizer, row, other, max_input, max_target) is not None:
            selected.append((row, item))
    return selected


def assess(row, objective, raw):
    result = {"prediction": None, "valid_plan": None, "exact": False, "error": None}
    if objective == "full":
        result.update(prediction=raw, exact=raw == row["new_text"])
        return result
    try:
        plan = json.loads(raw)
        result["prediction"] = apply_plan(row["old_text"], plan)
        result["valid_plan"] = True
        result["exact"] = result["prediction"] == row["new_text"]
    except (ValueError, TypeError, KeyError) as exc:
        result["valid_plan"] = False
        result["error"] = str(exc)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--objective", choices=["full", "plan"], required=True)
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_steps < 1:
        ap.error("max-steps must be positive")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from train_sparse_editor import collate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    set_seed(args.seed)
    rows = load_rows(args.data)
    train, dev = project_split(rows, args.positive_control)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    train_pairs = matched_encode(tokenizer, train, args.objective,
                                 args.max_input_length, args.max_target_length)
    dev_pairs = matched_encode(tokenizer, dev, args.objective,
                               args.max_input_length, args.max_target_length)
    if not train_pairs or not dev_pairs:
        raise ValueError("token limits excluded all train or dev rows")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "model": MODEL, "objective": args.objective,
                "positive_control": args.positive_control,
                "training_role": "unaudited_domain_silver",
                "train_count": len(train_pairs), "dev_count": len(dev_pairs),
                "train_projects": len({r["group"] for r, _ in train_pairs}),
                "dev_projects": len({r["group"] for r, _ in dev_pairs}),
                "max_steps": args.max_steps, "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length, "seed": args.seed}
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output directory already uses a different protocol")
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                 device_map="auto")
    if args.stage in {"train", "all"}:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"))
        model.config.use_cache = False
        settings = TrainingArguments(
            output_dir=str(output / "checkpoints"), max_steps=args.max_steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            learning_rate=1e-4, logging_steps=10, save_strategy="no",
            report_to="none", remove_unused_columns=False, seed=args.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit")
        trainer = Trainer(model=model, args=settings,
                          train_dataset=[x for _, x in train_pairs],
                          data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
                          processing_class=tokenizer)
        metrics = trainer.train().metrics
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
    for row, _ in dev_pairs:
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(row, args.objective)}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(source, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_target_length,
                                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        predictions.append({"id": row["id"], "group": row["group"],
                            "raw": raw, "gold": row["new_text"],
                            **assess(row, args.objective, raw)})
    (output / "dev_predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n",
        encoding="utf-8")
    summary = {**protocol, "dev_exact": sum(x["exact"] for x in predictions),
               "dev_n": len(predictions),
               "dev_valid_plans": sum(x["valid_plan"] is True for x in predictions)
               if args.objective == "plan" else None,
               "warning": "Silver code co-change is not audited factual support."}
    (output / "dev_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
