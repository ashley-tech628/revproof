"""Cheap, paired FRUIT development gate: copy, base model, and LoRA editor.

This is a diagnostic on silver development data, not the paper's gold-test
benchmark or a factual-support evaluation. Every condition uses the same rows.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from fruit_data import normalize_fruit
from train_fruit_editor import choose_rows, encode, load_rows


def words(text):
    return re.findall(r"\b\w+\b", text.casefold())


def lcs_f1(prediction, gold):
    a, b = words(prediction), words(gold)
    if not a or not b:
        return float(not a and not b)
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0] * (len(b) + 1)
        for j, reference in enumerate(b, 1):
            current[j] = previous[j - 1] + 1 if token == reference else max(
                previous[j], current[j - 1])
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def addition_f1(source, prediction, gold):
    """Unordered added-token F1; a diagnostic, not semantic grounding."""
    source_tokens = Counter(words(source))
    predicted_additions = Counter(words(prediction)) - source_tokens
    gold_additions = Counter(words(gold)) - source_tokens
    overlap = sum((predicted_additions & gold_additions).values())
    predicted_count = sum(predicted_additions.values())
    gold_count = sum(gold_additions.values())
    if not predicted_count and not gold_count:
        return 1.0
    if not predicted_count or not gold_count:
        return 0.0
    return 2 * overlap / (predicted_count + gold_count)


def score(source, prediction, gold):
    if prediction is None:
        return {"exact": False, "lcs_f1": 0.0,
                "addition_f1": 0.0, "changed": False}
    return {"exact": prediction == gold,
            "lcs_f1": lcs_f1(prediction, gold),
            "addition_f1": addition_f1(source, prediction, gold),
            "changed": prediction != source}


def summarize(items, key):
    values = [item[key] for item in items]
    n = len(values)
    return {"n": n, "exact": sum(x["scores"]["exact"] for x in values),
            "lcs_f1": sum(x["scores"]["lcs_f1"] for x in values) / n,
            "addition_f1": sum(x["scores"]["addition_f1"] for x in values) / n,
            "changed": sum(x["scores"]["changed"] for x in values),
            "reconstruction_valid": sum(x["error"] is None for x in values)}


def generate(model, tokenizer, row, encoded, max_target, device, torch):
    inputs = {"input_ids": torch.tensor([encoded["input_ids"]], device=device),
              "attention_mask": torch.tensor([encoded["attention_mask"]],
                                             device=device)}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_target,
                                   do_sample=False, num_beams=1)
    raw = tokenizer.decode(generated[0], skip_special_tokens=True)
    try:
        _, prediction = normalize_fruit(row["input"], raw)
        error = None
    except ValueError as exc:
        prediction, error = None, str(exc)
    return {"raw": raw, "prediction": prediction, "error": error,
            "copy_markers": len(re.findall(r"\[\d+\]", raw)),
            "scores": score(row["source"], prediction, row["gold"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="final adapter directory or Trainer checkpoint-N")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--max-dev-rows", type=int, default=30)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=384)
    args = parser.parse_args()
    if min(args.max_dev_rows, args.max_input_length, args.max_target_length) < 1:
        parser.error("row and token limits must be positive")
    if not (args.adapter / "adapter_config.json").exists():
        parser.error("adapter_config.json missing; select a saved adapter or checkpoint")
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(
        encoding="utf-8"))
    trained_base = adapter_config.get("base_model_name_or_path")
    if trained_base and trained_base != args.model:
        parser.error(f"adapter was trained for {trained_base}, not {args.model}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    selected = choose_rows(load_rows([args.dev], "dev"), args.max_dev_rows)
    pairs = [(row, encoded) for row in selected if
             (encoded := encode(tokenizer, row, args.max_input_length,
                                args.max_target_length)) is not None]
    if not pairs:
        raise ValueError("token limits exclude every selected development row")
    print(f"Selected {len(selected)} development rows; {len(pairs)} eligible.",
          flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16
                              else torch.float16)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto")
    base.eval()
    device = next(base.parameters()).device
    results = []
    for index, (row, encoded) in enumerate(pairs, 1):
        copy = {"raw": None, "prediction": row["source"], "error": None,
                "copy_markers": 0,
                "scores": score(row["source"], row["source"], row["gold"])}
        baseline = generate(base, tokenizer, row, encoded,
                            args.max_target_length, device, torch)
        results.append({"id": row["id"], "source": row["source"],
                        "gold": row["gold"], "evidence_input": row["input"],
                        "copy": copy, "base": baseline})
        print(f"Base: {index}/{len(pairs)}", flush=True)
    editor = PeftModel.from_pretrained(base, str(args.adapter))
    editor.eval()
    for index, (item, (row, encoded)) in enumerate(zip(results, pairs), 1):
        item["editor"] = generate(editor, tokenizer, row, encoded,
                                  args.max_target_length, device, torch)
        print(f"Editor: {index}/{len(pairs)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    predictions = args.output / "dev_predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(x, ensure_ascii=False)
                                     for x in results) + "\n", encoding="utf-8")
    summary = {"model": args.model, "adapter": str(args.adapter),
               "dev_file_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(),
               "max_dev_rows": args.max_dev_rows,
               "max_input_length": args.max_input_length,
               "max_target_length": args.max_target_length,
               "dev_ids": [x["id"] for x in results],
               "conditions": {name: summarize(results, name)
                              for name in ("copy", "base", "editor")},
               "warning": "Silver dev and token-overlap scores do not establish factual support; audit predictions."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
