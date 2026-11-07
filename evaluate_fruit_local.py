"""Paired oracle-locus local editor evaluation on FRUIT silver rows.

Copy, unchanged FLAN-T5-XL and adapted FLAN-T5-XL receive the same examples.
The positive control deliberately evaluates training rows and is not a dev
result. Lexical metrics are diagnostics; factual support needs manual audit.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluate_fruit_gate import lcs_f1, words
from fruit_local import choose, load, prompt, reconstruct
from train_fruit_local import encode


def edit_content_f1(old, prediction, gold):
    """Unordered addition+deletion token F1; not a factual-support metric."""
    source = Counter(words(old))
    predicted = Counter(words(prediction))
    reference = Counter(words(gold))
    pred_added, gold_added = predicted - source, reference - source
    pred_deleted, gold_deleted = source - predicted, source - reference
    matched = sum((pred_added & gold_added).values()) + sum(
        (pred_deleted & gold_deleted).values())
    pred_edits = sum(pred_added.values()) + sum(pred_deleted.values())
    gold_edits = sum(gold_added.values()) + sum(gold_deleted.values())
    return 2 * matched / (pred_edits + gold_edits) if pred_edits + gold_edits else 1.0


def score(row, replacement):
    if not replacement:
        return {"valid": False, "replacement_exact": False,
                "full_exact": False, "replacement_lcs_f1": 0.0,
                "edit_content_f1": 0.0, "changed": False}
    full = reconstruct(row, replacement)
    return {"valid": True, "replacement_exact": replacement == row["replacement"],
            "full_exact": full == row["gold"],
            "replacement_lcs_f1": lcs_f1(replacement, row["replacement"]),
            "edit_content_f1": edit_content_f1(row["old"], replacement,
                                               row["replacement"]),
            "changed": replacement != row["old"]}


def generate(model, tokenizer, encoded, max_target, torch):
    device = next(model.parameters()).device
    inputs = {"input_ids": torch.tensor([encoded["input_ids"]], device=device),
              "attention_mask": torch.tensor([encoded["attention_mask"]], device=device)}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_target,
                                do_sample=False, num_beams=1)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def summary(items, name):
    scores = [x[name]["scores"] for x in items]
    n = len(scores)
    return {"n": n, "replacement_exact": sum(x["replacement_exact"] for x in scores),
            "full_exact": sum(x["full_exact"] for x in scores),
            "valid": sum(x["valid"] for x in scores),
            "changed": sum(x["changed"] for x in scores),
            "mean_replacement_lcs_f1": sum(x["replacement_lcs_f1"] for x in scores) / n,
            "mean_edit_content_f1": sum(x["edit_content_f1"] for x in scores) / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--positive-control", action="store_true")
    args = ap.parse_args()
    if args.max_rows < 1:
        ap.error("max-rows must be positive")
    if not (args.adapter / "adapter_config.json").exists():
        ap.error("adapter_config.json missing")
    protocol_path = args.adapter.parent / "protocol.json"
    if not protocol_path.exists():
        ap.error("training protocol.json missing beside adapter")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["model"] != args.model or protocol["positive_control"] != args.positive_control:
        ap.error("model or positive-control flag disagrees with training protocol")
    expected_hash = protocol["train_sha256" if args.positive_control else "dev_sha256"]
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != expected_hash:
        ap.error("evaluation file hash differs from training protocol")
    split = "train" if args.positive_control else "dev"
    rows = load(args.data, split)
    if args.positive_control:
        by_id = {x["id"]: x for x in rows}
        rows = [by_id[i] for i in protocol["train_ids"]]
    else:
        rows = choose(rows, args.max_rows)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = [(row, encoded) for row in rows if
             (encoded := encode(tokenizer, row, protocol["max_input_length"],
                                protocol["max_target_length"])) is not None]
    if not pairs:
        raise ValueError("no token-eligible evaluation rows")
    if args.positive_control and len(pairs) != len(rows):
        raise ValueError("positive control rows no longer match training encoding")
    print(f"Selected {len(rows)} {split} rows; {len(pairs)} token-eligible", flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                 quantization_config=quant,
                                                 device_map="auto")
    base.eval()
    results = []
    for i, (row, encoded) in enumerate(pairs, 1):
        raw = generate(base, tokenizer, encoded, protocol["max_target_length"], torch)
        results.append({"id": row["id"], "split": split, "oracle_edit_index": row["edit_index"],
                        "old": row["old"], "gold_replacement": row["replacement"],
                        "previous": row["previous"], "next": row["next"],
                        "evidence": row["evidence"], "gold_full": row["gold"],
                        "copy": {"replacement": row["old"], "scores": score(row, row["old"])},
                        "base": {"replacement": raw, "scores": score(row, raw)}})
        print(f"Base: {i}/{len(pairs)}", flush=True)
    editor = PeftModel.from_pretrained(base, str(args.adapter))
    editor.eval()
    for i, (item, (row, encoded)) in enumerate(zip(results, pairs), 1):
        raw = generate(editor, tokenizer, encoded, protocol["max_target_length"], torch)
        item["editor"] = {"replacement": raw, "scores": score(row, raw)}
        print(f"Editor: {i}/{len(pairs)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    report = {"role": "train_overlap_positive_control" if args.positive_control else
              "oracle_locus_silver_development", "model": args.model,
              "adapter": str(args.adapter), "data_sha256": expected_hash,
              "selected": len(rows), "eligible": len(pairs),
              "ids": [x["id"] for x in results],
              "conditions": {name: summary(results, name) for name in ("copy", "base", "editor")},
              "warning": "Gold edit location is supplied. Silver lexical scores do not establish factual support or end-to-end performance."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
