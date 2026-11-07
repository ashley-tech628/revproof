"""Dev-only diagnostic: generation when the gold edit location is supplied.

The location is derived from the target paragraph. Scores are an oracle-location
upper bound and must not be compared as a deployable method.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate_sparse_editor import load_model
from revproof_data import check_candidate, read_records
from sparse_edit import gold_script, oracle_infill_prompt


def assess_infill(row, replacement):
    script = gold_script(row["old_span"], row["new_span"])
    anchor = script["old"]
    start = row["old_span"].index(anchor)
    prediction = row["old_span"][:start] + replacement + row["old_span"][start + len(anchor):]
    checks = check_candidate(row, prediction)
    return {
        "oracle_old_anchor": anchor,
        "gold_replacement": script["new"],
        "raw_replacement": replacement,
        "prediction": prediction,
        "exact_replacement": replacement == script["new"],
        "exact_span": prediction == row["new_span"],
        "passes_string_checks": checks["passes_string_checks"],
        "checks": checks["checks"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["seq2seq", "causal"], default="seq2seq")
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_new_tokens < 1:
        ap.error("max-new-tokens must be positive")
    model_name = args.model or ("google/flan-t5-large" if args.backend == "seq2seq"
                                else "Qwen/Qwen2.5-1.5B-Instruct")
    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    rows = [r for r in read_records(args.data) if r["split"] == "dev"]
    if not rows or any(r["source_type"] != "real" or
                       r.get("prompt_version") != "v2_fact_card" for r in rows):
        raise ValueError("requires nonempty real v2 fact-card dev rows")
    if args.adapter:
        metadata = json.loads((Path(args.adapter).parent / "run.json").read_text(encoding="utf-8"))
        if (metadata["objective"] != "oracle_infill" or
                metadata["data_sha256"] != data_hash or
                metadata["model"] != model_name or
                metadata["backend"] != args.backend):
            raise ValueError("adapter objective, data, or model does not match oracle protocol")

    import torch
    torch.manual_seed(args.seed)
    tokenizer, model = load_model(model_name, args.backend, args.adapter)
    results = []
    for row in rows:
        source = oracle_infill_prompt(row)
        if args.backend == "causal":
            source = tokenizer.apply_chat_template(
                [{"role": "user", "content": source}], tokenize=False,
                add_generation_prompt=True)
        inputs = tokenizer(source, return_tensors="pt",
                           add_special_tokens=args.backend == "seq2seq")
        inputs = inputs.to(next(model.parameters()).device)
        kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
        if args.backend == "causal":
            kwargs["pad_token_id"] = tokenizer.eos_token_id
        with torch.no_grad():
            ids = model.generate(**inputs, **kwargs)
        if args.backend == "causal":
            ids = ids[:, inputs["input_ids"].shape[1]:]
        replacement = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        result = assess_infill(row, replacement)
        results.append({
            "id": row["id"], "group": row["group"], "z": row["z"],
            "edit_type": row["edit_type"], "gold_span": row["new_span"],
            "oracle_location": True, "location_source": "gold_new_span",
            **result,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n",
                      encoding="utf-8")
    groups = {}
    for item in results:
        groups.setdefault(item["group"], []).append(item)
    n = len(results)
    summary = {
        "n": n, "n_groups": len(groups), "split": "dev",
        "exact_replacement_rate": sum(x["exact_replacement"] for x in results) / n,
        "exact_span_rate": sum(x["exact_span"] for x in results) / n,
        "selected_string_check_rate": sum(x["passes_string_checks"] for x in results) / n,
        "all_questions_exact_rate": sum(all(x["exact_span"] for x in group)
                                        for group in groups.values()) / len(groups),
        "model": model_name, "backend": args.backend, "adapter": args.adapter,
        "data_sha256": data_hash, "seed": args.seed,
        "oracle_location": True, "location_source": "gold_new_span",
        "interpretation": "Gold target locates edit; diagnostic upper bound, not deployable.",
        "test_policy": "Previously inspected test is excluded.",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Oracle target location leaks into dev input; do not report as fair model accuracy.")


if __name__ == "__main__":
    main()
