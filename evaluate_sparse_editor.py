"""Evaluate predicted edit scripts; preserve invalid outputs for diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

from revproof_data import check_candidate, read_records
from sparse_edit import apply_script, edit_prompt, parse_script


def assess(row, raw):
    result = {"raw_script": raw, "valid_json": False, "applicable": False,
              "prediction": None, "exact_span": False,
              "passes_string_checks": False, "checks": None, "error": None,
              "old_fraction": None}
    try:
        script = parse_script(raw)
        result["valid_json"] = True
        span = apply_script(row["old_span"], script)
        result["applicable"] = True
        result["prediction"] = span
        result["exact_span"] = span == row["new_span"]
        result["old_fraction"] = len(script["old"]) / len(row["old_span"])
        checks = check_candidate(row, span)
        result["checks"] = checks["checks"]
        result["passes_string_checks"] = checks["passes_string_checks"]
    except ValueError as exc:
        result["error"] = str(exc)
    return result


def summarize(items):
    def rates(part):
        return {"n": len(part),
                "json_rate": sum(x["valid_json"] for x in part) / len(part),
                "applicable_rate": sum(x["applicable"] for x in part) / len(part),
                "exact_span_rate": sum(x["exact_span"] for x in part) / len(part),
                "selected_string_check_rate": sum(x["passes_string_checks"] for x in part) / len(part),
                "string_pass_at_n": sum(x["any_candidate_passes_string_checks"] for x in part) / len(part)}
    groups = {}
    for item in items:
        groups.setdefault(item["group"], []).append(item)
    return {**rates(items), "n_groups": len(groups),
            "all_questions_exact_rate": sum(all(x["exact_span"] for x in part)
                                            for part in groups.values()) / len(groups),
            "by_edit_type": {kind: rates(part) for kind in sorted({x["edit_type"] for x in items})
                             if (part := [x for x in items if x["edit_type"] == kind])},
            "by_answer_relation": {kind: rates(part) for kind in ("keep", "change")
                                   if (part := [x for x in items if x["z"] == kind])}}


def load_model(model_name, backend, adapter):
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoTokenizer, BitsAndBytesConfig)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if backend == "causal":
        tokenizer.pad_token = tokenizer.eos_token
        kw = {}
        if torch.cuda.is_available():
            kw = {"device_map": "auto", "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                else torch.float16)}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported() else torch.float32)
        if torch.cuda.is_available():
            model.to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, backend, row, n, temperature, max_new_tokens):
    import torch
    source = edit_prompt(row)
    if backend == "causal":
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": source}], tokenize=False,
            add_generation_prompt=True)
    inputs = tokenizer(source, return_tensors="pt", add_special_tokens=backend == "seq2seq")
    inputs = inputs.to(next(model.parameters()).device)
    sampled = n > 1
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sampled,
              "num_return_sequences": n}
    if sampled:
        kwargs.update({"temperature": temperature, "top_p": 0.9})
    if backend == "causal":
        kwargs["pad_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        ids = model.generate(**inputs, **kwargs)
    if backend == "causal":
        ids = ids[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(x, skip_special_tokens=True).strip() for x in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["causal", "seq2seq"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--filter", action="store_true", help="select first literal-check passing candidate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_candidates < 1 or args.max_new_tokens < 1 or args.temperature <= 0:
        ap.error("candidate count, token budget, and temperature must be positive")
    model_name = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if args.backend == "causal"
                                else "google/flan-t5-large")
    rows = [r for r in read_records(args.data) if r["split"] == args.split]
    if not rows or any(r["source_type"] != "real" or
                       r.get("prompt_version") != "v2_fact_card" for r in rows):
        raise ValueError("evaluation requires nonempty real v2 fact-card rows")
    import torch
    torch.manual_seed(args.seed)
    tokenizer, model = load_model(model_name, args.backend, args.adapter)
    results = []
    for row in rows:
        candidates = [assess(row, raw) for raw in generate(
            tokenizer, model, args.backend, row, args.num_candidates,
            args.temperature, args.max_new_tokens)]
        passing = [i for i, x in enumerate(candidates) if x["passes_string_checks"]]
        selected = passing[0] if args.filter and passing else 0
        chosen = candidates[selected]
        results.append({"id": row["id"], "group": row["group"],
                        "source_type": row["source_type"], "z": row["z"],
                        "edit_type": row["edit_type"], "gold_span": row["new_span"],
                        "candidate_count": len(candidates), "selected_index": selected,
                        "any_candidate_passes_string_checks": bool(passing),
                        "candidates": candidates, **chosen})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n",
                      encoding="utf-8")
    summary = {**summarize(results), "model": model_name, "backend": args.backend,
               "adapter": args.adapter, "filter": args.filter,
               "num_candidates": args.num_candidates, "seed": args.seed,
               "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               "split": args.split,
               "test_status": "previously inspected; exploratory" if args.split == "test" else None}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("String checks and exact match do not establish semantic validity; audit outputs manually.")


if __name__ == "__main__":
    main()
