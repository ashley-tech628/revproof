"""Evaluate local document edits at a fixed candidate budget."""
import argparse
import json
from pathlib import Path

from revproof_data import check_candidate, prompt, read_records, template_edit


def load_model(name, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    kw = {}
    if torch.cuda.is_available():
        kw = {"device_map": "auto", "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate(tok, model, row, max_new_tokens, n, temperature):
    import torch
    text = tok.apply_chat_template([{"role": "user", "content": prompt(row)}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    sampling = n > 1
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=sampling, num_return_sequences=n,
                             temperature=temperature if sampling else None,
                             top_p=0.9 if sampling else None,
                             pad_token_id=tok.eos_token_id)
    length = inputs["input_ids"].shape[1]
    return [tok.decode(output[length:], skip_special_tokens=True).strip() for output in ids]


def summarize(results):
    def rates(part):
        return {"n": len(part),
                "exact_span_rate": sum(x["exact_span"] for x in part) / len(part),
                "selected_string_check_rate": sum(x["passes_string_checks"] for x in part) / len(part),
                "string_pass_at_n": sum(x["any_candidate_passes_string_checks"] for x in part) / len(part),
                "mean_candidates_generated": sum(x["candidate_count"] for x in part) / len(part)}
    return {**rates(results),
            "source_type": sorted({x["source_type"] for x in results}),
            "by_answer_relation": {z: rates(part) for z in ("keep", "change")
                                   if (part := [x for x in results if x["z"] == z])},
            "by_edit_type": {t: rates(part) for t in sorted({x["edit_type"] for x in results})
                             if (part := [x for x in results if x["edit_type"] == t])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--filter", action="store_true", help="choose first candidate passing string checks")
    ap.add_argument("--template-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_candidates < 1 or args.temperature <= 0:
        raise ValueError("num-candidates and temperature must be positive")
    if args.template_only and args.num_candidates != 1:
        raise ValueError("template baseline has one deterministic candidate")
    rows = [r for r in read_records(args.data) if r["split"] == args.split]
    if not rows:
        raise ValueError(f"empty {args.split} split")
    tok = model = None
    if not args.template_only:
        import torch
        torch.manual_seed(args.seed)
        tok, model = load_model(args.model, args.adapter)
    results = []
    for row in rows:
        spans = ([template_edit(row)] if args.template_only else
                 generate(tok, model, row, args.max_new_tokens, args.num_candidates, args.temperature))
        assessments = [check_candidate(row, span) for span in spans]
        passing = [i for i, a in enumerate(assessments) if a["passes_string_checks"]]
        selected = passing[0] if args.filter and passing else 0
        assessment = assessments[selected]
        results.append({"id": row["id"], "group": row["group"],
                        "source_type": row["source_type"], "z": row["z"],
                        "edit_type": row["edit_type"], "prediction": spans[selected],
                        "gold_span": row["new_span"], "exact_span": spans[selected] == row["new_span"],
                        "candidate_count": len(spans), "selected_index": selected,
                        "any_candidate_passes_string_checks": bool(passing),
                        "candidates": [{"span": s, "checks": a["checks"],
                                        "passes_string_checks": a["passes_string_checks"]}
                                       for s, a in zip(spans, assessments)],
                        **assessment})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    summary = summarize(results)
    summary.update({"model": "template" if args.template_only else args.model,
                    "adapter": args.adapter, "filter": args.filter, "seed": args.seed})
    (output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("String checks and exact match do not establish semantic validity; audit predictions manually.")


if __name__ == "__main__":
    main()
