"""Paired copy, oracle-span and end-to-end span-editor evaluation on FRUIT silver.

Oracle conditions use gold edit locations as an upper bound. The pipeline uses
only the source and evidence to locate an edit, then generates the replacement.
Exact match is measured against noisy silver targets, not factual support.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluate_fruit_gate import lcs_f1, words
from fruit_local import collapse, load as load_local
from fruit_span import load, prompt
from span_editor.common import load_protocol, ranked, sha256, write_json
from span_editor.locator import lexical_overlap, locate


def edit_content_f1(old, prediction, gold):
    source, predicted, reference = Counter(words(old)), Counter(words(prediction)), Counter(words(gold))
    pa, ga = predicted - source, reference - source
    pd, gd = source - predicted, source - reference
    match = sum((pa & ga).values()) + sum((pd & gd).values())
    size = sum(pa.values()) + sum(ga.values()) + sum(pd.values()) + sum(gd.values())
    return 2 * match / size if size else 1.0


def reconstruct_at(row, sentence_index, replacement):
    parts = list(row["source_sentences"])
    parts[sentence_index] = collapse(replacement)
    return collapse(" ".join(parts))


def make_prediction(row, sentence_index, start, end, phrase, max_span_chars):
    if (sentence_index is None or start is None or end is None or not phrase or
            len(phrase) > max_span_chars or "\n" in phrase):
        sentence_index, start, end, phrase = None, None, None, ""
        full = row["source"]
        replacement = row["old"]
    else:
        source_sentence = row["source_sentences"][sentence_index]
        if not (0 <= start < end <= len(source_sentence)):
            raise ValueError("invalid predicted span")
        replacement = source_sentence[:start] + phrase + source_sentence[end:]
        full = reconstruct_at(row, sentence_index, replacement)
    correct_sentence = sentence_index == row["edit_index"]
    scores = {"full_exact": full == row["gold"],
              "replacement_exact": correct_sentence and replacement == row["replacement"],
              "changed": full != row["source"],
              "replacement_lcs_f1": lcs_f1(replacement, row["replacement"])
              if correct_sentence or sentence_index is None else 0.0,
              "edit_content_f1": edit_content_f1(row["source"], full, row["gold"])}
    return {"sentence_index": sentence_index, "span_start": start,
            "span_end": end, "phrase": phrase, "replacement": replacement,
            "full": full, "scores": scores}


def summarize(rows, condition):
    items = [row[condition]["scores"] for row in rows]
    n = len(items)
    return {"n": n,
            "full_exact": sum(x["full_exact"] for x in items),
            "replacement_exact": sum(x["replacement_exact"] for x in items),
            "changed": sum(x["changed"] for x in items),
            "mean_replacement_lcs_f1": sum(x["replacement_lcs_f1"] for x in items) / n,
            "mean_edit_content_f1": sum(x["edit_content_f1"] for x in items) / n}


def generate(model, tokenizer, text, max_input, max_target, torch):
    fields = tokenizer(text, add_special_tokens=True, truncation=False)
    if len(fields["input_ids"]) > max_input:
        return None
    device = next(model.parameters()).device
    inputs = {key: torch.tensor([fields[key]], device=device)
              for key in ("input_ids", "attention_mask") if key in fields}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_target,
                                do_sample=False, num_beams=1)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def predicted_row(row, location):
    index = location["sentence_index"]
    sentences = row["source_sentences"]
    # Only source/evidence fields are used in the generation prompt.
    return {"old": sentences[index], "previous": sentences[index - 1] if index else "",
            "next": sentences[index + 1] if index + 1 < len(sentences) else "",
            "evidence": row["evidence"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--locator", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--exclude-local-dev", type=Path,
                    help="Earlier local-dev JSONL; exclude its first hash-ranked rows")
    ap.add_argument("--exclude-local-rows", type=int, default=100)
    ap.add_argument("--max-span-chars", type=int, default=80)
    ap.add_argument("--locator-threshold", type=float, default=0.0,
                    help="Abstain when the best span logit score is below this null-relative threshold")
    ap.add_argument("--positive-control", action="store_true")
    args = ap.parse_args()
    if min(args.max_rows, args.max_span_chars) < 1 or args.skip_rows < 0 or args.exclude_local_rows < 0:
        ap.error("invalid row or span limit")
    if not args.positive_control and args.exclude_local_dev is None:
        ap.error("--exclude-local-dev is required to avoid the previously inspected development IDs")
    if not (args.adapter / "adapter_config.json").is_file():
        ap.error("infiller adapter_config.json missing")
    if args.locator and not (args.locator / "config.json").is_file():
        ap.error("locator model config.json missing")
    protocol = load_protocol(args.adapter)
    if protocol["role"] != "oracle_span_infiller" or protocol["model"] != args.model:
        ap.error("infiller training protocol disagrees with requested model")
    if protocol["positive_control"] != args.positive_control:
        ap.error("positive-control flag disagrees with infiller protocol")
    split = "train" if args.positive_control else "dev"
    if sha256(args.data) != protocol["train_sha256" if args.positive_control else "dev_sha256"]:
        ap.error("data hash differs from infiller training protocol")
    if args.locator:
        locator_protocol_path = args.locator.parent / "protocol.json"
        if not locator_protocol_path.is_file():
            ap.error("locator training protocol missing")
        locator_protocol = json.loads(locator_protocol_path.read_text(encoding="utf-8"))
        if (locator_protocol["role"] != "evidence_conditioned_span_locator" or
                sha256(args.data) != locator_protocol["train_sha256" if args.positive_control else "dev_sha256"]):
            ap.error("locator training data hash or role disagrees")
    all_rows = load(args.data, split)
    if args.positive_control:
        by_id = {x["id"]: x for x in all_rows}
        rows = [by_id[i] for i in protocol["train_ids"]]
        excluded = set()
    else:
        excluded = {x["id"] for x in ranked(load_local(args.exclude_local_dev, "dev"))
                    [:args.exclude_local_rows]}
        rows = ranked([x for x in all_rows if x["id"] not in excluded])[
            args.skip_rows:args.skip_rows + args.max_rows]
    if not rows:
        raise ValueError("no selected evaluation rows")

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                              AutoTokenizer, BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    print(f"LOAD INFILLER: {args.model}", flush=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                 quantization_config=quant, device_map="auto")
    base.eval()
    predictions = []
    for i, row in enumerate(rows, 1):
        item = {"id": row["id"], "split": split, "source": row["source"],
                "evidence": row["evidence"], "gold_full": row["gold"],
                "gold_sentence_index": row["edit_index"],
                "gold_span_start": row["span_start"],
                "gold_span_end": row["span_end"],
                "gold_old_span": row["old_span"], "gold_new_span": row["new_span"],
                "copy": make_prediction(row, None, None, None, "", args.max_span_chars)}
        phrase = generate(base, tokenizer, prompt(row), protocol["max_input_length"],
                          protocol["max_target_length"], torch)
        item["base_oracle"] = make_prediction(row, row["edit_index"],
                                                row["span_start"], row["span_end"],
                                                phrase, args.max_span_chars)
        predictions.append(item)
        print(f"BASE ORACLE: {i}/{len(rows)}", flush=True)
    editor = PeftModel.from_pretrained(base, str(args.adapter))
    editor.eval()
    for i, (row, item) in enumerate(zip(rows, predictions), 1):
        phrase = generate(editor, tokenizer, prompt(row), protocol["max_input_length"],
                          protocol["max_target_length"], torch)
        item["editor_oracle"] = make_prediction(row, row["edit_index"],
                                                  row["span_start"], row["span_end"],
                                                  phrase, args.max_span_chars)
        print(f"EDITOR ORACLE: {i}/{len(rows)}", flush=True)

    if args.locator:
        print(f"LOAD LOCATOR: {locator_protocol['model']}", flush=True)
        locate_tokenizer = AutoTokenizer.from_pretrained(str(args.locator), use_fast=True)
        locate_model = AutoModelForQuestionAnswering.from_pretrained(str(args.locator)).to("cuda")
        locate_model.eval()
        for i, (row, item) in enumerate(zip(rows, predictions), 1):
            location = locate(locate_model, locate_tokenizer, row, torch,
                              locator_protocol["max_length"])
            item["locator"] = location
            if location and location["score_above_null"] >= args.locator_threshold:
                draft = predicted_row(row, location)
                phrase = generate(editor, tokenizer,
                                  prompt(draft, location["start"], location["end"]),
                                  protocol["max_input_length"], protocol["max_target_length"], torch)
                item["pipeline"] = make_prediction(row, location["sentence_index"],
                                                    location["start"], location["end"],
                                                    phrase, args.max_span_chars)
            else:
                item["pipeline"] = make_prediction(row, None, None, None, "",
                                                    args.max_span_chars)
            print(f"PIPELINE: {i}/{len(rows)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for item in predictions:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    names = ["copy", "base_oracle", "editor_oracle"] + (["pipeline"] if args.locator else [])
    report = {"role": "train_overlap_positive_control" if args.positive_control else
              "held_out_silver_span_edit", "model": args.model,
              "adapter": str(args.adapter), "locator": str(args.locator) if args.locator else None,
              "locator_threshold": args.locator_threshold if args.locator else None,
              "data_sha256": sha256(args.data), "selected": len(rows),
              "skip_rows": 0 if args.positive_control else args.skip_rows,
              "exclude_local_dev_sha256": sha256(args.exclude_local_dev)
              if args.exclude_local_dev and not args.positive_control else None,
              "excluded_earlier_ids": len(excluded),
              "ids": [x["id"] for x in predictions],
              "conditions": {name: summarize(predictions, name) for name in names},
              "warning": "Oracle results use gold locations. Silver exact match is not factual support; audit predictions."}
    if args.locator:
        report["locator_scores"] = {
            "n": len(rows),
            "lexical_sentence_baseline_exact": sum(
                max(range(len(row["source_sentences"])),
                    key=lambda i: lexical_overlap(row["source_sentences"][i], row["evidence"]))
                == row["edit_index"] for row in rows),
            "sentence_exact": sum(x["locator"] is not None and
                                  x["locator"]["sentence_index"] == x["gold_sentence_index"]
                                  for x in predictions),
            "span_exact": sum(x["locator"] is not None and
                              x["locator"]["sentence_index"] == x["gold_sentence_index"] and
                              x["locator"]["start"] == x["gold_span_start"] and
                              x["locator"]["end"] == x["gold_span_end"]
                              for x in predictions),
            "no_location": sum(x["locator"] is None for x in predictions)}
        report["locator_scores"]["abstained"] = sum(
            x["pipeline"]["sentence_index"] is None for x in predictions)
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
