"""Evaluate a reference-planned span editor on a frozen FRUIT development ID list.

Gold spans and reference IDs are used only for metrics after inference.
"""

import argparse
import json
import math
from pathlib import Path
import re

from fruit_span import prompt
from span_editor.common import sha256, write_json
from span_editor.evaluate import generate, make_prediction, summarize
from span_editor.locator import locate_sentence_candidates
from span_editor.train_infiller_v2 import referenced_view
from span_editor.train_selector import rank_pairs
from span_editor.v2_data import load


_NUMBER = re.compile(r"\d+(?:[./–-]\d+)*")


def inference_view(row):
    """Whitelist fields available at deployment; fail closed against label use."""
    return {"id": row["id"], "source": row["source"],
            "source_sentences": row["source_sentences"],
            "evidence": row["evidence"],
            "evidence_items": row["evidence_items"]}


def evidence_for_candidate(row, pair, ranked_pairs, max_items):
    ids = [pair["evidence_id"]]
    for other in ranked_pairs:
        identifier = other["evidence_id"]
        if (other["sentence_index"] == pair["sentence_index"] and
                identifier not in ids and len(ids) < max_items):
            ids.append(identifier)
    return ids, " ".join(f"({i}) {row['evidence_items'][i]}" for i in ids)


def phrase_checks(phrase, old_span, evidence, source_sentence, max_chars):
    if not phrase or len(phrase) > max_chars or "\n" in phrase:
        return ["empty_or_long_phrase"]
    reasons = []
    if phrase == old_span:
        reasons.append("unchanged_phrase")
    context_numbers = set(_NUMBER.findall(evidence)) | set(_NUMBER.findall(source_sentence))
    if any(number not in context_numbers for number in _NUMBER.findall(phrase)):
        reasons.append("new_number_absent_from_context")
    return reasons


def choose_candidate(candidates, min_selector_prob, min_span_score):
    eligible = [x for x in candidates if not x["checks"]]
    eligible.sort(key=lambda x: (-x["rank_score"], x["pair_rank"],
                                 x["span_rank"]))
    forced = eligible[0] if eligible else None
    gated = forced if (forced and forced["selector_probability"] >= min_selector_prob
                       and forced["score_above_null"] >= min_span_score) else None
    return forced, gated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-sentences", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=6)
    parser.add_argument("--top-spans", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-span-chars", type=int, default=80)
    parser.add_argument("--infiller-evidence-items", type=int, default=3)
    parser.add_argument("--min-selector-prob", type=float, default=0.5)
    parser.add_argument("--min-span-score", type=float, default=0.0)
    parser.add_argument("--locator-weight", type=float, default=0.1)
    args = parser.parse_args()
    if min(args.max_source_sentences, args.top_pairs, args.top_spans,
           args.max_candidates, args.max_span_chars,
           args.infiller_evidence_items) < 1:
        parser.error("candidate limits must be positive")
    if not 0 <= args.min_selector_prob <= 1 or args.locator_weight < 0:
        parser.error("invalid selector probability or locator weight")
    protocols = {}
    expected = {"selector": "reference_supervised_sentence_evidence_selector",
                "locator": "evidence_item_span_locator_v2",
                "adapter": "reference_conditioned_span_infiller_v2"}
    for name in expected:
        path = getattr(args, name)
        file = path.parent / "protocol.json"
        model_file = path / ("adapter_config.json" if name == "adapter" else "config.json")
        if not file.is_file() or not model_file.is_file():
            parser.error(f"missing {name} trained model or protocol")
        protocols[name] = json.loads(file.read_text(encoding="utf-8"))
        if protocols[name].get("role") != expected[name] or \
           protocols[name].get("dev_sha256") != sha256(args.data):
            parser.error(f"{name} role or development data hash mismatch")
    if len({p["train_sha256"] for p in protocols.values()}) != 1:
        parser.error("components were trained on different data versions")
    frozen = json.loads(args.ids_file.read_text(encoding="utf-8"))
    ids = frozen["ids"]
    if (frozen.get("role") != "future_development_slice_not_yet_evaluated" or
            len(ids) != len(set(ids)) or not ids):
        parser.error("expected a nonempty frozen future-development ID list")
    by_id = {x["id"]: x for x in load(args.data, "dev")}
    if any(i not in by_id for i in ids):
        parser.error("frozen ID missing from development file")
    rows = [by_id[i] for i in ids]

    run_protocol = {"role": "frozen_silver_span_v2", "ids": ids,
                    "frozen_ids_sha256": sha256(args.ids_file),
                    "dev_sha256": sha256(args.data),
                    "models": {name: str(getattr(args, name)) for name in expected},
                    "model_protocols": protocols,
                    "settings": {"max_source_sentences": args.max_source_sentences,
                                 "top_pairs": args.top_pairs,
                                 "top_spans": args.top_spans,
                                 "max_candidates": args.max_candidates,
                                 "max_span_chars": args.max_span_chars,
                                 "infiller_evidence_items": args.infiller_evidence_items,
                                 "min_selector_prob": args.min_selector_prob,
                                 "min_span_score": args.min_span_score,
                                 "locator_weight": args.locator_weight}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / "evaluation_protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding="utf-8")) != run_protocol:
            parser.error("output directory contains a different evaluation protocol")
    else:
        write_json(protocol_file, run_protocol)
    row_dir = args.output / "completed_rows"
    row_dir.mkdir(exist_ok=True)
    completed = {}
    for index, row in enumerate(rows):
        file = row_dir / f"{index:04d}.json"
        if file.exists():
            result = json.loads(file.read_text(encoding="utf-8"))
            if (result.get("id") != row["id"] or
                    result.get("source") != row["source"] or
                    result.get("gold_full") != row["gold"]):
                parser.error(f"completed row mismatch: {file}")
            completed[index] = result
    if len(completed) == len(rows):
        print(f"RESUME: {len(completed)}/{len(rows)} rows already complete", flush=True)
    elif completed:
        print(f"RESUME: {len(completed)}/{len(rows)} rows already complete", flush=True)

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    selector_tokenizer = AutoTokenizer.from_pretrained(str(args.selector), use_fast=True)
    selector = AutoModelForSequenceClassification.from_pretrained(
        str(args.selector)).to("cuda").eval()
    locator_tokenizer = AutoTokenizer.from_pretrained(str(args.locator), use_fast=True)
    locator = AutoModelForQuestionAnswering.from_pretrained(
        str(args.locator)).to("cuda").eval()
    model_name = protocols["adapter"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                 quantization_config=quant,
                                                 device_map="auto")
    editor = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    predictions = []
    for n, gold in enumerate(rows, 1):
        if n - 1 in completed:
            predictions.append(completed[n - 1])
            continue
        source = inference_view(gold)
        pairs = rank_pairs(selector, selector_tokenizer, source, torch,
                           args.max_source_sentences, args.top_pairs,
                           protocols["selector"]["max_length"])
        candidates = []
        for pair_rank, pair in enumerate(pairs):
            if len(candidates) >= args.max_candidates:
                break
            evidence_id = pair["evidence_id"]
            sentence_index = pair["sentence_index"]
            local = {**source, "evidence": source["evidence_items"][evidence_id]}
            locations = locate_sentence_candidates(
                locator, locator_tokenizer, local, sentence_index, torch,
                protocols["locator"]["max_length"], args.top_spans)
            for span_rank, location in enumerate(locations):
                if len(candidates) >= args.max_candidates:
                    break
                evidence_ids, evidence = evidence_for_candidate(
                    source, pair, pairs, args.infiller_evidence_items)
                context = source["source_sentences"]
                draft = {"old": context[sentence_index],
                         "previous": context[sentence_index - 1] if sentence_index else "",
                         "next": context[sentence_index + 1]
                         if sentence_index + 1 < len(context) else "",
                         "evidence": evidence}
                phrase = generate(editor, tokenizer,
                                  prompt(draft, location["start"], location["end"]),
                                  protocols["adapter"]["max_input_length"],
                                  protocols["adapter"]["max_target_length"], torch)
                checks = phrase_checks(phrase, location["old_span"], evidence,
                                       context[sentence_index], args.max_span_chars)
                probability = pair["selector_probability"]
                rank_score = (math.log(max(probability, 1e-6) /
                                       max(1 - probability, 1e-6)) +
                              args.locator_weight * location["score_above_null"])
                candidates.append({**location, "evidence_id": evidence_id,
                                   "infiller_evidence_ids": evidence_ids,
                                   "selector_probability": probability,
                                   "pair_rank": pair_rank, "span_rank": span_rank,
                                   "phrase": phrase, "checks": checks,
                                   "rank_score": rank_score})
        forced, gated = choose_candidate(candidates, args.min_selector_prob,
                                         args.min_span_score)
        def prediction(candidate):
            if candidate is None:
                return make_prediction(gold, None, None, None, "",
                                       args.max_span_chars)
            return make_prediction(gold, candidate["sentence_index"],
                                   candidate["start"], candidate["end"],
                                   candidate["phrase"], args.max_span_chars)
        oracle_view = referenced_view(gold, protocols["adapter"]["max_reference_items"])
        oracle_phrase = (generate(editor, tokenizer, prompt(oracle_view),
                                  protocols["adapter"]["max_input_length"],
                                  protocols["adapter"]["max_target_length"], torch)
                         if oracle_view else None)
        oracle = make_prediction(gold, gold["edit_index"], gold["span_start"],
                                 gold["span_end"], oracle_phrase,
                                 args.max_span_chars)
        result = {"id": gold["id"], "source": gold["source"],
                            "evidence": gold["evidence"], "gold_full": gold["gold"],
                            "gold_sentence_index": gold["edit_index"],
                            "gold_span_start": gold["span_start"],
                            "gold_span_end": gold["span_end"],
                            "gold_new_span": gold["new_span"],
                            "gold_reference_ids": gold["reference_ids"],
                            "ranked_pairs": pairs, "candidates": candidates,
                            "v2_oracle_gold_location_and_references": oracle,
                            "v2_forced": prediction(forced),
                            "v2_gated": prediction(gated)}
        file = row_dir / f"{n - 1:04d}.json"
        temporary = file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(file)
        predictions.append(result)
        print(f"V2 EVALUATE: {n}/{len(rows)}; pairs={len(pairs)}; "
              f"candidates={len(candidates)}", flush=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in predictions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {k: v for k, v in run_protocol.items() if k != "model_protocols"}
    report.update({
              "conditions": {name: summarize(predictions, name)
                             for name in ("v2_oracle_gold_location_and_references",
                                          "v2_forced", "v2_gated")},
              "candidate_diagnostics": {
                  "silver_reference_pair_at_k": sum(any(
                      p["sentence_index"] == x["gold_sentence_index"] and
                      p["evidence_id"] in x["gold_reference_ids"]
                      for p in x["ranked_pairs"]) for x in predictions),
                  "gold_sentence_in_pairs": sum(any(
                      p["sentence_index"] == x["gold_sentence_index"]
                      for p in x["ranked_pairs"]) for x in predictions),
                  "gold_span_in_candidates": sum(any(
                      c["sentence_index"] == x["gold_sentence_index"] and
                      c["start"] == x["gold_span_start"] and
                      c["end"] == x["gold_span_end"]
                      for c in x["candidates"]) for x in predictions),
                  "no_valid_candidate": sum(not any(not c["checks"]
                                                    for c in x["candidates"])
                                            for x in predictions)},
              "warning": "Reference labels and exact text are silver; manually audit factual support."})
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
