"""Probe v2 candidate recall on the previously inspected FRUIT dev slice.

This uses gold locations and publisher references only for *metrics* after
label-free selector and locator inference. Never use the frozen next 100 IDs
for this diagnostic or for choosing training hyperparameters.
"""

import argparse
import json
from pathlib import Path

from span_editor.common import sha256, write_json
from span_editor.evaluate_v2 import inference_view
from span_editor.locator import locate_sentence_candidates
from span_editor.train_selector import rank_pairs
from span_editor.v2_data import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seen-summary", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-sentences", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=6)
    parser.add_argument("--top-spans", type=int, default=3)
    args = parser.parse_args()
    if min(args.top_sentences, args.top_pairs, args.top_spans) < 1:
        parser.error("top-k values must be positive")
    seen = json.loads(args.seen_summary.read_text(encoding="utf-8"))
    if seen.get("role") != "held_out_silver_span_edit":
        parser.error("expected the already inspected v1 development summary")
    ids = seen["ids"]
    if len(ids) != len(set(ids)) or not ids:
        parser.error("invalid seen ID list")
    by_id = {row["id"]: row for row in load(args.data, "dev")}
    if any(identifier not in by_id for identifier in ids):
        parser.error("seen ID is missing from v2 data")
    protocols = {}
    for name in ("selector", "locator"):
        path = getattr(args, name)
        config = path / "config.json"
        protocol_file = path.parent / "protocol.json"
        if not config.is_file() or not protocol_file.is_file():
            parser.error(f"missing trained {name} model or protocol")
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if protocol.get("dev_sha256") != sha256(args.data):
            parser.error(f"{name} development data hash mismatch")
        protocols[name] = protocol
    if protocols["selector"].get("role") != "reference_supervised_sentence_evidence_selector":
        parser.error("wrong selector role")
    if protocols["locator"].get("role") != "evidence_item_span_locator_v2":
        parser.error("wrong locator role")
    if protocols["selector"]["train_sha256"] != protocols["locator"]["train_sha256"]:
        parser.error("component training data versions differ")

    import torch
    from transformers import (AutoModelForQuestionAnswering,
                              AutoModelForSequenceClassification, AutoTokenizer)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    selector_tokenizer = AutoTokenizer.from_pretrained(str(args.selector), use_fast=True)
    selector = AutoModelForSequenceClassification.from_pretrained(
        str(args.selector)).to("cuda").eval()
    locator_tokenizer = AutoTokenizer.from_pretrained(str(args.locator), use_fast=True)
    locator = AutoModelForQuestionAnswering.from_pretrained(
        str(args.locator)).to("cuda").eval()
    details = []
    for n, identifier in enumerate(ids, 1):
        gold = by_id[identifier]
        source = inference_view(gold)
        pairs = rank_pairs(selector, selector_tokenizer, source, torch,
                           args.top_sentences, args.top_pairs,
                           protocols["selector"]["max_length"])
        candidates = []
        for pair in pairs:
            local = {**source, "evidence": source["evidence_items"][pair["evidence_id"]]}
            candidates.extend({**candidate, "evidence_id": pair["evidence_id"]}
                              for candidate in locate_sentence_candidates(
                                  locator, locator_tokenizer, local,
                                  pair["sentence_index"], torch,
                                  protocols["locator"]["max_length"], args.top_spans))
        details.append({"id": identifier,
                        "gold_sentence_in_pairs": any(
                            x["sentence_index"] == gold["edit_index"] for x in pairs),
                        "silver_reference_pair_in_pairs": any(
                            x["sentence_index"] == gold["edit_index"] and
                            x["evidence_id"] in gold["reference_ids"] for x in pairs),
                        "gold_span_in_candidates": any(
                            x["sentence_index"] == gold["edit_index"] and
                            x["start"] == gold["span_start"] and
                            x["end"] == gold["span_end"] for x in candidates),
                        "pair_count": len(pairs), "candidate_count": len(candidates)})
        print(f"PROBE V2: {n}/{len(ids)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    report = {"role": "seen_development_candidate_recall_diagnostic",
              "n": len(details), "seen_ids_sha256": sha256(args.seen_summary),
              "dev_sha256": sha256(args.data),
              "top_sentences": args.top_sentences, "top_pairs": args.top_pairs,
              "top_spans": args.top_spans,
              "gold_sentence_in_pairs": sum(x["gold_sentence_in_pairs"] for x in details),
              "silver_reference_pair_in_pairs": sum(
                  x["silver_reference_pair_in_pairs"] for x in details),
              "gold_span_in_candidates": sum(
                  x["gold_span_in_candidates"] for x in details),
              "warning": "Candidate recall is an upper bound; no replacement is generated."}
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
