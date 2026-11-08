"""Probe v3 span recall on previously inspected development IDs only."""

import argparse
import json
from pathlib import Path

from span_editor.boundary_v3 import merge_candidates, propose
from span_editor.common import sha256, write_json
from span_editor.evaluate_v2 import inference_view
from span_editor.v2_data import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seen-summary", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-sentences", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=6)
    parser.add_argument("--top-spans", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-variants", type=int, default=36)
    parser.add_argument("--max-per-pair", type=int, default=2)
    args = parser.parse_args()
    if min(args.top_sentences, args.top_pairs, args.top_spans,
           args.max_candidates, args.radius, args.max_variants,
           args.max_per_pair) < 1:
        parser.error("candidate limits must be positive")
    seen = json.loads(args.seen_summary.read_text(encoding="utf-8"))
    if seen.get("role") != "held_out_silver_span_edit":
        parser.error("expected already inspected v1 development IDs")
    ids = seen["ids"]
    if not ids or len(ids) != len(set(ids)):
        parser.error("invalid seen IDs")
    by_id = {row["id"]: row for row in load(args.data, "dev")}
    if any(identifier not in by_id for identifier in ids):
        parser.error("seen ID missing from development data")
    expected = {"selector": "reference_supervised_sentence_evidence_selector",
                "locator": "evidence_item_span_locator_v2",
                "boundary": "evidence_conditioned_boundary_ranker_v3"}
    protocols = {}
    for name, role in expected.items():
        path = getattr(args, name)
        protocol_file = path.parent / "protocol.json"
        if not (path / "config.json").is_file() or not protocol_file.is_file():
            parser.error(f"missing {name} trained model or protocol")
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if protocol.get("role") != role or protocol.get("dev_sha256") != sha256(args.data):
            parser.error(f"{name} model or data mismatch")
        protocols[name] = protocol
    if len({x["train_sha256"] for x in protocols.values()}) != 1:
        parser.error("training data versions differ")
    if (args.radius != protocols["boundary"]["radius"] or
            protocols["boundary"]["max_span_chars"] != 80):
        parser.error("boundary generation settings differ from training")

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
    boundary_tokenizer = AutoTokenizer.from_pretrained(str(args.boundary), use_fast=True)
    boundary = AutoModelForSequenceClassification.from_pretrained(
        str(args.boundary)).to("cuda").eval()
    details = []
    for n, identifier in enumerate(ids, 1):
        gold = by_id[identifier]
        pairs, candidates, diagnostic = propose(
            boundary, boundary_tokenizer, selector, selector_tokenizer,
            locator, locator_tokenizer, inference_view(gold), torch,
            protocols["selector"]["max_length"],
            protocols["locator"]["max_length"],
            protocols["boundary"]["max_length"],
            args.top_sentences, args.top_pairs, args.top_spans,
            args.radius, args.max_variants, args.max_candidates,
            80, args.max_per_pair, return_diagnostics=True)
        def has_gold(items):
            return any(x["sentence_index"] == gold["edit_index"] and
                       x["start"] == gold["span_start"] and
                       x["end"] == gold["span_end"] for x in items)
        baseline = diagnostic["v2_candidates"]
        hybrid = merge_candidates(baseline, candidates,
                                  args.max_candidates // 2,
                                  args.max_candidates)
        details.append({"id": identifier,
                        "gold_sentence_in_pairs": any(
                            x["sentence_index"] == gold["edit_index"] for x in pairs),
                        "gold_span_in_candidates": has_gold(candidates),
                        "gold_span_in_v2_candidates": has_gold(baseline),
                        "gold_span_in_pool": has_gold(diagnostic["pool_locations"]),
                        "gold_span_in_hybrid": has_gold(hybrid),
                        "gold_span_in_union16": has_gold(baseline + candidates),
                        "candidate_count": len(candidates),
                        "pool_count": len(diagnostic["pool_locations"])})
        print(f"PROBE V3: {n}/{len(ids)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"role": "seen_development_boundary_recall_diagnostic",
               "n": len(ids), "seen_ids": ids,
               "dev_sha256": sha256(args.data),
               "models": {name: str(getattr(args, name)) for name in expected},
               "settings": {"top_sentences": args.top_sentences,
                            "top_pairs": args.top_pairs,
                            "top_spans": args.top_spans,
                            "max_candidates": args.max_candidates,
                            "radius": args.radius,
                            "max_variants": args.max_variants,
                            "max_per_pair": args.max_per_pair},
               "gold_sentence_in_pairs": sum(x["gold_sentence_in_pairs"]
                                             for x in details),
               "gold_span_in_candidates": sum(x["gold_span_in_candidates"]
                                              for x in details),
               "gold_span_in_v2_candidates": sum(
                   x["gold_span_in_v2_candidates"] for x in details),
               "gold_span_in_pool": sum(x["gold_span_in_pool"] for x in details),
               "gold_span_in_hybrid": sum(x["gold_span_in_hybrid"]
                                          for x in details),
               "gold_span_in_union16": sum(x["gold_span_in_union16"]
                                           for x in details),
               "v3_only_gold_span": sum(x["gold_span_in_candidates"] and not
                                        x["gold_span_in_v2_candidates"]
                                        for x in details),
               "v2_only_gold_span": sum(x["gold_span_in_v2_candidates"] and not
                                        x["gold_span_in_candidates"]
                                        for x in details),
               "warning": "Candidate recall is an upper bound; no replacement is generated."}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
