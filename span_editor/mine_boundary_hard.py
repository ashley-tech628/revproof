"""Mine selector/QA boundary mistakes from training rows for hard-negative training.

Inference inputs are whitelisted before selector and QA calls. Silver gold
labels are consulted only afterward to select negatives for training.
"""

import argparse
import json
from pathlib import Path

from span_editor.boundary_v3 import nearby_spans
from span_editor.common import ranked, sha256, write_json
from span_editor.evaluate_v2 import inference_view
from span_editor.locator import locate_sentence_candidates
from span_editor.train_selector import rank_pairs
from span_editor.v2_data import load


def collect_proposals(selector, selector_tokenizer, locator,
                      locator_tokenizer, source, torch, selector_length,
                      locator_length, top_sentences=4, top_pairs=6,
                      top_spans=3, radius=3, max_variants=36,
                      max_span_chars=80):
    """Collect inference-time proposals without seeing a gold field."""
    pairs = rank_pairs(selector, selector_tokenizer, source, torch,
                       top_sentences, top_pairs, selector_length)
    proposals = {}
    for pair_rank, pair in enumerate(pairs):
        sentence_index = pair["sentence_index"]
        evidence_id = pair["evidence_id"]
        sentence = source["source_sentences"][sentence_index]
        local = {**source, "evidence": source["evidence_items"][evidence_id]}
        locations = locate_sentence_candidates(
            locator, locator_tokenizer, local, sentence_index, torch,
            locator_length, top_spans)
        for span_rank, location in enumerate(locations):
            for a, b in nearby_spans(sentence, location["start"],
                                     location["end"], radius,
                                     max_span_chars, max_variants):
                key = (sentence_index, evidence_id, a, b)
                item = {"sentence_index": sentence_index,
                        "evidence_id": evidence_id, "start": a, "end": b,
                        "pair_rank": pair_rank, "span_rank": span_rank}
                if key not in proposals or (pair_rank, span_rank) < (
                        proposals[key]["pair_rank"], proposals[key]["span_rank"]):
                    proposals[key] = item
    return list(proposals.values())


def select_hard_negatives(row, proposals, limit=12):
    """Sample boundary, evidence, and wrong-sentence errors from model proposals."""
    gold = row["edit_index"], row["span_start"], row["span_end"]
    refs = set(row["reference_ids"])
    groups = [[], [], [], []]
    for item in proposals:
        sentence, a, b = (item["sentence_index"], item["start"], item["end"])
        evidence = item["evidence_id"]
        if (sentence, a, b) == gold and evidence in refs:
            continue
        if sentence != gold[0]:
            group = 3
        elif evidence not in refs:
            group = 2
        elif a < gold[2] and b > gold[1]:
            group = 0
        else:
            group = 1
        distance = abs(a - gold[1]) + abs(b - gold[2]) if sentence == gold[0] else 9999
        groups[group].append((item["pair_rank"], item["span_rank"],
                              distance, sentence, a, b, evidence, item))
    selected, seen = [], set()
    def add(items, count):
        for entry in sorted(items):
            if len(selected) >= count:
                break
            item = entry[-1]
            key = (item["sentence_index"], item["evidence_id"],
                   item["start"], item["end"])
            if key in seen:
                continue
            selected.append({k: item[k] for k in ("sentence_index", "evidence_id",
                                                   "start", "end")})
            seen.add(key)
            if len(selected) >= count:
                break
    for group, fraction in zip(groups, (4, 3, 3, 2)):
        add(group, min(limit, len(selected) + fraction))
    if len(selected) < limit:
        add([entry for group in groups for entry in group], limit)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--locator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--top-sentences", type=int, default=4)
    parser.add_argument("--top-pairs", type=int, default=6)
    parser.add_argument("--top-spans", type=int, default=3)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-variants", type=int, default=36)
    parser.add_argument("--max-negatives", type=int, default=12)
    args = parser.parse_args()
    if min(args.max_train_rows, args.top_sentences, args.top_pairs,
           args.top_spans, args.radius, args.max_variants,
           args.max_negatives) < 1:
        parser.error("limits must be positive")
    train = ranked(load(args.train, "train"))[:args.max_train_rows]
    dev = load(args.dev, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("training/development IDs overlap")
    expected = {"selector": "reference_supervised_sentence_evidence_selector",
                "locator": "evidence_item_span_locator_v2"}
    protocols = {}
    for name, role in expected.items():
        model = getattr(args, name)
        protocol_file = model.parent / "protocol.json"
        if not (model / "config.json").is_file() or not protocol_file.is_file():
            parser.error(f"missing {name} trained model or protocol")
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if (protocol.get("role") != role or
                protocol.get("train_sha256") != sha256(args.train) or
                protocol.get("dev_sha256") != sha256(args.dev)):
            parser.error(f"{name} role or data mismatch")
        protocols[name] = protocol
    run_protocol = {"role": "training_only_qa_proposal_hard_negatives",
                    "train_sha256": sha256(args.train),
                    "dev_sha256": sha256(args.dev),
                    "train_ids": [x["id"] for x in train],
                    "selector_protocol_sha256": sha256(args.selector.parent / "protocol.json"),
                    "locator_protocol_sha256": sha256(args.locator.parent / "protocol.json"),
                    "settings": {"top_sentences": args.top_sentences,
                                 "top_pairs": args.top_pairs,
                                 "top_spans": args.top_spans,
                                 "radius": args.radius,
                                 "max_variants": args.max_variants,
                                 "max_negatives": args.max_negatives,
                                 "max_span_chars": 80}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output / "protocol.json"
    if protocol_file.is_file():
        if json.loads(protocol_file.read_text(encoding="utf-8")) != run_protocol:
            parser.error("mining output has a different protocol")
    else:
        write_json(protocol_file, run_protocol)
    row_dir = args.output / "completed_rows"
    row_dir.mkdir(exist_ok=True)

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
    for index, row in enumerate(train):
        path = row_dir / f"{index:05d}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("id") != row["id"]:
                parser.error(f"completed training row mismatch: {path}")
        else:
            source = inference_view(row)
            proposals = collect_proposals(
                selector, selector_tokenizer, locator, locator_tokenizer,
                source, torch, protocols["selector"]["max_length"],
                protocols["locator"]["max_length"], args.top_sentences,
                args.top_pairs, args.top_spans, args.radius,
                args.max_variants)
            negatives = select_hard_negatives(row, proposals,
                                              args.max_negatives)
            result = {"id": row["id"], "negatives": negatives,
                      "proposal_count": len(proposals),
                      "gold_span_in_pool": any(
                          x["sentence_index"] == row["edit_index"] and
                          x["start"] == row["span_start"] and
                          x["end"] == row["span_end"] for x in proposals)}
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result), encoding="utf-8")
            temporary.replace(path)
        details.append(result)
        if (index + 1) % 100 == 0 or index + 1 == len(train):
            print(f"MINE BOUNDARY: {index + 1}/{len(train)}, "
                  f"pool_gold={sum(x['gold_span_in_pool'] for x in details)}",
                  flush=True)
    with (args.output / "negatives.jsonl").open("w", encoding="utf-8") as stream:
        for item in details:
            stream.write(json.dumps(item) + "\n")
    write_json(args.output / "summary.json",
               {"role": run_protocol["role"], "n": len(details),
                "train_sha256": run_protocol["train_sha256"],
                "gold_span_in_pool": sum(x["gold_span_in_pool"] for x in details),
                "negative_examples": sum(len(x["negatives"]) for x in details)})


if __name__ == "__main__":
    main()
