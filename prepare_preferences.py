"""Construct same-prompt DPO pairs only from manually audited candidates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from revproof_data import prompt, read_records


def parse_label(value):
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"human_valid must be true or false, got {value!r}")


def build_pairs(rows, audit_rows, max_length_ratio=2.0):
    by_id = {row["id"]: row for row in rows}
    labeled = defaultdict(lambda: {True: [], False: []})
    seen = set()
    for audit in audit_rows:
        key = (audit["id"], audit["candidate_index"])
        if key in seen:
            raise ValueError(f"duplicate audit candidate: {key}")
        seen.add(key)
        if audit["id"] not in by_id:
            raise ValueError(f"unknown row id: {audit['id']}")
        row = by_id[audit["id"]]
        if row["split"] != "train":
            raise ValueError(f"preference data must come from train split: {row['id']}")
        if row["source_type"] != "real":
            raise ValueError(f"synthetic toy row cannot enter research preferences: {row['id']}")
        label = parse_label(audit["human_valid"])
        span = audit["candidate_span"].strip()
        if not span:
            continue
        labeled[row["id"]][label].append(span)
    pairs = []
    for row_id, choices in labeled.items():
        row = by_id[row_id]
        positives = list(dict.fromkeys(choices[True] + [row["new_span"]]))
        negatives = list(dict.fromkeys(choices[False]))
        for negative in negatives:
            positive = min(positives, key=lambda x: abs(len(x) - len(negative)))
            ratio = max(len(positive), len(negative)) / max(1, min(len(positive), len(negative)))
            if ratio > max_length_ratio or positive == negative:
                continue
            pairs.append({"id": row_id, "group": row["group"], "split": "train",
                          "prompt": [{"role": "user", "content": prompt(row)}],
                          "chosen": [{"role": "assistant", "content": positive}],
                          "rejected": [{"role": "assistant", "content": negative}]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-length-ratio", type=float, default=2.0)
    args = ap.parse_args()
    if args.max_length_ratio < 1:
        raise ValueError("max-length-ratio must be at least 1")
    rows = read_records(args.data)
    with Path(args.audit).open(newline="", encoding="utf-8-sig") as f:
        audits = list(csv.DictReader(f))
    pairs = build_pairs(rows, audits, args.max_length_ratio)
    if not pairs:
        raise ValueError("no eligible same-prompt pairs; audit actual model candidates first")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pairs) + "\n", encoding="utf-8")
    print(f"Wrote {len(pairs)} audited train-only pairs to {output}")


if __name__ == "__main__":
    main()
