"""Make a paired v1/v2 report and blinded manual-audit sheet."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from span_editor.common import sha256, write_json


def read_predictions(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate prediction IDs: {path}")
    return rows


def compare_rows(earlier, newer, condition):
    if {x["id"] for x in earlier} != {x["id"] for x in newer}:
        raise ValueError("v1 and v2 must have identical development IDs")
    old = {x["id"]: x for x in earlier}
    paired = [(old[x["id"]], x) for x in newer]
    for a, b in paired:
        if a["source"] != b["source"] or a["gold_full"] != b["gold_full"]:
            raise ValueError(f"source or silver target mismatch for ID {a['id']}")
    win = sum(not a["pipeline"]["scores"]["full_exact"] and
              b[condition]["scores"]["full_exact"] for a, b in paired)
    loss = sum(a["pipeline"]["scores"]["full_exact"] and
               not b[condition]["scores"]["full_exact"] for a, b in paired)
    return paired, {"n": len(paired),
                    "v1_exact": sum(a["pipeline"]["scores"]["full_exact"]
                                    for a, _ in paired),
                    "v2_exact": sum(b[condition]["scores"]["full_exact"]
                                    for _, b in paired),
                    "v1_changed": sum(a["pipeline"]["scores"]["changed"]
                                      for a, _ in paired),
                    "v2_changed": sum(b[condition]["scores"]["changed"]
                                      for _, b in paired),
                    "v2_only_exact": win, "v1_only_exact": loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--condition", choices=["v2_forced", "v2_gated"],
                        default="v2_gated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-count", type=int, default=40)
    args = parser.parse_args()
    if args.audit_count < 1:
        parser.error("audit count must be positive")
    old = read_predictions(args.v1)
    new = read_predictions(args.v2)
    paired, report = compare_rows(old, new, args.condition)
    args.output.mkdir(parents=True, exist_ok=True)
    report.update({"role": "paired_silver_development_comparison",
                   "condition": args.condition, "v1_sha256": sha256(args.v1),
                   "v2_sha256": sha256(args.v2),
                   "warning": "Silver exact match is not factual support."})
    write_json(args.output / "paired_summary.json", report)
    sample = sorted(paired, key=lambda pair: hashlib.sha256(
        str(pair[0]["id"]).encode()).digest())[:args.audit_count]
    fields = ["id", "source", "evidence", "output_A", "output_B",
              "A_target_fact_supported", "B_target_fact_supported",
              "A_old_fact_removed", "B_old_fact_removed",
              "A_other_facts_preserved", "B_other_facts_preserved",
              "A_unsupported_additions", "B_unsupported_additions",
              "silver_target_quality", "notes"]
    key = {}
    with (args.output / "audit_blind.csv").open("w", encoding="utf-8",
                                                 newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for a, b in sample:
            identifier = a["id"]
            swap = hashlib.sha256(f"blind:{identifier}".encode()).digest()[0] & 1
            left, right = ((b[args.condition]["full"], a["pipeline"]["full"])
                           if swap else
                           (a["pipeline"]["full"], b[args.condition]["full"]))
            writer.writerow({"id": identifier, "source": a["source"],
                             "evidence": a["evidence"],
                             "output_A": left, "output_B": right})
            key[str(identifier)] = {"A": args.condition if swap else "v1_pipeline",
                                    "B": "v1_pipeline" if swap else args.condition,
                                    "gold_full": a["gold_full"]}
    write_json(args.output / "audit_key_keep_separate.json", key)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
