"""Pair v2 and v3 predictions on identical frozen development examples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from span_editor.common import sha256, write_json
from span_editor.compare_v2 import read_predictions


def compare_rows(v2, v3, old_condition="v2_forced",
                 new_condition="v3_forced"):
    if {x["id"] for x in v2} != {x["id"] for x in v3}:
        raise ValueError("v2 and v3 must have identical development IDs")
    by_id = {x["id"]: x for x in v2}
    paired = [(by_id[x["id"]], x) for x in v3]
    for old, new in paired:
        if old["source"] != new["source"] or old["gold_full"] != new["gold_full"]:
            raise ValueError(f"source or silver target mismatch for ID {old['id']}")
    def score(row, condition, name):
        return bool(row[condition]["scores"][name])
    report = {"n": len(paired), "v2_condition": old_condition,
              "v3_condition": new_condition,
              "v2_exact": sum(score(a, old_condition, "full_exact") for a, _ in paired),
              "v3_exact": sum(score(b, new_condition, "full_exact") for _, b in paired),
              "v2_changed": sum(score(a, old_condition, "changed") for a, _ in paired),
              "v3_changed": sum(score(b, new_condition, "changed") for _, b in paired),
              "v3_only_exact": sum(not score(a, old_condition, "full_exact") and
                                   score(b, new_condition, "full_exact")
                                   for a, b in paired),
              "v2_only_exact": sum(score(a, old_condition, "full_exact") and
                                   not score(b, new_condition, "full_exact")
                                   for a, b in paired)}
    return paired, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v2-condition", choices=["v2_forced", "v2_gated"],
                        default="v2_forced")
    parser.add_argument("--v3-condition", choices=["v3_forced", "v3_gated"],
                        default="v3_forced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-count", type=int, default=40)
    args = parser.parse_args()
    if args.audit_count < 1:
        parser.error("audit count must be positive")
    paired, report = compare_rows(read_predictions(args.v2),
                                  read_predictions(args.v3),
                                  args.v2_condition, args.v3_condition)
    report.update({"role": "paired_silver_development_v2_v3_comparison",
                   "v2_sha256": sha256(args.v2), "v3_sha256": sha256(args.v3),
                   "warning": "Silver exact match is not factual support."})
    args.output.mkdir(parents=True, exist_ok=True)
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
        for old, new in sample:
            identifier = old["id"]
            swap = hashlib.sha256(f"v3blind:{identifier}".encode()).digest()[0] & 1
            first = new[args.v3_condition]["full"] if swap else old[args.v2_condition]["full"]
            second = old[args.v2_condition]["full"] if swap else new[args.v3_condition]["full"]
            writer.writerow({"id": identifier, "source": old["source"],
                             "evidence": old["evidence"],
                             "output_A": first, "output_B": second})
            key[str(identifier)] = {"A": args.v3_condition if swap else args.v2_condition,
                                    "B": args.v2_condition if swap else args.v3_condition,
                                    "gold_full": old["gold_full"]}
    write_json(args.output / "audit_key_keep_separate.json", key)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
