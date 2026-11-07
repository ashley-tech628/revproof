"""Export a fixed paired sample for human factual-support and preservation review."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=40)
    args = ap.parse_args()
    if args.count < 1:
        ap.error("count must be positive")
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    if not rows or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("empty or duplicate prediction IDs")
    selected = sorted(rows, key=lambda x: hashlib.sha256(str(x["id"]).encode()).digest())[:args.count]
    fields = ["id", "evidence", "source", "gold_full", "copy_full",
              "base_oracle_full", "editor_oracle_full", "pipeline_full",
              "locator_sentence_index", "locator_old_span", "factual_support",
              "preserves_unchanged_facts", "silver_target_quality", "notes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            location = row.get("locator") or {}
            writer.writerow({"id": row["id"], "evidence": row["evidence"],
                             "source": row["source"], "gold_full": row["gold_full"],
                             "copy_full": row["copy"]["full"],
                             "base_oracle_full": row["base_oracle"]["full"],
                             "editor_oracle_full": row["editor_oracle"]["full"],
                             "pipeline_full": row.get("pipeline", {}).get("full", ""),
                             "locator_sentence_index": location.get("sentence_index", ""),
                             "locator_old_span": location.get("old_span", ""),
                             "factual_support": "", "preserves_unchanged_facts": "",
                             "silver_target_quality": "", "notes": ""})
    print(f"Wrote {len(selected)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
