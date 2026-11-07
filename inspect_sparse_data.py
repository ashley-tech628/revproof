"""Reproducible, model-free diagnostics for the sparse-edit target."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from revproof_data import read_records
from sparse_edit import gold_script, script_text


def inspect(rows):
    report = {}
    for split in ("train", "dev", "test"):
        selected = [r for r in rows if r["split"] == split]
        groups = {r["group"]: r for r in selected}
        entries = []
        for row in groups.values():
            script = gold_script(row["old_span"], row["new_span"])
            entries.append({"group": row["group"], "old_chars": len(row["old_span"]),
                            "anchor_chars": len(script["old"]),
                            "replacement_chars": len(script["new"]),
                            "script_chars": len(script_text(script)),
                            "anchor_fraction": len(script["old"]) / len(row["old_span"])})
        report[split] = {
            "question_rows": len(selected), "independent_groups": len(groups),
            "old_chars": sum(x["old_chars"] for x in entries),
            "anchor_chars": sum(x["anchor_chars"] for x in entries),
            "replacement_chars": sum(x["replacement_chars"] for x in entries),
            "script_chars": sum(x["script_chars"] for x in entries),
            "median_anchor_fraction": statistics.median(x["anchor_fraction"] for x in entries),
            "whole_span_windows": sum(x["anchor_fraction"] == 1.0 for x in entries),
            "groups": entries,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--output")
    args = ap.parse_args()
    rows = read_records(args.data)
    report = {"data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "method": "one shortest unique word-aligned contiguous window per independent group",
              "splits": inspect(rows)}
    text = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
