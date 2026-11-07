"""Inspect a completed generator pilot without treating literal checks as truth.

The audit CSV has empty human columns by design. The exact-in-candidates metric
is an oracle diagnostic only: it cannot be used to select a generation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from revproof_data import read_records


METHODS = ("template", "prompt", "sft", "prompt_filter4", "sft_filter4")
SPLITS = ("dev", "test")


def load_results(data, runs):
    rows = read_records(data)
    expected_hash = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    by_id = {row["id"]: row for row in rows}
    protocol = json.loads((runs / "protocol.json").read_text(encoding="utf-8"))
    if protocol["data_sha256"] != expected_hash:
        raise ValueError("pilot outputs and dataset have different SHA-256 hashes")
    records = []
    for split in SPLITS:
        expected_ids = {row["id"] for row in rows if row["split"] == split}
        for method in METHODS:
            path = runs / f"{method}_{split}.jsonl"
            predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
            ids = [prediction["id"] for prediction in predictions]
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                raise ValueError(f"{path}: missing, extra, or duplicate prediction ids")
            for result in predictions:
                row = by_id[result["id"]]
                if result["gold_span"] != row["new_span"]:
                    raise ValueError(f"{path}: gold mismatch for {row['id']}")
                candidates = result["candidates"]
                selected = result["selected_index"]
                if not candidates or not 0 <= selected < len(candidates):
                    raise ValueError(f"{path}: invalid candidate selection")
                prediction = result["prediction"]
                if prediction != candidates[selected]["span"]:
                    raise ValueError(f"{path}: selected prediction mismatch")
                records.append({
                    "split": split, "method": method, "id": row["id"],
                    "group": row["group"], "edit_type": row["edit_type"],
                    "question": row["question"], "old_span": row["old_span"],
                    "change_spec": json.dumps(row.get("change_spec", []), ensure_ascii=False),
                    "protected_facts": json.dumps(row["protected_facts"], ensure_ascii=False),
                    "gold_span": row["new_span"], "prediction": prediction,
                    "selected_exact": prediction == row["new_span"],
                    "selected_literal": result["passes_string_checks"],
                    "any_exact_in_candidates": any(c["span"] == row["new_span"] for c in candidates),
                    "literal_accept_nonexact": (result["passes_string_checks"]
                                                and prediction != row["new_span"]),
                    "selected_index": selected, "candidate_count": len(candidates),
                    "selected_to_gold_length_ratio": (len(prediction) / len(row["new_span"])),
                    "starts_code_fence": prediction.lstrip().startswith("```"),
                    "human_valid": "", "error_type": "", "audit_notes": "",
                })
    return records, expected_hash


def summarize(records, data_hash):
    output = {"data_sha256": data_hash, "diagnostic_notice":
              "Exact-in-candidates uses the gold reference as an oracle diagnostic, not a selector. "
              "Literal acceptance and non-exactness do not determine semantic validity.",
              "methods": []}
    for split in SPLITS:
        for method in METHODS:
            part = [r for r in records if r["split"] == split and r["method"] == method]
            output["methods"].append({
                "split": split, "method": method, "questions": len(part),
                "revision_groups": len({r["group"] for r in part}),
                "selected_exact": sum(r["selected_exact"] for r in part),
                "selected_literal": sum(r["selected_literal"] for r in part),
                "any_exact_in_candidates": sum(r["any_exact_in_candidates"] for r in part),
                "literal_accept_nonexact": sum(r["literal_accept_nonexact"] for r in part),
                "starts_code_fence": sum(r["starts_code_fence"] for r in part),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/real.jsonl")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records, data_hash = load_results(args.data, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "selected_predictions_for_human_audit.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, data_hash)
    (args.output / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for item in summary["methods"]:
        print(item)
    print("Human validity remains unlabelled in the CSV.")


if __name__ == "__main__":
    main()
