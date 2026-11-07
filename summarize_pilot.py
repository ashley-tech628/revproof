"""Aggregate generator summaries without treating literal checks as human truth."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    rows = []
    hashes = set()
    for path in sorted(root.glob("*.summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        hashes.add(summary.get("data_sha256"))
        groups = summary.get("by_revision_group", {})
        stem = path.name.removesuffix(".summary.json")
        split = "dev" if stem.endswith("_dev") else "test" if stem.endswith("_test") else "unknown"
        method = stem[:-(len(split) + 1)] if split != "unknown" else stem
        rows.append({"split": split, "method": method, "n_questions": summary["n"],
                     "n_revision_groups": groups.get("n_groups"),
                     "exact_span_rate": summary["exact_span_rate"],
                     "literal_check_rate": summary["selected_string_check_rate"],
                     "pass_at_n_literal": summary["string_pass_at_n"],
                     "all_questions_exact_group_rate": groups.get("all_questions_exact_rate"),
                     "all_questions_literal_group_rate": groups.get("all_questions_string_check_rate"),
                     "mean_candidates": summary["mean_candidates_generated"],
                     "data_sha256": summary.get("data_sha256")})
    if not rows:
        raise ValueError(f"no *.summary.json files in {root}")
    if len(hashes) != 1:
        raise ValueError("evaluation summaries use different dataset versions")
    output = root / "generator_table.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} method/split summaries to {output}")
    for row in rows:
        print(f"{row['split']:4} {row['method']:16} n={row['n_questions']} "
              f"groups={row['n_revision_groups']} exact={row['exact_span_rate']:.3f} "
              f"literal={row['literal_check_rate']:.3f}")
    print("Literal checks are not semantic validity; audit all dev/test candidates manually.")


if __name__ == "__main__":
    main()
