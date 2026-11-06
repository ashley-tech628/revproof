"""Export every generated candidate for independent human annotation."""
import argparse
import csv
import json
from pathlib import Path

from revproof_data import read_records


def export(rows, predictions, path):
    by_id = {r["id"]: r for r in rows}
    fields = ["id", "group", "split", "source_url", "edit_type", "question",
              "old_span", "gold_span", "candidate_index", "candidate_span",
              "passes_string_checks", "human_valid", "error_type", "notes"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in predictions:
            if result["id"] not in by_id:
                raise ValueError(f"prediction id absent from data: {result['id']}")
            row = by_id[result["id"]]
            for i, candidate in enumerate(result["candidates"]):
                writer.writerow({"id": row["id"], "group": row["group"], "split": row["split"],
                                 "source_url": row["source_url"], "edit_type": row["edit_type"],
                                 "question": row["question"], "old_span": row["old_span"],
                                 "gold_span": row["new_span"], "candidate_index": i,
                                 "candidate_span": candidate["span"],
                                 "passes_string_checks": candidate["passes_string_checks"],
                                 "human_valid": "", "error_type": "", "notes": ""})
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_records(args.data)
    predictions = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line]
    print(f"Exported {export(rows, predictions, args.output)} candidates to {args.output}")
    print("Fill human_valid with true or false after checking fact, scope, preservation and fluency.")


if __name__ == "__main__":
    main()
