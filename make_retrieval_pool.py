"""Build a fixed real-document pool with same-version distractors.

Each query sees every distinct revision group in its held-out split: one old
and one gold new document per group. Version filtering therefore does not
solve the query by itself when several groups share the requested version.
"""
import argparse
import json
from pathlib import Path

from revproof_data import apply_edit, old_document, read_records


def build(rows, split):
    selected = [r for r in rows if r["split"] == split]
    representatives = {}
    for row in selected:
        representatives.setdefault(row["group"], row)
    if len(representatives) < 2:
        raise ValueError("fixed pool needs at least two distinct revision groups")
    pool = []
    for group, row in sorted(representatives.items()):
        pool.extend([
            {"doc_id": group + ":old", "text": old_document(row),
             "version": row["version_before"]},
            {"doc_id": group + ":new", "text": apply_edit(row, row["new_span"]),
             "version": row["version_after"]},
        ])
    result = []
    for row in selected:
        candidates = [{**candidate, "relevant": candidate["doc_id"] == row["group"] + ":new"}
                      for candidate in pool]
        same_version_distractors = sum(c["version"] == row["version_after"] and not c["relevant"]
                                       for c in candidates)
        if same_version_distractors < 1:
            raise ValueError(f"{row['id']}: no same-version distractor")
        result.append({"id": row["id"], "group": row["group"], "split": split,
                       "query": row["question"], "requested_version": row["version_after"],
                       "candidate_pool": "all held-out groups, old and new tagged excerpts",
                       "candidates": candidates})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = build(read_records(args.data), args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    print(f"Wrote {len(rows)} queries, {len(rows[0]['candidates'])} candidates/query "
          f"from {len({r['group'] for r in rows})} revision groups")


if __name__ == "__main__":
    main()
