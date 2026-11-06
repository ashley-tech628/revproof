"""Build a tiny old/new pair benchmark for wiring checks, not final claims."""
import argparse
import json
from pathlib import Path

from revproof_data import apply_edit, old_document, read_records


def build(rows, split):
    output = []
    for row in rows:
        if row["split"] != split:
            continue
        output.append({"id": row["id"], "group": row["group"], "split": split,
                       "query": row["question"], "requested_version": row["version_after"],
                       "candidates": [
                           {"doc_id": row["id"] + ":old", "text": old_document(row),
                            "version": row["version_before"], "relevant": False},
                           {"doc_id": row["id"] + ":new", "text": apply_edit(row, row["new_span"]),
                            "version": row["version_after"], "relevant": True}]})
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    records = build(read_records(args.data), args.split)
    if not records:
        raise ValueError("empty split")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} two-document sanity queries to {output}")
    print("For a research result, curate a larger fixed candidate pool with same-version distractors.")


if __name__ == "__main__":
    main()
