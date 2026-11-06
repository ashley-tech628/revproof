"""Build labeled query-document pairs from real revisions and human-audited generations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from prepare_preferences import parse_label
from revproof_data import apply_edit, old_document, read_records


def build(rows, split, audit_rows=None, generated_source="generated", max_generated_per_row=1):
    if split not in {"train", "dev"}:
        raise ValueError("reranker builder uses train or dev only")
    audit_by_id = defaultdict(list)
    for audit in audit_rows or []:
        audit_by_id[audit["id"]].append(audit)
    output = []
    for row in rows:
        if row["split"] != split:
            continue
        if row["source_type"] != "real":
            raise ValueError("reranker research data must be audited real revisions")
        query = f"{row['question']} Requested version: {row['version_after']}. Scope: {row['scope']}."
        common = {"id": row["id"], "group": row["group"], "split": split, "query": query}
        output.append({**common, "document": apply_edit(row, row["new_span"]),
                       "label": 1, "source": "real_new"})
        output.append({**common, "document": old_document(row),
                       "label": 0, "source": "real_old"})
        if max_generated_per_row == 0:
            continue
        added = 0
        seen_spans = {row["old_span"], row["new_span"]}
        for audit in audit_by_id[row["id"]]:
            span = audit["candidate_span"].strip()
            if not span or span in seen_spans:
                continue
            seen_spans.add(span)
            output.append({**common, "document": apply_edit(row, span),
                           "label": int(parse_label(audit["human_valid"])),
                           "source": generated_source})
            added += 1
            if added >= max_generated_per_row:
                break
    unknown = set(audit_by_id) - {r["id"] for r in rows if r["split"] == split}
    if unknown:
        raise ValueError(f"audit ids outside {split} data: {sorted(unknown)}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--audit", help="human-completed CSV; optional for real-only baseline")
    ap.add_argument("--generated-source", default="generated")
    ap.add_argument("--max-generated-per-row", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.max_generated_per_row < 0:
        raise ValueError("max-generated-per-row must be nonnegative")
    audits = []
    if args.audit:
        with Path(args.audit).open(newline="", encoding="utf-8-sig") as f:
            audits = list(csv.DictReader(f))
    pairs = build(read_records(args.data), args.split, audits, args.generated_source,
                  args.max_generated_per_row)
    if not pairs:
        raise ValueError("empty split")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pairs) + "\n", encoding="utf-8")
    print(f"Wrote {len(pairs)} labeled pairs to {output}")


if __name__ == "__main__":
    main()
