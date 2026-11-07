"""Convert CoDocBench train rows to unaudited code-evidence revision candidates."""
import argparse
import json
from pathlib import Path


def convert_row(row):
    versions = row.get("version_data")
    if not isinstance(versions, list) or len(versions) != 2:
        raise ValueError("expected two CoDocBench version_data entries")
    before, after = versions
    old_doc, new_doc = before.get("docstring"), after.get("docstring")
    code_diff = row.get("diff_code")
    if not all(isinstance(x, str) and x.strip() for x in (old_doc, new_doc, code_diff)):
        raise ValueError("missing old/new docstring or code diff")
    if old_doc == new_doc or row.get("whitespace_only_docstring"):
        return None
    project = row.get("project")
    if not isinstance(project, str) or not project:
        raise ValueError("missing project name")
    return {"id": f"codocbench:{project}:{after.get('commit_sha')}:{row.get('function')}",
            "group": project, "old_text": old_doc, "new_text": new_doc,
            "evidence_text": code_diff, "evidence_origin": "code_diff",
            "evidence_commit": after.get("commit_sha"),
            "source_dataset": "kunpai/codocbench", "source_split": "train",
            "training_role": "domain_silver_candidate", "support_audited": False,
            "audit_note": "Code and docstring changed in one commit; code diff may not entail docstring edit."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CoDocBench dataset/train.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with source.open(encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = convert_row(json.loads(line))
            if item is None:
                skipped += 1
                continue
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"kept": kept, "skipped": skipped,
                      "training_role": "domain_silver_candidate"}))


if __name__ == "__main__":
    main()
