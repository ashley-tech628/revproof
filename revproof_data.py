"""Data contract and deterministic checks for the RevProof pilot."""
import json
from pathlib import Path


REQUIRED = {
    "id", "group", "split", "source_type", "source_url", "question",
    "answer_before", "answer_after", "prefix", "old_span", "new_span",
    "suffix", "field", "old_value", "new_value", "version_before",
    "version_after", "scope",
}


def read_records(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"line {number}: missing {sorted(missing)}")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"line {number}: invalid split")
        if row["source_type"] not in {"real", "synthetic_toy"}:
            raise ValueError(f"line {number}: invalid source_type")
        if not all(isinstance(row[k], str) for k in REQUIRED):
            raise ValueError(f"line {number}: required values must be strings")
        if row["source_type"] == "real" and not row["source_url"].strip():
            raise ValueError(f"line {number}: real record needs a source URL")
        protected = row.get("protected_facts", [])
        if not isinstance(protected, list) or not all(isinstance(x, str) and x for x in protected):
            raise ValueError(f"line {number}: protected_facts must be a list of nonempty strings")
        if any(x not in row["prefix"] + row["old_span"] + row["suffix"] for x in protected):
            raise ValueError(f"line {number}: protected fact absent from old document")
        row["protected_facts"] = protected
        if row.get("edit_type", "scalar") not in {"scalar", "conditional", "cross_sentence"}:
            raise ValueError(f"line {number}: invalid edit_type")
        row.setdefault("edit_type", "scalar")
        if not row["old_span"] or not row["new_span"]:
            raise ValueError(f"line {number}: edit spans must not be empty")
        if row["old_span"] == row["new_span"]:
            raise ValueError(f"line {number}: no actual document edit")
        if row["old_value"] not in row["old_span"]:
            raise ValueError(f"line {number}: old_value absent from old_span")
        if row["new_value"] not in row["new_span"]:
            raise ValueError(f"line {number}: new_value absent from new_span")
        row["z"] = "keep" if row["answer_before"] == row["answer_after"] else "change"
        records.append(row)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate id")
    splits_by_group = {}
    for row in records:
        splits_by_group.setdefault(row["group"], set()).add(row["split"])
    leaks = {g: s for g, s in splits_by_group.items() if len(s) != 1}
    if leaks:
        raise ValueError(f"group leakage across splits: {leaks}")
    splits_by_revision = {}
    for row in records:
        if row["source_type"] == "real":
            revision = (row["source_url"], row["version_before"], row["version_after"])
            splits_by_revision.setdefault(revision, set()).add(row["split"])
    revision_leaks = {k: v for k, v in splits_by_revision.items() if len(v) != 1}
    if revision_leaks:
        raise ValueError(f"same source revision across splits: {revision_leaks}")
    return records


def old_document(row):
    return row["prefix"] + row["old_span"] + row["suffix"]


def apply_edit(row, generated_span):
    return row["prefix"] + generated_span + row["suffix"]


def prompt(row):
    return (
        "You edit one marked span in a versioned technical document. "
        "Return only the replacement span. Keep all other facts consistent.\n"
        f"Question: {row['question']}\n"
        f"Answer relation: {row['z']}\n"
        f"Field: {row['field']}\n"
        f"Old value: {row['old_value']}\n"
        f"New value: {row['new_value']}\n"
        f"Version: {row['version_before']} -> {row['version_after']}\n"
        f"Scope: {row['scope']}\n"
        f"Facts to preserve: {json.dumps(row['protected_facts'], ensure_ascii=False)}\n"
        f"Before marked span: {row['prefix']}\n"
        f"Marked span to replace: {row['old_span']}\n"
        f"After marked span: {row['suffix']}"
    )


def template_edit(row):
    return row["old_span"].replace(row["old_value"], row["new_value"], 1)


def check_candidate(row, generated_span):
    """Conservative string checks; NOT proof of semantic correctness."""
    span = generated_span.strip()
    reconstructed = apply_edit(row, span)
    checks = {
        "nonempty": bool(span),
        "new_value_present": row["new_value"] in span,
        "old_value_removed": (row["old_value"] not in span
                              if row["old_value"] != row["new_value"]
                              and row["old_value"] not in row["new_value"]
                              and not row.get("allow_old_value_in_new_span", False) else True),
        "protected_facts_present": all(x in reconstructed for x in row["protected_facts"]),
    }
    return {"checks": checks, "passes_string_checks": all(checks.values()),
            "reconstructed_document": reconstructed}
