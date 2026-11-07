"""Exact-copy, multi-span edit representation for versioned documents.

The diff is a *training label* derived from paired texts. It is never an
inference-time input. Character offsets refer to the original source text.
"""
from difflib import SequenceMatcher
import re


def _tokens(text):
    return re.findall(r"\s+|[^\s]+", text)


def apply_plan(source, edits):
    """Apply ordered, nonoverlapping source-offset replacements."""
    if not isinstance(source, str) or not isinstance(edits, list):
        raise ValueError("source must be a string and edits must be a list")
    output = []
    cursor = 0
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"start", "end", "new"}:
            raise ValueError("each edit requires exactly start, end, new")
        start, end, new = edit["start"], edit["end"], edit["new"]
        if (not isinstance(start, int) or isinstance(start, bool) or
                not isinstance(end, int) or isinstance(end, bool) or
                not isinstance(new, str) or cursor > start or start > end or
                end > len(source)):
            raise ValueError("invalid or overlapping source offsets")
        if start == end and not new:
            raise ValueError("empty edit")
        output.extend((source[cursor:start], new))
        cursor = end
    output.append(source[cursor:])
    return "".join(output)


def gold_plan(source, target):
    """Align whitespace-preserving tokens; return multiple minimal edit blocks."""
    if not source or not target:
        raise ValueError("source and target must be nonempty")
    left, right = _tokens(source), _tokens(target)
    offsets = [0]
    for token in left:
        offsets.append(offsets[-1] + len(token))
    edits = []
    for tag, i, j, k, l in SequenceMatcher(
            a=left, b=right, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append({"start": offsets[i], "end": offsets[j],
                          "new": "".join(right[k:l])})
    if apply_plan(source, edits) != target:
        raise AssertionError("alignment failed to reconstruct target")
    return edits


def validate_grounded_record(record):
    """Require independent evidence provenance before grounded training."""
    required = {"id", "group", "old_text", "new_text", "evidence_text",
                "evidence_url", "evidence_version", "target_version",
                "evidence_origin", "support_audited"}
    if not isinstance(record, dict) or required - record.keys():
        raise ValueError("grounded record is missing required fields")
    if any(not isinstance(record[key], str) or not record[key].strip()
           for key in required - {"support_audited"}):
        raise ValueError("grounded record has empty text or provenance")
    if record["evidence_origin"] == "target_document":
        raise ValueError("target document cannot serve as independent evidence")
    if record["evidence_version"] != record["target_version"]:
        raise ValueError("evidence and target versions differ")
    if record["support_audited"] is not True:
        raise ValueError("evidence support must be explicitly audited")
    return record
