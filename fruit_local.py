"""Strict one-source-sentence FRUIT extraction for an oracle-locus diagnostic.

The old sentence location comes from the gold EDIT5 target. It may be used
only for this explicitly labelled upper-bound experiment, never for an
end-to-end updater. Evidence is taken solely from the original input.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from fruit_data import _REFERENCE, _SENTENCE, normalize_fruit


def collapse(text):
    return re.sub(r"\s{2,}", " ", text).strip()


def source_sentences(inputs):
    source = inputs.split("[CONTEXT]", 1)[0].strip()
    markers = list(_SENTENCE.finditer(source))
    if not markers or markers[0].start() != 0:
        raise ValueError("source does not start at [0]")
    result = []
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != index:
            raise ValueError("source markers are not consecutive")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        result.append(collapse(source[marker.end():end]))
    return result


def reconstruct(row, replacement):
    parts = row["source_sentences"].copy()
    parts[row["edit_index"]] = collapse(replacement)
    return collapse(" ".join(parts))


def extract(row, max_chars=400):
    """Return a row only when one literal replaces exactly one old sentence."""
    if row.get("source_dataset") != "FRUIT-WIKI" or row.get("split") not in {"train", "dev"}:
        raise ValueError("requires converted FRUIT-WIKI train/dev data")
    if "[CONTEXT]" not in row["input"]:
        return None, "no_context"
    try:
        sentences = source_sentences(row["input"])
        source, gold = normalize_fruit(row["input"], row["target"])
    except ValueError:
        return None, "invalid_source_or_target"
    if source != row["source"] or gold != row["gold"]:
        return None, "stored_reconstruction_mismatch"
    clean_target = _REFERENCE.sub("", row["target"])
    markers = list(_SENTENCE.finditer(clean_target))
    indices = [int(match.group(1)) for match in markers]
    if len(indices) != len(sentences) - 1 or indices != sorted(set(indices)):
        return None, "not_one_omitted_sentence"
    missing = set(range(len(sentences))) - set(indices)
    if len(missing) != 1:
        return None, "not_one_omitted_sentence"
    edit_index = missing.pop()
    boundaries = [0] + [match.end() for match in markers]
    ends = [match.start() for match in markers] + [len(clean_target)]
    literals = [(slot, collapse(clean_target[start:end]))
                for slot, (start, end) in enumerate(zip(boundaries, ends))
                if collapse(clean_target[start:end])]
    if len(literals) != 1 or literals[0][0] != edit_index:
        return None, "not_one_literal_at_edit"
    replacement = literals[0][1]
    old = sentences[edit_index]
    if not old or not replacement or old == replacement:
        return None, "empty_or_noop"
    if len(old) > max_chars or len(replacement) > max_chars:
        return None, "span_too_long"
    result = {
        "id": row["id"], "split": row["split"],
        "source_dataset": "FRUIT-WIKI-local-oracle",
        "edit_index": edit_index, "source_sentences": sentences,
        "old": old, "replacement": replacement,
        "previous": sentences[edit_index - 1] if edit_index else "",
        "next": sentences[edit_index + 1] if edit_index + 1 < len(sentences) else "",
        "evidence": row["input"].split("[CONTEXT]", 1)[1].strip(),
        "source": source, "gold": gold,
    }
    if reconstruct(result, replacement) != gold:
        return None, "local_roundtrip_mismatch"
    return result, None


def prompt(row):
    return ("Revise the old sentence using the evidence. Output only the replacement text.\n"
            f"Previous: {row['previous'] or '(none)'}\n"
            f"Old: {row['old']}\n"
            f"Next: {row['next'] or '(none)'}\n"
            f"Evidence: {row['evidence']}\nReplacement:")


def choose(rows, limit):
    ranked = sorted(rows, key=lambda x: hashlib.sha256(str(x["id"]).encode()).digest())
    return ranked if limit is None else ranked[:limit]


def load(path, split):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    if not rows or any(x.get("split") != split or
                       x.get("source_dataset") != "FRUIT-WIKI-local-oracle" for x in rows):
        raise ValueError(f"requires nonempty converted local {split} rows")
    if len({x["id"] for x in rows}) != len(rows):
        raise ValueError("duplicate IDs")
    for row in rows:
        if reconstruct(row, row["replacement"]) != row["gold"]:
            raise ValueError(f"roundtrip failed for ID {row['id']}")
    return rows


def convert(path, output, split, max_chars=400):
    if split not in {"train", "dev"}:
        raise ValueError("training and development only")
    name = re.fullmatch(r"(train|dev)_(0000[0-9])\.jsonl", Path(path).name)
    if not name or name.group(1) != split or (split == "dev") != (name.group(2) == "00009"):
        raise ValueError("train shards 00000-00008; dev shard 00009 only")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    ids = set()
    with Path(path).open(encoding="utf-8") as stream, output.open("w", encoding="utf-8") as dest:
        for line in stream:
            counts["read"] += 1
            row = json.loads(line)
            if row.get("split") != split or row["id"] in ids:
                raise ValueError("wrong split or duplicate ID")
            ids.add(row["id"])
            local, reason = extract(row, max_chars)
            if local is None:
                counts[reason] += 1
            else:
                dest.write(json.dumps(local, ensure_ascii=False) + "\n")
                counts["kept"] += 1
            if counts["read"] % 1000 == 0:
                print(f"EXTRACT {split}: {counts['read']} read, {counts['kept']} kept", flush=True)
    return {"input": str(path), "output": str(output), "split": split,
            "max_chars": max_chars, **dict(counts)}


def audit_sample(path, output, count=40):
    rows = choose(load(path, "dev"), count)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "old", "gold_replacement",
            "previous", "next", "evidence", "evidence_support", "meaningful_change",
            "silver_quality", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "old": row["old"],
                             "gold_replacement": row["replacement"],
                             "previous": row["previous"], "next": row["next"],
                             "evidence": row["evidence"], "evidence_support": "",
                             "meaningful_change": "", "silver_quality": "", "notes": ""})
    return {"path": str(output), "count": len(rows), "ids": [x["id"] for x in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--audit-count", type=int, default=40)
    args = parser.parse_args()
    if args.max_chars < 1 or args.audit_count < 1:
        parser.error("length and audit count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    train_out = args.output / "local_train_00000.jsonl"
    dev_out = args.output / "local_dev_00009.jsonl"
    summaries = [convert(args.train, train_out, "train", args.max_chars),
                 convert(args.dev, dev_out, "dev", args.max_chars)]
    if {x["id"] for x in load(train_out, "train")} & {x["id"] for x in load(dev_out, "dev")}:
        raise ValueError("train/dev ID overlap")
    audit = audit_sample(dev_out, args.output / f"dev_audit_{args.audit_count}.csv",
                         args.audit_count)
    manifest = {"role": "oracle_locus_silver_diagnostic", "splits": summaries,
                "audit": audit, "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in [args.train, args.dev, train_out, dev_out]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
