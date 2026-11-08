"""Preserve FRUIT evidence references while building multi-shard span data.

References and gold spans are training labels only. The evaluation pipeline
must never read them while selecting evidence, a sentence or an edit span.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from fruit_data import _REFERENCE, _SENTENCE
from fruit_local import collapse, extract as extract_local, reconstruct
from fruit_span import apply_span, extract_span
from span_editor.common import sha256, write_json


_ITEM = re.compile(r"(?:^|\s)\((\d+)\)\s+")


def split_evidence(text):
    """Return items indexed by publisher marker, with empty missing slots."""
    candidates = list(_ITEM.finditer(text))
    if not candidates or candidates[0].group(1) != "0" or text[:candidates[0].start()].strip():
        return []
    markers = [candidates[0]]
    for marker in candidates[1:]:
        number = int(marker.group(1))
        # Parenthesized years and list numbers inside an item are not markers.
        if int(markers[-1].group(1)) < number <= 100:
            markers.append(marker)
    items = [""] * (int(markers[-1].group(1)) + 1)
    for j, marker in enumerate(markers):
        end = markers[j + 1].start() if j + 1 < len(markers) else len(text)
        value = text[marker.end():end].strip()
        if not value:
            return []
        items[int(marker.group(1))] = value
    return items


def edit_references(target, edit_index, replacement):
    """Read references in the one literal edit slot of an EDIT5 target."""
    markers = list(_SENTENCE.finditer(target))
    boundaries = [0] + [m.end() for m in markers]
    ends = [m.start() for m in markers] + [len(target)]
    if not 0 <= edit_index < len(boundaries):
        return []
    slot = target[boundaries[edit_index]:ends[edit_index]]
    if collapse(_REFERENCE.sub("", slot)) != replacement:
        return []
    return list(dict.fromkeys(int(x[1:-1]) for x in _REFERENCE.findall(slot)))


def extract(raw):
    local, reason = extract_local(raw)
    if local is None:
        return None, reason
    span, reason = extract_span(local)
    if span is None:
        return None, reason
    items = split_evidence(span["evidence"])
    if not items:
        return None, "ambiguous_evidence_items"
    refs = edit_references(raw["target"], span["edit_index"], span["replacement"])
    refs = [i for i in refs if i < len(items) and items[i]]
    span["evidence_items"] = items
    span["reference_ids"] = refs
    span["source_dataset"] = "FRUIT-WIKI-span-v2"
    return span, None


def load(path, split):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("source_dataset") != "FRUIT-WIKI-span-v2" or row.get("split") != split:
                raise ValueError("wrong span v2 source or split")
            if row["id"] in seen:
                raise ValueError("duplicate example ID")
            if any(not 0 <= i < len(row["evidence_items"]) or not row["evidence_items"][i]
                   for i in row["reference_ids"]):
                raise ValueError("invalid evidence reference")
            if (apply_span(row, row["new_span"]) != row["replacement"] or
                    reconstruct(row, row["replacement"]) != row["gold"]):
                raise ValueError("span reconstruction mismatch")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("empty span v2 data")
    return rows


def convert(paths, destination, split):
    if split not in {"train", "dev"}:
        raise ValueError("invalid split")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    with destination.open("w", encoding="utf-8") as out:
        for path in paths:
            name = path.name
            match = re.fullmatch(r"(train|dev)_(0000[0-9])\.jsonl", name)
            if not match or match.group(1) != split or (split == "dev") != (match.group(2) == "00009"):
                raise ValueError(f"invalid publisher partition: {path}")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    raw = json.loads(line)
                    counts["read"] += 1
                    if raw["id"] in seen:
                        counts["duplicate_id"] += 1
                        continue
                    seen.add(raw["id"])
                    row, reason = extract(raw)
                    if row is None:
                        counts[reason] += 1
                        continue
                    counts["kept"] += 1
                    if row["reference_ids"]:
                        counts["with_references"] += 1
                    else:
                        counts["without_references"] += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if counts["read"] % 5000 == 0:
                        print(f"CONVERT {split}: {counts['read']} read, {counts['kept']} kept", flush=True)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.train)) != len(args.train):
        parser.error("duplicate train shard")
    args.output.mkdir(parents=True, exist_ok=True)
    train_out = args.output / "span_train_v2.jsonl"
    dev_out = args.output / "span_dev_v2.jsonl"
    train_counts = convert(args.train, train_out, "train")
    dev_counts = convert([args.dev], dev_out, "dev")
    if {x["id"] for x in load(train_out, "train")} & {x["id"] for x in load(dev_out, "dev")}:
        raise ValueError("train/development ID overlap")
    manifest = {"role": "multi_shard_reference_aware_span_v2",
                "train": [str(x) for x in args.train], "dev": str(args.dev),
                "train_counts": train_counts, "dev_counts": dev_counts,
                "sha256": {str(p): sha256(p) for p in [*args.train, args.dev, train_out, dev_out]}}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
