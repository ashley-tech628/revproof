"""Build copy-constrained short-span FRUIT examples from local-edit rows.

Gold spans are training labels. Evaluation with these spans is an oracle
upper bound; deployment must obtain them from a locator. Unchanged source
characters are always copied byte-for-byte at reconstruction time.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from fruit_local import load as load_local, reconstruct


def _inside_word(text, boundary):
    return (0 < boundary < len(text) and
            (text[boundary - 1].isalnum() or text[boundary - 1] == "_") and
            (text[boundary].isalnum() or text[boundary] == "_"))


def _left_space(text):
    return len(text) - len(text.lstrip())


def _right_space(text):
    return len(text) - len(text.rstrip())


def extract_span(row, max_span_chars=80, min_retained=0.4):
    old, new = row["old"], row["replacement"]
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(old) - prefix, len(new) - prefix)
           and old[len(old) - suffix - 1] == new[len(new) - suffix - 1]):
        suffix += 1
    # Keep names/words whole. Boundary expansion only consumes shared text.
    while prefix and (_inside_word(old, prefix) or _inside_word(new, prefix)):
        prefix -= 1
    while suffix and (_inside_word(old, len(old) - suffix) or
                      _inside_word(new, len(new) - suffix)):
        suffix -= 1
    old_end = len(old) - suffix
    new_end = len(new) - suffix
    removed, inserted = old[prefix:old_end], new[prefix:new_end]
    if not removed.strip() or not inserted.strip():
        return None, "insertion_or_deletion_only"
    if max(len(removed), len(inserted)) > max_span_chars:
        return None, "span_too_long"
    if (len(old) - len(removed)) / len(old) < min_retained:
        return None, "too_little_retained"
    if (_left_space(removed) != _left_space(inserted) or
            _right_space(removed) != _right_space(inserted)):
        return None, "boundary_whitespace_differs"
    lead = _left_space(removed)
    trail = _right_space(removed)
    if removed[:lead] != inserted[:lead] or (trail and removed[-trail:] != inserted[-trail:]):
        return None, "boundary_whitespace_differs"
    start = prefix + lead
    end = old_end - trail
    core_old = old[start:end]
    core_new = inserted[lead:len(inserted) - trail if trail else len(inserted)]
    if not core_old or not core_new:
        return None, "empty_core"
    result = {**row, "source_dataset": "FRUIT-WIKI-span",
              "span_start": start, "span_end": end,
              "old_span": core_old, "new_span": core_new,
              "prefix": old[:start], "suffix": old[end:]}
    if apply_span(result, core_new) != new:
        return None, "sentence_roundtrip_mismatch"
    if reconstruct(result, apply_span(result, core_new)) != row["gold"]:
        return None, "article_roundtrip_mismatch"
    return result, None


def apply_span(row, phrase, start=None, end=None):
    a = row["span_start"] if start is None else start
    b = row["span_end"] if end is None else end
    old = row["old"]
    if not (0 <= a < b <= len(old)):
        raise ValueError("invalid span offsets")
    return old[:a] + phrase + old[b:]


def prompt(row, start=None, end=None):
    a = row["span_start"] if start is None else start
    b = row["span_end"] if end is None else end
    old = row["old"]
    if not (0 <= a < b <= len(old)):
        raise ValueError("invalid span offsets")
    masked = old[:a] + "<extra_id_0>" + old[b:]
    return ("Fill the missing text using the evidence. Output only the missing text.\n"
            f"Previous sentence: {row['previous'] or '(none)'}\n"
            f"Sentence: {masked}\nRemoved text: {old[a:b]}\n"
            f"Next sentence: {row['next'] or '(none)'}\n"
            f"Evidence: {row['evidence']}\nMissing text:")


def load(path, split):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    if not rows or any(x.get("source_dataset") != "FRUIT-WIKI-span" or
                       x.get("split") != split for x in rows):
        raise ValueError("requires nonempty FRUIT-WIKI-span train/dev data")
    if len({x["id"] for x in rows}) != len(rows):
        raise ValueError("duplicate IDs")
    for row in rows:
        if apply_span(row, row["new_span"]) != row["replacement"] or\
           reconstruct(row, row["replacement"]) != row["gold"]:
            raise ValueError(f"span roundtrip failed: {row['id']}")
    return rows


def convert(path, output, split, max_span_chars=80, min_retained=0.4):
    rows = load_local(path, split)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            span, reason = extract_span(row, max_span_chars, min_retained)
            counts["read"] += 1
            if span is None:
                counts[reason] += 1
            else:
                out.write(json.dumps(span, ensure_ascii=False) + "\n")
                counts["kept"] += 1
            if i % 500 == 0 or i == len(rows):
                print(f"SPAN {split}: {i}/{len(rows)}; kept {counts['kept']}", flush=True)
    return {"split": split, "input": str(path), "output": str(output), **dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-span-chars", type=int, default=80)
    parser.add_argument("--min-retained", type=float, default=0.4)
    args = parser.parse_args()
    if args.max_span_chars < 1 or not 0 <= args.min_retained < 1:
        parser.error("invalid span length or retained fraction")
    args.output.mkdir(parents=True, exist_ok=True)
    train_out = args.output / "span_train_00000.jsonl"
    dev_out = args.output / "span_dev_00009.jsonl"
    splits = [convert(args.train, train_out, "train", args.max_span_chars, args.min_retained),
              convert(args.dev, dev_out, "dev", args.max_span_chars, args.min_retained)]
    train = load(train_out, "train")
    dev = load(dev_out, "dev")
    if {x["id"] for x in train} & {x["id"] for x in dev}:
        raise ValueError("train/dev ID overlap")
    manifest = {"role": "copy_constrained_span_edit",
                "max_span_chars": args.max_span_chars,
                "min_retained": args.min_retained,
                "splits": splits,
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.train, args.dev, train_out, dev_out)}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
