"""Read official FRUIT-WIKI EDIT5 examples and reconstruct their edited text.

The official processed TFRecords contain `inputs` (numbered source sentences
plus evidence) and `targets` (new sentences plus source-sentence copy tokens).
Only FRUIT train shards 0-8 may be used for training; shard 9 is development.
The later-period gold_test is reserved for final evaluation.
"""

import argparse
import json
import re
from pathlib import Path


_SENTENCE = re.compile(r"\[(\d+)\]")
_REFERENCE = re.compile(r"\(\d+\)")


def normalize_fruit(inputs: str, edit: str) -> tuple[str, str]:
    """Expand copy tokens using source sentences, following official format.

    This is a lightweight implementation of the reconstruction specified in
    google-research/language/language/fruit/rendering_utils.py. It raises on
    malformed copy references instead of silently dropping them.
    """
    source = inputs.split("[CONTEXT]", 1)[0].strip()
    markers = list(_SENTENCE.finditer(source))
    if not markers or markers[0].start() != 0:
        raise ValueError("source must start with a sentence marker")
    sentences = []
    for i, match in enumerate(markers):
        if int(match.group(1)) != i:
            raise ValueError("source sentence markers must be consecutive")
        end = markers[i + 1].start() if i + 1 < len(markers) else len(source)
        sentences.append(source[match.end():end].strip())
    clean_source = " ".join(sentences)
    clean_edit = _REFERENCE.sub("", edit)
    parts = []
    cursor = 0
    for match in _SENTENCE.finditer(clean_edit):
        before = clean_edit[cursor:match.start()].strip()
        if before:
            parts.append(before)
        index = int(match.group(1))
        if index >= len(sentences):
            raise ValueError(f"copy marker [{index}] outside source")
        parts.append(sentences[index])
        cursor = match.end()
    tail = clean_edit[cursor:].strip()
    if tail:
        parts.append(tail)
    return (re.sub(r"\s{2,}", " ", clean_source).strip(),
            re.sub(r"\s{2,}", " ", " ".join(parts)).strip())


def convert_tfrecord(path: Path, output: Path, split: str) -> dict:
    """Stream one official processed shard into compact JSONL."""
    if split not in {"train", "dev"}:
        raise ValueError("only official train/development shards are accepted")
    name = path.name
    official = re.fullmatch(
        r"article_pairs\.update\.diff\.all\.text\.reference\.tfrecords-(0000\d)-of-00010",
        name)
    short = re.fullmatch(r"(train|dev)_(0000\d)\.tfrecord", name)
    if official:
        shard = int(official.group(1))
    elif short:
        shard = int(short.group(2))
        if short.group(1) != split:
            raise ValueError("filename split and requested split disagree")
    else:
        raise ValueError("requires an official FRUIT train shard filename")
    if (split == "dev") != (shard == 9):
        raise ValueError("FRUIT development is shard 9; training uses shards 0-8")
    from tfrecord import tfrecord_loader

    output.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0
    with output.open("w", encoding="utf-8") as stream:
        for item in tfrecord_loader(str(path), None):
            inputs = item["inputs"].decode("utf-8")
            target = item["targets"].decode("utf-8")
            try:
                source, gold = normalize_fruit(inputs, target)
            except ValueError:
                rejected += 1
                continue
            row = {"id": int(item["id"][0]), "split": split,
                   "source_dataset": "FRUIT-WIKI", "input": inputs,
                   "target": target, "source": source, "gold": gold}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return {"split": split, "kept": kept, "rejected": rejected,
            "input": str(path), "output": str(output)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    args = ap.parse_args()
    print(json.dumps(convert_tfrecord(args.input, args.output, args.split), indent=2))


if __name__ == "__main__":
    main()
