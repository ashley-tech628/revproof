"""Shared pure helpers for the locator and short-span infiller."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked(rows):
    return sorted(rows, key=lambda x: hashlib.sha256(str(x["id"]).encode()).digest())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def encode_infiller(tokenizer, row, max_input=1024, max_target=96):
    from fruit_span import prompt

    source = tokenizer(prompt(row), add_special_tokens=True, truncation=False).input_ids
    target = tokenizer(row["new_span"], add_special_tokens=True, truncation=False).input_ids
    if len(source) > max_input or len(target) > max_target:
        return None
    return {"input_ids": source, "attention_mask": [1] * len(source), "labels": target}


def load_protocol(adapter):
    path = Path(adapter).parent / "protocol.json"
    if not path.is_file():
        raise ValueError(f"missing training protocol: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
