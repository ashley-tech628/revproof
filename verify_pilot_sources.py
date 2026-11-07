"""Verify every selected excerpt against immutable upstream release tags."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def fetch(url, cache):
    digest = hashlib.sha256(url.encode()).hexdigest()
    path = cache / digest
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/source_verification_cache")
    args = ap.parse_args()
    root = Path(__file__).parent
    cache = Path(args.cache)
    node = [json.loads(line) for line in (root / "data/node_selected_sources.jsonl")
            .read_text(encoding="utf-8").splitlines() if line]
    flask = [json.loads(line) for line in (root / "data/flask_selected_sources.jsonl")
             .read_text(encoding="utf-8").splitlines() if line]
    checked = 0
    for row in node:
        for side in ("old", "new"):
            raw = fetch(row[f"source_url_{side}"], cache)
            if hashlib.sha256(raw).hexdigest() != row[f"source_sha256_{side}"]:
                raise ValueError(f"{row['candidate_id']}: {side} source SHA-256 mismatch")
            span = row["old_span" if side == "old" else "new_span"]
            if normalize(span) not in normalize(raw.decode("utf-8")):
                raise ValueError(f"{row['candidate_id']}: {side} span absent from official tag")
        checked += 1
    for row in flask:
        path = row["path"]
        for tag, key in (("3.0.3", "old_lines"), ("3.1.0", "new_lines")):
            url = f"https://raw.githubusercontent.com/pallets/flask/{tag}/{path}"
            raw = fetch(url, cache).decode("utf-8")
            span = " ".join(row[key])
            if normalize(span) not in normalize(raw):
                raise ValueError(f"{row['candidate_id']}: {tag} span absent from official tag")
        checked += 1
    print(f"Verified {checked} selected edits against old and new official tagged sources")


if __name__ == "__main__":
    main()
