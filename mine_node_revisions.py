"""Mine *unreviewed* prose edits from official tagged Node.js API docs.

The output is an audit queue, never a labeled training set. Review changed
facts, scope, and answer relations against both upstream tags before inclusion.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


def paragraphs(markdown):
    sections = re.split(r"\n\s*\n", markdown.replace("\r\n", "\n"))
    output = []
    heading = ""
    fenced = False
    for section in sections:
        raw = section.strip()
        if raw.startswith("#") and "\n" not in raw:
            heading = raw.lstrip("# ")
        if raw.count("```") % 2 or raw.count("~~~") % 2:
            fenced = not fenced
        if not raw or fenced or raw.startswith(("<!--", "```", "~~~", "|", "[")):
            continue
        if "```" in raw or "~~~" in raw or "<!--" in raw or "-->" in raw:
            continue
        text = re.sub(r"\s+", " ", raw)
        if 45 <= len(text) <= 900 and not text.startswith(("#", "* ", "- ", ">")):
            output.append({"text": text, "heading": heading})
    return output


def fetch_doc(version, name, cache):
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError(f"invalid Node.js version: {version}")
    if not re.fullmatch(r"[a-z][a-z_]*", name):
        raise ValueError(f"unsupported document: {name}")
    url = f"https://raw.githubusercontent.com/nodejs/node/{version}/doc/api/{name}.md"
    target = cache / version / f"{name}.md"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            target.write_bytes(response.read())
    raw = target.read_bytes()
    return paragraphs(raw.decode("utf-8")), url, hashlib.sha256(raw).hexdigest()


def candidates(old, new, old_version, new_version, doc, old_url, new_url, old_sha, new_sha):
    left = [p["text"] for p in old]
    right = [p["text"] for p in new]
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    found = []
    for tag, i, j, k, l in matcher.get_opcodes():
        if tag != "replace":
            continue
        used = set()
        for old_index in range(i, j):
            choices = [(SequenceMatcher(None, left[old_index], right[m]).ratio(), m)
                       for m in range(k, l) if m not in used]
            if not choices:
                continue
            similarity, new_index = max(choices)
            if not 0.64 <= similarity < 0.99:
                continue
            used.add(new_index)
            row = {"candidate_id": f"node_{doc}_{old_version}_{new_version}_{old_index:04d}",
                   "document": doc, "version_before": old_version,
                   "version_after": new_version,
                   "source_url_old": old_url, "source_url_new": new_url,
                   "source_sha256_old": old_sha, "source_sha256_new": new_sha,
                   "heading_old": old[old_index]["heading"],
                   "heading_new": new[new_index]["heading"],
                   "old_span": left[old_index], "new_span": right[new_index],
                   "similarity": round(similarity, 4),
                   "audit_status": "pending"}
            found.append(row)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", required=True,
                        help="ordered tags such as v20.19.0 v21.7.3 v22.14.0")
    parser.add_argument("--docs", nargs="+", default=["assert", "errors"])
    parser.add_argument("--cache", default="data/node_source_cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.versions) < 2:
        parser.error("need at least two versions")
    cache = Path(args.cache)
    rows = []
    for doc in args.docs:
        loaded = {v: fetch_doc(v, doc, cache) for v in args.versions}
        for older, newer in zip(args.versions, args.versions[1:]):
            old, old_url, old_sha = loaded[older]
            new, new_url, new_sha = loaded[newer]
            rows.extend(candidates(old, new, older, newer, doc,
                                   old_url, new_url, old_sha, new_sha))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    print(f"Wrote {len(rows)} UNREVIEWED prose-edit candidates to {output}")


if __name__ == "__main__":
    main()
