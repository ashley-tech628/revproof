"""Evaluate version-aware retrieval with fixed candidate pools."""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def terms(text):
    return re.findall(r"[a-z0-9_.]+", text.lower())


def bm25(query, documents, k1=1.5, b=0.75):
    tokens = [terms(d) for d in documents]
    n = len(tokens)
    avgdl = sum(map(len, tokens)) / max(n, 1)
    dfs = Counter(t for doc in tokens for t in set(doc))
    result = []
    for doc in tokens:
        tf = Counter(doc)
        score = 0.0
        for term in set(terms(query)):
            if term not in tf:
                continue
            idf = math.log(1 + (n - dfs[term] + 0.5) / (dfs[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(doc) / max(avgdl, 1)))
        result.append(score)
    return result


def validate_query(row):
    if not all(k in row for k in ("id", "group", "query", "requested_version", "candidates")):
        raise ValueError("retrieval row lacks required fields")
    candidates = row["candidates"]
    if not candidates or not any(c.get("relevant") is True for c in candidates):
        raise ValueError(f"{row['id']}: need at least one relevant candidate")
    ids = [c.get("doc_id") for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{row['id']}: duplicate doc_id")
    if any(not all(k in c for k in ("text", "version", "relevant")) for c in candidates):
        raise ValueError(f"{row['id']}: incomplete candidate")


def score_row(row, method, reranker=None):
    validate_query(row)
    candidates = row["candidates"]
    if method.startswith("version_"):
        candidates = [c for c in candidates if c["version"] == row["requested_version"]]
    if not candidates:
        return []
    query = row["query"] + " Requested version: " + row["requested_version"]
    if method.endswith("reranker"):
        scores = reranker.predict([(query, c["text"]) for c in candidates])
        scores = [float(x) for x in scores]
    else:
        scores = bm25(query, [c["text"] for c in candidates])
    return [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]


def metrics(rows, ranked, k):
    n = len(rows)
    reciprocal = []
    recall = []
    top1 = []
    wrong_version = []
    coverage = []
    for row, candidates in zip(rows, ranked):
        hits = [i + 1 for i, c in enumerate(candidates[:k]) if c["relevant"]]
        reciprocal.append(1 / hits[0] if hits else 0)
        relevant_count = sum(c["relevant"] for c in row["candidates"])
        recall.append(len(hits) / relevant_count)
        top1.append(bool(candidates and candidates[0]["relevant"]))
        wrong_version.append(bool(candidates and candidates[0]["version"] != row["requested_version"]))
        coverage.append(bool(candidates))
    return {"queries": n, "revision_groups": len({r["group"] for r in rows}),
            f"mrr_at_{k}": sum(reciprocal) / n,
            f"recall_at_{k}": sum(recall) / n,
            "top1_relevance": sum(top1) / n,
            "top1_wrong_version": sum(wrong_version) / n,
            "nonempty_result_rate": sum(coverage) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--method", choices=["bm25", "version_bm25", "reranker", "version_reranker"], required=True)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.k < 1:
        raise ValueError("k must be positive")
    rows = [json.loads(line) for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty benchmark")
    reranker = None
    if args.method.endswith("reranker"):
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.model)
    ranked = [score_row(row, args.method, reranker) for row in rows]
    summary = {"method": args.method, "model": args.model if reranker else None,
               **metrics(rows, ranked, args.k)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
