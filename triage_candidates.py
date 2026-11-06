"""Prioritize Git documentation diffs for human review; never auto-label truth."""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


FACT_WORDS = {"default", "support", "supports", "require", "requires", "must", "cannot",
              "deprecated", "minimum", "maximum", "limit", "version", "available",
              "enabled", "disabled", "timeout", "error", "response", "request"}
MARKUP = ("{*", "{!", "```", "//// tab", "/// note", ".. include::", ".. literalinclude::")


def triage(row):
    old = " ".join(row["old_lines"]).strip()
    new = " ".join(row["new_lines"]).strip()
    if not old or not new:
        return None
    if any(marker in old or marker in new for marker in MARKUP):
        return None
    if len(old) < 35 or len(new) < 35 or max(len(old), len(new)) > 1500:
        return None
    old_tokens = set(re.findall(r"[a-z]+", old.lower()))
    new_tokens = set(re.findall(r"[a-z]+", new.lower()))
    old_numbers = re.findall(r"\b\d+(?:\.\d+)*\b", old)
    new_numbers = re.findall(r"\b\d+(?:\.\d+)*\b", new)
    score = 0
    reasons = []
    if old_numbers != new_numbers:
        score += 3
        reasons.append("numeric_or_version_change")
    if (old_tokens ^ new_tokens) & FACT_WORDS:
        score += 2
        reasons.append("factual_qualifier_change")
    similarity = SequenceMatcher(None, old, new).ratio()
    if 0.5 <= similarity < 0.98:
        score += 1
        reasons.append("localized_rewrite")
    if not reasons:
        return None
    return {**row, "triage_score": score, "triage_reasons": reasons,
            "text_similarity": round(similarity, 3), "audit_status": "pending"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    if args.top < 1:
        raise ValueError("top must be positive")
    raw = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    ranked = [x for row in raw if (x := triage(row)) is not None]
    ranked.sort(key=lambda r: (-r["triage_score"], -r["text_similarity"], r["candidate_id"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ranked[:args.top]) + "\n", encoding="utf-8")
    print(f"Ranked {len(ranked)} of {len(raw)} unreviewed hunks; wrote top {min(len(ranked), args.top)}")
    print("Triage is heuristic; audit every retained hunk and sample excluded hunks for missed facts.")


if __name__ == "__main__":
    main()
