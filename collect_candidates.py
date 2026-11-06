"""Export changed documentation lines between two local Git tags for human audit."""
import argparse
import json
import subprocess
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local checkout of a public documentation repository")
    ap.add_argument("--old-tag", required=True)
    ap.add_argument("--new-tag", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--path", default="docs", help="git pathspec for docs")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise ValueError("--repo must be a local Git checkout")
    old_commit = git(repo, "rev-parse", "--verify", args.old_tag + "^{commit}").strip()
    new_commit = git(repo, "rev-parse", "--verify", args.new_tag + "^{commit}").strip()
    diff = git(repo, "diff", "--no-ext-diff", "--unified=0", args.old_tag,
               args.new_tag, "--", args.path)
    records = []
    path = None
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@"):
            if current and current["old_lines"] and current["new_lines"]:
                records.append(current)
            current = {"path": path, "hunk": line,
                       "old_lines": [], "new_lines": []}
        elif current and line.startswith("-") and not line.startswith("---"):
            current["old_lines"].append(line[1:])
        elif current and line.startswith("+") and not line.startswith("+++"):
            current["new_lines"].append(line[1:])
    if current and current["old_lines"] and current["new_lines"]:
        records.append(current)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            r.update({"candidate_id": f"{old_commit[:8]}_{new_commit[:8]}_{i:04d}",
                      "old_commit": old_commit, "new_commit": new_commit,
                      "source_type": "real_unreviewed", "audit_status": "pending"})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} unreviewed candidate hunks to {output}")
    print("Manually confirm factual change, scope, question and answers before converting to training records.")


if __name__ == "__main__":
    main()
