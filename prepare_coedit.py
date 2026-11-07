"""Stream CoEdIT train rows as editing pretraining data, never grounded labels."""
import argparse
import json
from pathlib import Path


def convert_row(row):
    source, target = row.get("src"), row.get("tgt")
    if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
        raise ValueError("CoEdIT row lacks nonempty src/tgt")
    return {"id": f"coedit:{row['_id']}", "source_dataset": "grammarly/coedit",
            "source_split": "train", "task": row.get("task"),
            "model_input": source, "target_text": target,
            "training_role": "editing_pretraining_only", "grounded": False,
            "evidence_text": None, "evidence_url": None,
            "license": "apache-2.0"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 streams the entire public train split")
    args = ap.parse_args()
    if args.max_rows < 0:
        ap.error("max-rows must be nonnegative")
    from datasets import load_dataset
    dataset = load_dataset("grammarly/coedit", split="train", streaming=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            count += 1
            if args.max_rows and count >= args.max_rows:
                break
    print(json.dumps({"rows": count, "output": str(output),
                      "training_role": "editing_pretraining_only"}))


if __name__ == "__main__":
    main()
