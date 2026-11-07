"""Optional DPO experiment, gated on independent train-only human labels."""
import argparse
import json
from pathlib import Path

from run_datahub_pilot import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real.jsonl")
    ap.add_argument("--audit", required=True, help="human-completed train candidate CSV")
    ap.add_argument("--sft-adapter", default="runs/pilot/sft/adapter")
    ap.add_argument("--output", default="runs/pilot_dpo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--min-groups", type=int, default=3)
    args = ap.parse_args()
    root = Path(__file__).parent
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "commands.jsonl"
    pairs = output / "train_preferences.jsonl"
    run(root, log, "prepare_audited_preferences", "prepare_preferences.py",
        "--data", root / args.data, "--audit", root / args.audit, "--output", pairs)
    records = [json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines() if line]
    groups = {r["group"] for r in records}
    if len(records) < args.min_pairs or len(groups) < args.min_groups:
        raise ValueError(f"Only {len(records)} pairs from {len(groups)} groups; "
                         f"need at least {args.min_pairs} pairs and {args.min_groups} groups")
    run(root, log, "dpo_train", "train_dpo.py",
        "--pairs", pairs, "--sft-adapter", root / args.sft_adapter,
        "--output", output / "dpo", "--model", args.model)
    for split in ("dev", "test"):
        run(root, log, f"dpo_{split}", "evaluate_generator.py",
            "--data", root / args.data, "--split", split,
            "--adapter", output / "dpo" / "adapter", "--model", args.model,
            "--max-new-tokens", 384, "--output", output / f"dpo_{split}.jsonl")


if __name__ == "__main__":
    main()
