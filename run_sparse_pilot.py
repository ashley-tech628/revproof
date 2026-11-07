"""Run a dev-only sparse-edit ablation on a single GPU.

The v2 test set was already inspected. This runner deliberately does not use it.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from revproof_data import read_records


def run(root, log, name, script, *args):
    command = [sys.executable, str(root / script), *map(str, args)]
    print("\n$", " ".join(command), flush=True)
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=root)
    with log.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"name": name, "command": command,
                               "started_utc": started,
                               "finished_utc": datetime.now(timezone.utc).isoformat(),
                               "exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["causal", "seq2seq"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--stage", choices=["prompt", "sft", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    args = ap.parse_args()
    root = Path(__file__).parent
    data = (root / args.data).resolve()
    rows = read_records(data)
    if any(r["source_type"] != "real" or r.get("prompt_version") != "v2_fact_card"
           for r in rows):
        raise ValueError("pilot requires audited real v2 fact cards")
    if not any(r["split"] == "train" for r in rows) or not any(r["split"] == "dev" for r in rows):
        raise ValueError("train and dev splits must be nonempty")
    model = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if args.backend == "causal"
                           else "google/flan-t5-large")
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "model": model, "backend": args.backend,
                "task": "single-window exact-copy JSON edit script",
                "splits_evaluated": ["dev"],
                "test_policy": "v2 test has been inspected; no test call in this runner",
                "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
                "gradient_accumulation": args.gradient_accumulation,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "max_new_tokens": args.max_new_tokens,
                "num_candidates": 1, "filter": False}
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output contains a different protocol; choose a new directory")
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    log = output / "commands.jsonl"
    common = ["--data", data, "--backend", args.backend, "--model", model]
    if args.stage in {"prompt", "all"}:
        run(root, log, "prompt_dev", "evaluate_sparse_editor.py", *common,
            "--split", "dev", "--seed", args.seed,
            "--max-new-tokens", args.max_new_tokens,
            "--output", output / "prompt_dev.jsonl")
    if args.stage in {"sft", "all"}:
        run(root, log, "train", "train_sparse_editor.py", *common,
            "--output", output / "sft", "--seed", args.seed,
            "--epochs", args.epochs, "--lr", args.lr,
            "--gradient-accumulation", args.gradient_accumulation,
            "--max-input-length", args.max_input_length,
            "--max-target-length", args.max_target_length)
        run(root, log, "sft_dev", "evaluate_sparse_editor.py", *common,
            "--split", "dev", "--adapter", output / "sft" / "adapter",
            "--seed", args.seed, "--max-new-tokens", args.max_new_tokens,
            "--output", output / "sft_dev.jsonl")
    table = []
    for name in ("prompt_dev", "sft_dev"):
        file = output / (name + ".summary.json")
        if file.exists():
            summary = json.loads(file.read_text(encoding="utf-8"))
            table.append({"method": name, "n": summary["n"],
                          "json_rate": summary["json_rate"],
                          "applicable_rate": summary["applicable_rate"],
                          "exact_span_rate": summary["exact_span_rate"],
                          "selected_string_check_rate": summary["selected_string_check_rate"]})
    (output / "generator_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
