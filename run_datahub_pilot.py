"""Run the frozen real-data generator comparison on one DataHub GPU.

The test split is evaluated once under the same predeclared configuration as
dev. No model or parameter selection is made from test outputs.
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
        file.write(json.dumps({"name": name, "command": command, "started_utc": started,
                               "finished_utc": datetime.now(timezone.utc).isoformat(),
                               "exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; earlier outputs remain intact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baselines", "sft", "filter", "all"], default="all")
    ap.add_argument("--data", default="data/real.jsonl")
    ap.add_argument("--output", default="runs/pilot")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--candidate-budget", type=int, default=4)
    ap.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    if args.candidate_budget < 2:
        ap.error("candidate-budget must be at least 2")
    root = Path(__file__).parent
    data = (root / args.data).resolve()
    rows = read_records(data)
    if any(r["source_type"] != "real" for r in rows):
        raise ValueError("the real experiment cannot include synthetic or unreviewed rows")
    if args.prompt_version == "v2" and any(r.get("prompt_version") != "v2_fact_card" for r in rows):
        raise ValueError("v2 experiment requires a fact card for every row")
    if args.prompt_version == "v1" and any(r.get("prompt_version") == "v2_fact_card" for r in rows):
        raise ValueError("fact-card rows require --prompt-version v2")
    groups = {s: {r["group"] for r in rows if r["split"] == s}
              for s in ("train", "dev", "test")}
    if any(not value for value in groups.values()):
        raise ValueError("train, dev, and test must each contain a revision group")
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "groups": {k: len(v) for k, v in groups.items()},
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "candidate_budget": args.candidate_budget, "max_new_tokens": 384,
        "prompt_version": args.prompt_version,
        "method": "same-backbone template, prompt, QLoRA SFT; fixed-budget filtered prompt/SFT",
        "test_policy": "one predeclared test run per method; no tuning on test",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        previous.setdefault("prompt_version", "v1")
        if previous != protocol:
            raise ValueError(f"{output} contains outputs from a different protocol; choose a new output path")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    log = output / "commands.jsonl"
    if args.stage in {"baselines", "sft", "filter", "all"}:
        run(root, log, "gpu_preflight", "check_datahub_runtime.py")
    if args.stage in {"baselines", "all"}:
        for split in ("dev", "test"):
            run(root, log, f"template_{split}", "evaluate_generator.py",
                "--data", data, "--split", split, "--template-only",
                "--prompt-version", args.prompt_version,
                "--output", output / f"template_{split}.jsonl")
            run(root, log, f"prompt_{split}", "evaluate_generator.py",
                "--data", data, "--split", split, "--model", args.model,
                "--prompt-version", args.prompt_version,
                "--max-new-tokens", 384, "--seed", args.seed,
                "--output", output / f"prompt_{split}.jsonl")
    adapter = output / "sft" / "adapter"
    if args.stage in {"sft", "all"}:
        run(root, log, "sft_train", "train_generator.py",
            "--data", data, "--output", output / "sft", "--model", args.model,
            "--prompt-version", args.prompt_version,
            "--epochs", args.epochs, "--gradient-accumulation", args.gradient_accumulation,
            "--max-length", 1024, "--seed", args.seed)
        for split in ("dev", "test"):
            run(root, log, f"sft_{split}", "evaluate_generator.py",
                "--data", data, "--split", split, "--model", args.model,
                "--prompt-version", args.prompt_version,
                "--adapter", adapter, "--max-new-tokens", 384,
                "--seed", args.seed, "--output", output / f"sft_{split}.jsonl")
    if args.stage in {"filter", "all"}:
        if not adapter.exists():
            raise FileNotFoundError(f"SFT adapter not found: {adapter}")
        for split in ("dev", "test"):
            for method, adapter_args in (("prompt", []), ("sft", ["--adapter", adapter])):
                run(root, log, f"{method}_filter{args.candidate_budget}_{split}",
                    "evaluate_generator.py", "--data", data, "--split", split,
                    "--prompt-version", args.prompt_version,
                    "--model", args.model, *adapter_args,
                    "--num-candidates", args.candidate_budget, "--filter",
                    "--max-new-tokens", 384, "--seed", args.seed,
                    "--output", output / f"{method}_filter{args.candidate_budget}_{split}.jsonl")
    run(root, log, "aggregate", "summarize_pilot.py", "--runs", output)
    if args.stage == "all":
        run(root, log, "candidate_diagnostics", "analyze_pilot_artifacts.py",
            "--data", data, "--runs", output,
            "--output", output / "diagnostics")


if __name__ == "__main__":
    main()
