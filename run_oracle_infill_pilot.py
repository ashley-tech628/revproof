"""Run a dev-only FLAN-T5 replacement-generation diagnostic on one GPU."""
import argparse
import hashlib
import json
from pathlib import Path

from run_sparse_pilot import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/real_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="google/flan-t5-large")
    ap.add_argument("--stage", choices=["prompt", "sft", "all"], default="all")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-target-length", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(__file__).parent
    data = (root / args.data).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "model": args.model, "backend": "seq2seq", "objective": "oracle_infill",
        "oracle_location": True, "location_source": "gold_new_span",
        "splits_evaluated": ["dev"],
        "interpretation": "Diagnostic upper bound with target-location leakage, not deployable.",
        "test_policy": "Previously inspected test is excluded.",
        "epochs": args.epochs, "lr": args.lr,
        "gradient_accumulation": args.gradient_accumulation,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    }
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("output contains a different protocol; choose a new directory")
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    log = output / "commands.jsonl"
    common = ["--data", data, "--model", args.model, "--seed", args.seed]
    if args.stage in {"prompt", "all"}:
        run(root, log, "prompt_dev", "evaluate_oracle_infill.py", *common,
            "--max-new-tokens", args.max_new_tokens,
            "--output", output / "prompt_dev.jsonl")
    if args.stage in {"sft", "all"}:
        run(root, log, "train", "train_sparse_editor.py", *common,
            "--backend", "seq2seq", "--objective", "oracle_infill",
            "--epochs", args.epochs, "--lr", args.lr,
            "--gradient-accumulation", args.gradient_accumulation,
            "--max-input-length", args.max_input_length,
            "--max-target-length", args.max_target_length,
            "--output", output / "sft")
        run(root, log, "sft_dev", "evaluate_oracle_infill.py", *common,
            "--adapter", output / "sft" / "adapter",
            "--max-new-tokens", args.max_new_tokens,
            "--output", output / "sft_dev.jsonl")
    table = []
    for name in ("prompt_dev", "sft_dev"):
        path = output / (name + ".summary.json")
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            table.append({"method": name, "n": summary["n"],
                          "exact_replacement_rate": summary["exact_replacement_rate"],
                          "exact_span_rate": summary["exact_span_rate"],
                          "selected_string_check_rate": summary["selected_string_check_rate"],
                          "oracle_location": True})
    (output / "generator_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
