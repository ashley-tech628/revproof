"""Fixed-pool retrieval and real-only cross-encoder reranker pilot."""
import argparse
from pathlib import Path

from revproof_data import read_records
from run_datahub_pilot import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real.jsonl")
    ap.add_argument("--output", default="runs/pilot_retrieval")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=float, default=2)
    args = ap.parse_args()
    root = Path(__file__).parent
    data = (root / args.data).resolve()
    rows = read_records(data)
    if any(r["source_type"] != "real" for r in rows):
        raise ValueError("real-only reranker pilot requires audited real rows")
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "commands.jsonl"
    for split in ("train", "dev"):
        run(root, log, f"reranker_pairs_{split}", "make_reranker_pairs.py",
            "--data", data, "--split", split,
            "--output", output / f"pairs_{split}.jsonl")
    for split in ("dev", "test"):
        pool = output / f"pool_{split}.jsonl"
        run(root, log, f"fixed_pool_{split}", "make_retrieval_pool.py",
            "--data", data, "--split", split, "--output", pool)
        for method in ("bm25", "version_bm25", "reranker", "version_reranker"):
            run(root, log, f"pretrained_{method}_{split}", "retrieval_eval.py",
                "--benchmark", pool, "--method", method, "--model", args.reranker_model,
                "--output", output / f"pretrained_{method}_{split}.json")
    run(root, log, "train_reranker_real_only", "train_reranker.py",
        "--train", output / "pairs_train.jsonl",
        "--dev", output / "pairs_dev.jsonl",
        "--output", output / "reranker_real_only",
        "--model", args.reranker_model, "--epochs", args.epochs)
    trained = output / "reranker_real_only" / "model"
    for split in ("dev", "test"):
        for method in ("reranker", "version_reranker"):
            run(root, log, f"trained_{method}_{split}", "retrieval_eval.py",
                "--benchmark", output / f"pool_{split}.jsonl", "--method", method,
                "--model", trained,
                "--output", output / f"trained_{method}_{split}.json")


if __name__ == "__main__":
    main()
