# CoDocBench code-conditioned editing experiment

This is a **silver-data domain adaptation pilot**, not the final evidence-grounded RevProof system. The official CoDocBench `train.jsonl` contains 2,300 rows from 100 projects; we create a deterministic 90/10 project holdout *within that training file*. The publisher's `test.jsonl` is not loaded. Both model objectives use the same token-length-eligible examples.

The model is Qwen2.5-1.5B-Instruct with 4-bit QLoRA on the working DataHub Blackwell environment. `full` predicts the complete target docstring. `plan` predicts a JSON array of source-offset edits that are applied by deterministic copying. The plan objective is an exploratory multi-span baseline; it is **not yet** the proposed learned tagger plus infiller. Code diff co-change is not guaranteed to justify every new docstring claim.

Upload the latest project zip to DataHub `~`, then run:

```bash
cd ~
unzip -o revproof_training.zip
cd ~/revproof
.venv_blackwell/bin/python -m unittest discover -p 'test_*.py' -q
mkdir -p data/external
curl -L --fail --silent --show-error https://raw.githubusercontent.com/kunpai/codocbench/main/dataset/train.jsonl -o data/external/codocbench_train.jsonl
sha256sum data/external/codocbench_train.jsonl
.venv_blackwell/bin/python prepare_codocbench.py --input data/external/codocbench_train.jsonl --output data/external/codocbench_train_silver.jsonl
```

The raw SHA-256 observed on 2026-11-06 was `9eef860afedcbf059a4f2e408cafbaa6c61ffbf4369e687a9b45ae80f7ac7c6d`. Stop and inspect if it differs; the publisher may have updated the file. The converter should report 2,300 kept rows. Do not call them audited factual examples.

First run the deliberately overlapping **training positive controls** (not research scores):

```bash
.venv_blackwell/bin/python codoc_experiment.py --data data/external/codocbench_train_silver.jsonl --objective full --positive-control --max-steps 120 --output runs/codoc_smoke_full
.venv_blackwell/bin/python codoc_experiment.py --data data/external/codocbench_train_silver.jsonl --objective plan --positive-control --max-steps 120 --output runs/codoc_smoke_plan
```

Read each `dev_summary.json` and `dev_predictions.jsonl`. The three evaluation examples are the *same* three training examples by design. The intended check is that the model can learn and decode these examples. If either stays at zero, inspect target formatting, token lengths, adapter loading, and generation before trusting a longer run; increasing model size is not the next step.

Only after the positive controls work, run the matched silver development comparison:

```bash
.venv_blackwell/bin/python codoc_experiment.py --data data/external/codocbench_train_silver.jsonl --objective full --max-steps 300 --output runs/codoc_full_300
.venv_blackwell/bin/python codoc_experiment.py --data data/external/codocbench_train_silver.jsonl --objective plan --max-steps 300 --output runs/codoc_plan_300
```

Each output directory contains `protocol.json`, `train_metrics.json`, `dev_predictions.jsonl`, `dev_summary.json`, and `adapter/`. Record project counts, eligible row counts and exact/valid-plan results. The two objectives use the same eligible rows and optimizer-step budget, but target lengths and token-level loss differ. This pilot diagnoses feasibility; a final model comparison requires a larger tuned budget, semantic audit, and a new untouched technical-document holdout.

To send back only summaries and a small prediction sample, use `tar` on the four output directories' JSON metadata/predictions and omit adapter checkpoints unless needed for debugging.
