# Running RevProof on Colab

Open `revproof_colab.ipynb` in Google Colab, choose **Runtime → Change runtime type → GPU**, and run cells in order. Upload `revproof.zip` when prompted. The notebook checks the GPU and installs `requirements-colab.txt` before loading any model. Keep a copy of all predictions and run metadata outside the ephemeral Colab runtime.

## Gate 1: audited data

Upload a `real.jsonl` in the format of `data/flask_pilot_audited.jsonl`. It must contain nonempty, disjoint train and dev revision groups; reserve test groups untouched. The included Flask pilot has only **one dev group** and cannot train a research model. `read_records` validates the schema and common leakage cases. Manually check old/new facts, version, scope, and protected facts. Do not use raw Git hunks as labels.

Suggested pilot target, subject to data quality: several dozen *independent* train revisions, multiple dev revisions, and at least one held-out release or repository for test. Questions derived from the same revision count as one independent group. If the data are smaller, label the run a feasibility check.

## Gate 2: SFT and controlled generation

Run QLoRA SFT, then evaluate template, same-base prompt-only, and SFT on the frozen dev split. Record the GPU, package versions, seed, train/dev group counts, loss, generation budget, and every prediction. `runs/sft/run.json` stores key training metadata. Use `--model Qwen/Qwen3-4B-Instruct-2507` for all model conditions only if GPU memory permits; the 1.5B default is a smaller starting point. If 4B is selected, use Transformers 4.51 or newer as required by its model card.

For each edit, audit whether the target fact is correct, version/scope are correct, protected facts remain true, and the text is fluent. Sample some **rejected** candidates too. Report human validity alongside automatic string pass@N, and distinguish `scalar`, `conditional`, and `cross_sentence` edits.

## Gate 3: DPO if labels support it

Generate four SFT candidates for each **train** input, export the audit CSV, fill `human_valid=true/false`, and create same-prompt pairs. Inspect the count and failure types before DPO; do not train on empty, fake, or trivially malformed negatives. `train_dpo.py` continues from the SFT adapter with TRL's initial-policy reference. Evaluate SFT and DPO on the same frozen dev cases and same decoding budget. Keep test until choices are final.

## Gate 4: downstream data utility

The built-in Flask pair benchmark is a wiring check. A research benchmark needs a fixed candidate pool with old, current, and **same-version irrelevant** documents. Explicitly store the requested version and per-candidate relevance labels. Run BM25 and exact-version-filtered BM25 before any trained reranker. If exact filtering solves all questions, report that honestly; select a justified remaining semantic ambiguity rather than removing metadata only from the baseline.

To compare augmentation sources, create an audit CSV for each generator. Use the same number of generated train examples per source:

```bash
python make_reranker_pairs.py --data data/real.jsonl --split train --audit data/sft_train_audit.csv --generated-source sft --max-generated-per-row 1 --output runs/sft_rerank_train.jsonl
python make_reranker_pairs.py --data data/real.jsonl --split dev --output runs/rerank_dev.jsonl
python train_reranker.py --train runs/sft_rerank_train.jsonl --dev runs/rerank_dev.jsonl --output runs/reranker_sft
python retrieval_eval.py --benchmark data/retrieval_test.jsonl --method version_reranker --model runs/reranker_sft/model --output runs/reranker_sft_test.json
```

Repeat from the same cross-encoder initialization for real-only, template, prompt-only, SFT, and any DPO augmentation. Match train-example counts. `retrieval_eval.py` reports query-level metrics; use revision-group resampling for confidence intervals once there are enough independent groups. Never train the reranker on test gold answers or test-derived generations.

## What to write in the midterm

The assignment asks for revised proposal content, detailed mathematical/model/algorithm setup, and discussion of novelty, advantages, and limitations in about five pages excluding references. Report only code and experiments that were actually run. Current local evidence is a 13-hunk Flask audit, one verified factual revision group, a template check, and a two-document retrieval sanity check. No model gains have yet been measured.
