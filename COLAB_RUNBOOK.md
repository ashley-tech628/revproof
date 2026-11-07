# Running RevProof on Colab

Open `revproof_colab.ipynb` in Google Colab, choose **Runtime → Change runtime type → GPU**, and run cells in order. Upload `revproof.zip` when prompted. The notebook checks the GPU and installs `requirements-colab.txt` before loading any model. Keep a copy of all predictions and run metadata outside the ephemeral Colab runtime.

Before preparing real data, the optional toy GPU smoke cell can test QLoRA model loading, training, adapter saving, and inference. It uses four fictional training rows and is never a research result.

## Gate 1: audited data

Use the current fact-card pilot `data/real_v2.jsonl` for replication, with `--prompt-version v2`. It contains 15 train, 7 dev, and 4 test revision groups; this is a feasibility pilot, not enough evidence for a generalization claim. Its test predictions have already been inspected. Freeze a new holdout before developing and comparing a new architecture. `read_records` validates the schema and common leakage cases. Manually check old/new facts, version, scope, and protected facts. Do not use raw Git hunks as labels.

Suggested pilot target, subject to data quality: several dozen *independent* train revisions, multiple dev revisions, and at least one held-out release or repository for test. Questions derived from the same revision count as one independent group. If the data are smaller, label the run a feasibility check.

## Gate 2: SFT and controlled generation

The full-paragraph QLoRA SFT, template, and same-base prompt-only conditions have already run on DataHub. Their v2 pilot results are summarized in [the architecture decision](research/architecture_decision.md). For the next experiment, compare Qwen edit-script output and an encoder-decoder edit-script model under the same frozen data, fact-card, and generation budget. Record the GPU, package versions, seed, group counts, loss, generation budget, and every prediction. Do not claim that a larger backbone improves editing until those controlled ablations have run.

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

## Colab Pro compute check

The current DataHub 24 GB Blackwell GPU already works for the QLoRA pilot. A Colab Pro purchase does not guarantee any specific GPU or uninterrupted runtime; [Google's FAQ](https://research.google.com/colaboratory/faq.html) says GPU type, limits, and availability vary. Before moving a run, select a GPU runtime and inspect what was actually assigned:

```python
import subprocess, torch
print(subprocess.check_output(["nvidia-smi"], text=True))
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

Use the small encoder-decoder experiment on DataHub first. Move the 3B experiment to Colab Pro only when the assigned GPU passes a model-loading and short-training-step memory check. Save the exact dataset hash, checkpoint, environment versions, and outputs outside the transient runtime.

## What to write in the midterm

The assignment asks for revised proposal content, detailed mathematical/model/algorithm setup, and discussion of novelty, advantages, and limitations in about five pages excluding references. Report only code and experiments that were actually run. The v2 GPU pilot currently shows no exact-match gain over the template on its small dev or test sets. The proposed copy-constrained editor is a hypothesis until its ablations and semantic audit run.
