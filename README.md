# RevProof: verifiable revision generation for versioned retrieval

RevProof studies whether a conditional autoregressive language model can learn to write accurate, local technical-document revisions from real version changes, and whether those generations improve a fixed retrieval/reranking task. The generator is given the **new fact** and its scope. This is controlled editing, not prediction of future releases.

## Current status

- Local data, split, audit, preference, retrieval, and template-baseline checks pass. See `test_pipeline.py`.
- `data/flask_pilot_audited.jsonl` has **two questions from one real revision group**, both in dev. The template matches both spans. The audit is documented in `data/flask_candidate_audit.md`.
- No QLoRA, DPO, or reranker GPU training has been run in this repository. Do not report toy scores or unrun stages as model results.
- `revproof_colab.ipynb` gives a cell-by-cell Colab workflow; `COLAB_RUNBOOK.md` and `DATAHUB_RUNBOOK.md` cover both platforms.

## Research pipeline

```text
tagged docs + release notes -> candidate diffs -> human fact/scope audit
    -> revision-group train/dev/test split
    -> template / same-base prompt-only / QLoRA SFT
    -> fixed-N generation + transparent string checks + human audit
    -> optional same-prompt DPO on audited train preferences
    -> fixed candidate-pool retrieval and reranker evaluation
```

The key comparison is simple replacement versus conditional or cross-sentence edits. The program copies text outside the marked span exactly. `protected_facts` checks are literal string checks only; semantic correctness and verifier false positives/negatives require independent human audit.

## Setup

Use Python 3.10+ in a Colab GPU runtime. Install `requirements-colab.txt`. Qwen3 requires Transformers 4.51 or newer. The default model is `Qwen/Qwen2.5-1.5B-Instruct` for a small GPU pilot. If GPU memory permits, use `--model Qwen/Qwen3-4B-Instruct-2507` consistently across prompt-only, SFT, and DPO conditions. Do not compare different backbones as if fine-tuning were the only change.

Run local checks:

```bash
python -m unittest discover -p test_pipeline.py -v
python make_toy_data.py
python evaluate_generator.py --data data/toy.jsonl --split test --template-only --output runs/toy_template.jsonl
```

Toy rows are fictional pipeline checks only. `--allow-synthetic-smoke` is required to train on them and records that the run is not a research result.

## Real data

Each JSONL row has `id`, `group`, `split`, `source_type`, `source_url`, `question`, `answer_before`, `answer_after`, `prefix`, `old_span`, `new_span`, `suffix`, `field`, `old_value`, `new_value`, `version_before`, `version_after`, and `scope`. See `make_toy_data.py` and `data/flask_pilot_audited.jsonl`. Optional `protected_facts` is a list of strings to preserve; `edit_type` is `scalar`, `conditional`, or `cross_sentence`. `z=keep/change` is derived from the two audited answers.

Only manually confirmed factual revisions belong in `data/real.jsonl`. Use a stable source permalink, verify both tagged versions and applicable scope, and keep all questions and descendants of the same revision in one group and split. Hold out complete release transitions or document families where possible. The loader detects duplicate IDs, group leakage, and identical source-revision leakage, but no automatic check replaces a provenance audit.

To collect *unreviewed* candidates from a public repository:

```bash
git clone --filter=blob:none https://github.com/pallets/flask.git /content/flask
python collect_candidates.py --repo /content/flask --old-tag 3.0.3 --new-tag 3.1.0 --path docs --output data/flask_candidates.jsonl
```

The 13 Flask hunks contained one directly usable factual change. Raw diff counts are not training-set counts. Inspect official release notes first, then align old/new documentation around actual behavior changes.

For a large diff, `triage_candidates.py` removes obvious snippet-markup changes and ranks numeric or qualifier changes for human review. It ranked 95 of 780 FastAPI English-documentation hunks from one local tag comparison; the remaining 95 are **still unverified** and excluded hunks must be sampled for false negatives:

```bash
python triage_candidates.py --input data/candidates.jsonl --output data/triaged.jsonl --top 100
```

## Generator experiment

Use the *same model ID* throughout a comparison:

```bash
python train_generator.py --data data/real.jsonl --output runs/sft --model Qwen/Qwen2.5-1.5B-Instruct
python evaluate_generator.py --data data/real.jsonl --split dev --template-only --output runs/template_dev.jsonl
python evaluate_generator.py --data data/real.jsonl --split dev --model Qwen/Qwen2.5-1.5B-Instruct --output runs/prompt_dev.jsonl
python evaluate_generator.py --data data/real.jsonl --split dev --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/sft/adapter --output runs/sft_dev.jsonl
python evaluate_generator.py --data data/real.jsonl --split dev --model Qwen/Qwen2.5-1.5B-Instruct --num-candidates 4 --filter --output runs/prompt_filter_dev.jsonl
python evaluate_generator.py --data data/real.jsonl --split dev --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/sft/adapter --num-candidates 4 --filter --output runs/sft_filter_dev.jsonl
```

Each prediction file includes all candidates and selected output; its `.summary.json` reports exact match, selected string-check rate, and string pass@N by answer relation and edit type. Greedy comparisons use one candidate; filter comparisons use the same four-attempt budget. String pass@N is **not** semantic validity.

## Audited preferences and optional DPO

Sample candidates only from the **train split**, including ones the string checks accept and reject. Annotate every candidate's actual fact, scope, preservation, and fluency:

```bash
python evaluate_generator.py --data data/real.jsonl --split train --adapter runs/sft/adapter --num-candidates 4 --output runs/sft_train_candidates.jsonl
python export_audit.py --data data/real.jsonl --predictions runs/sft_train_candidates.jsonl --output data/sft_train_audit.csv
```

Fill `human_valid` in the CSV with `true` or `false` after independent review. Do not infer it from `passes_string_checks`. Then:

```bash
python prepare_preferences.py --data data/real.jsonl --audit data/sft_train_audit.csv --output data/train_preferences.jsonl
python train_dpo.py --pairs data/train_preferences.jsonl --sft-adapter runs/sft/adapter --output runs/dpo
```

The preference builder requires train-only, real rows and pairs completions under the same prompt, with a simple length-ratio guard. DPO is optional until enough trustworthy positive/negative pairs exist. Evaluate its adapter on the frozen dev/test split using the same `evaluate_generator.py` command.

## Downstream retrieval

`retrieval_eval.py` consumes a fixed JSONL candidate pool: each query has `id`, `group`, `query`, `requested_version`, and `candidates` with `doc_id`, `text`, `version`, `relevant`. Compare `bm25`, `version_bm25`, `reranker`, and `version_reranker` on exactly that pool. It reports MRR@k, Recall@k, top-1 relevance, wrong-version rate, and result coverage. `make_retrieval_benchmark.py` builds a **two-document sanity check** from audited rows; it is not a final benchmark because exact version filtering can solve it by construction.

```bash
python make_retrieval_benchmark.py --data data/flask_pilot_audited.jsonl --split dev --output runs/flask_pair.jsonl
python retrieval_eval.py --benchmark runs/flask_pair.jsonl --method bm25 --output runs/flask_bm25.json
python retrieval_eval.py --benchmark runs/flask_pair.jsonl --method version_bm25 --output runs/flask_version_bm25.json
```

For a genuine downstream data-utility study, curate same-version distractors and hold-out real revisions. `make_reranker_pairs.py` builds labeled training examples from real train/dev revisions and optionally human-audited generated candidates. `train_reranker.py` trains a fixed MiniLM cross-encoder with binary relevance loss. Keep its backbone, training budget, candidate pool, metadata, and generated-example count equal across augmentation methods. `COLAB_RUNBOOK.md` gives the commands.

## Evidence and limits

Current pilot: one Flask factual revision group gives two questions; the template is exact on both. A two-document retrieval sanity check ranks the old version first under BM25 and the current version first under exact version filtering. These checks establish neither SFT improvement nor downstream generalization. Publication-level claims would require more independent revision groups, complex edit types, calibrated human assessment, and held-out repository evaluation.

Implementation references: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [PEFT 4-bit training](https://huggingface.co/docs/peft/developer_guides/quantization), [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer), and [Sentence Transformers cross-encoders](https://www.sbert.net/docs/cross_encoder/training_overview.html).
