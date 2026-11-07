# RevProof: verifiable revision generation for versioned retrieval

RevProof studies whether a conditional autoregressive language model can learn to write accurate, local technical-document revisions from real version changes, and whether those generations improve a fixed retrieval/reranking task. The generator is given the **new fact** and its scope. This is controlled editing, not prediction of future releases.

## Current status

- Local data, split, audit, preference, retrieval, and template-baseline checks pass. See `test_pipeline.py`.
- `data/real.jsonl` now has **26 rows from 22 source-linked revision groups**: 15 train groups (Node.js), 4 dev groups (held-out Node.js release transition), and 3 test groups (Flask). The selected old/new excerpts were verified against official release tags. See `data/pilot_data_audit.md` for selection rules and limitations.
- An earlier user-run DataHub GPU smoke test completed QLoRA SFT and dev inference on fictional toy rows (Qwen2.5-1.5B-Instruct, one epoch, two dev rows). The screenshot reported 2/2 exact spans. Toy scores are pipeline checks, not research results.
- A prompt-only real Flask pilot produced two invalid shortened spans; the template was exact on both. One invalid span initially passed literal checks. The case audit and updated protected-fact annotation are in `data/flask_candidate_audit.md`.
- The expanded real-data DataHub pilot has completed prompt-only generation, three-epoch QLoRA SFT, four-sample filtering, and real-only MiniLM reranker training. The deterministic template is exact on **2/5 dev** and **2/4 test** questions; SFT is exact on **1/5 dev** and **0/4 test**. The trained reranker does not improve top-1 ranking over the pretrained model in this tiny pool. See [the archived run and error analysis](results/datahub_223b776/ANALYSIS.md). DPO and generated-document retrieval augmentation have not been run.
- The [v2 fact-card pilot](data/v2_protocol.md) has been run on a DataHub GPU. Full-paragraph QLoRA SFT has 0/9 dev and 0/4 test exact spans, compared with 4/9 and 1/4 for the deterministic template. The four test outputs have been inspected; they are no longer a fresh holdout for new architecture choices. Exact match and literal checks are not semantic-validity scores.
- A [new architecture plan](research/architecture_decision.md) and `sparse_edit.py` implement a single-window, exact-copy edit-script objective. `train_sparse_editor.py`, `evaluate_sparse_editor.py`, and `run_sparse_pilot.py` provide Qwen causal and FLAN-T5 encoder-decoder ablations. The new model has **not yet been trained on a GPU**, so no gain is claimed. `revproof_sparse_colab.ipynb` is the current Colab workflow; `revproof_colab.ipynb` is the historical v1 workflow.

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

For UCSD DataHub's Blackwell GPU, follow [DATAHUB_RUNBOOK.md](DATAHUB_RUNBOOK.md) and install `requirements-datahub.txt` in a fresh virtual environment without system site packages. Its system PyTorch build lacks `sm_120` kernels.

Run local checks:

```bash
python -m unittest discover -p 'test_*.py' -v
python make_toy_data.py
python evaluate_generator.py --data data/toy.jsonl --split test --template-only --output runs/toy_template.jsonl
```

Toy rows are fictional pipeline checks only. `--allow-synthetic-smoke` is required to train on them and records that the run is not a research result.

## Real data

Each JSONL row has `id`, `group`, `split`, `source_type`, `source_url`, `question`, `answer_before`, `answer_after`, `prefix`, `old_span`, `new_span`, `suffix`, `field`, `old_value`, `new_value`, `version_before`, `version_after`, and `scope`. See `make_toy_data.py` and `data/flask_pilot_audited.jsonl`. Optional `protected_facts` is a list of strings to preserve; `edit_type` is `scalar`, `conditional`, or `cross_sentence`. `z=keep/change` is derived from the two audited answers.

Only source-checked revisions belong in `data/real.jsonl`. The current pilot uses official tagged Node.js and Flask documentation, with normalized line wraps and manually selected question/fact cues; it does **not** independently test the API runtime behavior. Use a stable source permalink, verify both tagged versions and applicable scope, and keep all questions and descendants of the same revision in one group and split. Hold out complete release transitions or document families where possible. The loader detects duplicate IDs, group leakage, identical source-revision leakage, and gold spans that fail annotated literal checks. These checks do not establish semantic validity.

Rebuild and verify the pilot:

```bash
python build_pilot_data.py
python verify_pilot_sources.py --cache ../work/pilot_source_cache
python -m unittest discover -p test_pipeline.py -v
```

The source verifier needs network access once; the committed JSONL dataset does not.

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

On the ECE285 DataHub Blackwell GPU, the frozen pilot runner executes template, prompt-only, QLoRA SFT, and equal-budget four-candidate filtered comparisons on dev and test, then aggregates a table. It records the dataset SHA-256 and every command. This is a small pilot, not a population-level performance estimate:

```bash
.venv_blackwell/bin/python run_datahub_pilot.py --stage all
.venv_blackwell/bin/python run_retrieval_pilot.py
```

For the revised fact-card task, use a **new output directory** and the predeclared [v2 protocol](data/v2_protocol.md):

```bash
.venv_blackwell/bin/python run_datahub_pilot.py --data data/real_v2.jsonl --output runs/v2 --prompt-version v2 --stage all
```

The v2 runner also exports per-example diagnostics and an unlabelled semantic-audit CSV. The original `data/real.jsonl` and its archived results remain immutable.

The exploratory sparse-editor runner evaluates **dev only**, since the v2 test has been viewed. On the working DataHub Blackwell environment, run both model conditions with separate output directories:

```bash
.venv_blackwell/bin/python run_sparse_pilot.py --backend seq2seq --model google/flan-t5-large --output runs/sparse_flan_large --stage all
.venv_blackwell/bin/python run_sparse_pilot.py --backend causal --model Qwen/Qwen2.5-1.5B-Instruct --output runs/sparse_qwen_1p5b --stage all
```

The sparse pilot results are analyzed in [results/sparse_15c6e2f/ANALYSIS.md](results/sparse_15c6e2f/ANALYSIS.md). To isolate replacement generation from failed location prediction, a **diagnostic only** FLAN-T5 run supplies the gold target-derived edit location on dev. Its scores have target-location leakage and are not deployable accuracy:

```bash
.venv_blackwell/bin/python run_oracle_infill_pilot.py --output runs/oracle_flan_large --stage all
```

The [FLAN oracle results](results/oracle_flan_8007a70/ANALYSIS.md) show 0/9 exact dev replacements. A matched Qwen causal condition is available via `run_oracle_infill_pilot.py --backend causal --model Qwen/Qwen2.5-1.5B-Instruct --output runs/oracle_qwen_1p5b --stage all`; it remains a target-location-leaking diagnostic.

The next GPU run is paused while the [research reset](research/evidence_grounded_revision_reset.md) addresses task identifiability, evidence provenance, data scale, and evaluation. The Qwen command above remains available as a limited model-family diagnostic.

The [large-data plan](research/dataset_sources.md) distinguishes FRUIT evidence-grounded examples, CoDocBench technical-domain silver revisions, and CoEdIT editing pretraining. `multispan_edit.py` adds exact-copy multi-span labels and an evidence provenance gate; external rows are not yet mixed into the v2 training/evaluation pipeline.

The runnable [CoDocBench training pilot](research/codoc_training_runbook.md) starts with three-example training positive controls, then compares same-project-split full-docstring generation to multi-span edit-plan generation on the publisher's train portion. This is unaudited code-conditioned editing; the proposed verified evidence-grounded tagger/infiller remains future work.

The retrieval runner builds fixed pools with same-version distractors, evaluates BM25 and a pretrained cross-encoder, then trains the same cross-encoder on real-only pairs. DPO requires an independently completed train-candidate audit; run `run_dpo_pilot.py --audit data/sft_train_audit.csv` only after that audit. Do not treat literal pass/fail labels as human preference labels.

Individual commands remain available:

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

The original Flask-only pilot has one factual revision group with two questions; the template is exact on both. The expanded real pilot adds Node.js train/dev transitions and Flask test groups. Its GPU run found no SFT gain over the template and no reranker gain over its pretrained backbone. Literal checks can accept incomplete edits. Fixed-pool BM25 and version-filtered BM25 include same-version distractors, but the held-out pools remain small and version filtering saturates them. Generated-document downstream data utility has not been measured. Some complex target paragraphs contain facts absent from the input cue, so the task specification must be improved before interpreting model failures as solely model limitations. Publication-level claims require more independent release transitions, document families, calibrated human assessment, and uncertainty analysis by revision group.

Implementation references: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [PEFT 4-bit training](https://huggingface.co/docs/peft/developer_guides/quantization), [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer), and [Sentence Transformers cross-encoders](https://www.sbert.net/docs/cross_encoder/training_overview.html).
