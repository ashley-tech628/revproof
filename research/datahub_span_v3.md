# DataHub: boundary refinement experiment v3

The archived 100-row comparison found 20/100 full silver exact matches for
v2 forced editing versus 9/100 for v1. On v2, only 52/100 final candidate
sets contained the exact silver character span. V3 tests whether a classifier
that scores **complete** start/end pairs can improve that bottleneck. This is
an experiment, not a guaranteed accuracy gain. It reuses the trained v2
selector, QA locator, and FLAN-T5-XL infiller. No new generator training is
needed. The next 100 development IDs were frozen before writing v3 code.

Run in a DataHub **GPU pod terminal**. Upload `revproof_span_v3.zip` to `~`
through JupyterLab. The zip contains code and the frozen ID manifest, not
model weights or publisher data. Keep `runs/span_selector_v2/model`,
`runs/span_locator_v2/model`, `runs/span_infiller_v2/adapter`, and
`data/fruit/span_v2/{span_train_v2,span_dev_v2}.jsonl`.

## 1. Install and verify

```bash
cd ~/revproof
unzip -o ~/revproof_span_v3.zip -d ~/revproof
test -x .venv_blackwell/bin/python
test -s data/fruit/span_v2/span_train_v2.jsonl
test -s data/fruit/span_v2/span_dev_v2.jsonl
test -s runs/span_selector_v2/model/config.json
test -s runs/span_locator_v2/model/config.json
test -s runs/span_infiller_v2/adapter/adapter_config.json
.venv_blackwell/bin/python -m unittest test_span_editor_v3.py -v
nvidia-smi
df -h . /tmp
mkdir -p runs
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
set -o pipefail
```

Stop if a `test` command fails. Do not run training from the DSMLP login
node. The RoBERTa download cache goes to `/tmp`; its saved trained model is
under `runs/` on the workspace volume. Check disk space before training.

## 2. Train the boundary ranker

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.train_boundary_v3 \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --max-train-rows 12000 --epochs 2 \
  --output runs/span_boundary_v3 \
  2>&1 | tee runs/span_boundary_v3.log
test -s runs/span_boundary_v3/model/config.json
cat runs/span_boundary_v3/train_metrics.json
```

The trainer logs encoding progress and training loss. It saves the final
model only, so an interrupted run must be repeated. Training loss alone says
nothing about exact edits. The protocol records hashes and training IDs.

## 3. Screen candidate recall on an already inspected set

This probe uses the **original, already inspected** v1 development IDs. It
does not generate replacement text and does not open the next frozen 100 IDs.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.probe_v3 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --seen-summary results/span_v1_fresh_dev/summary.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --boundary runs/span_boundary_v3/model \
  --output runs/span_v3_seen_probe \
  2>&1 | tee runs/span_v3_seen_probe.log
cat runs/span_v3_seen_probe/summary.json
```

The comparable v2 eight-candidate probe found the gold span in **65/100**
seen rows. If v3 does not exceed 65, stop here and send the probe summary;
the boundary ranker has not demonstrated better span coverage. If it exceeds
65, continue. Even better recall is only an upper bound on final correctness.

## 4. Evaluate v2 and v3 on the newly frozen 100 IDs

Run both on exactly `next_dev_100_v3_frozen.json`. The v2 evaluator's new
`--boundary` option activates v3 proposals and writes `v3_*` conditions.
Without the option it preserves the original v2 path. Each output directory
stores completed rows; rerunning the exact same command resumes. These
evaluations can be long because each row calls FLAN up to eight times.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.evaluate_v2 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --ids-file results/span_v2_frozen_dev/next_dev_100_v3_frozen.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --adapter runs/span_infiller_v2/adapter \
  --output runs/span_v2_v3frozen100 \
  2>&1 | tee runs/span_v2_v3frozen100.log
```

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.evaluate_v2 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --ids-file results/span_v2_frozen_dev/next_dev_100_v3_frozen.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --adapter runs/span_infiller_v2/adapter \
  --boundary runs/span_boundary_v3/model \
  --output runs/span_v3_frozen100 \
  2>&1 | tee runs/span_v3_frozen100.log
```

## 5. Pair the outcomes and package diagnostics

```bash
cd ~/revproof
.venv_blackwell/bin/python -m span_editor.compare_v3 \
  --v2 runs/span_v2_v3frozen100/predictions.jsonl \
  --v3 runs/span_v3_frozen100/predictions.jsonl \
  --output runs/span_v2_v3_paired
cat runs/span_v2_v3_paired/paired_summary.json
cat runs/span_v2_v3frozen100/summary.json
cat runs/span_v3_frozen100/summary.json
```

The paired report counts wins and regressions, rather than relying on two
unpaired totals. `audit_blind.csv` is for manual factual review. Fill it
before looking at `audit_key_keep_separate.json`. Silver exact match alone
does not measure factual support.

```bash
cd ~/revproof
tar -czf ~/revproof_span_v3_results.tgz \
  data/fruit/span_v2/manifest.json \
  runs/span_selector_v2/protocol.json \
  runs/span_locator_v2/protocol.json \
  runs/span_infiller_v2/protocol.json \
  runs/span_boundary_v3/protocol.json \
  runs/span_boundary_v3/train_metrics.json \
  runs/span_v3_seen_probe/summary.json \
  runs/span_v2_v3frozen100/evaluation_protocol.json \
  runs/span_v2_v3frozen100/summary.json \
  runs/span_v2_v3frozen100/predictions.jsonl \
  runs/span_v3_frozen100/evaluation_protocol.json \
  runs/span_v3_frozen100/summary.json \
  runs/span_v3_frozen100/predictions.jsonl \
  runs/span_v2_v3_paired/paired_summary.json \
  runs/span_v2_v3_paired/audit_blind.csv
ls -lh ~/revproof_span_v3_results.tgz
```

If stopping after step 3, package only the boundary protocol, metrics, and
seen-probe summary. Keep model weights on DataHub; the archive does not
need them for result analysis.

## After this experiment

Boundary ranking addresses only part of the error: 27 of the prior 80
nonexact cases already had the exact silver span but the infiller did not
produce the silver phrase. Numeric updates were 0/15. If v3 improves span
coverage but barely improves exact match, next isolate **replacement
generation** using gold-location development diagnostics and an evidence
copy/extraction candidate. If the selector misses the right evidence, improve
retrieval. Make those decisions from the paired outputs and manual audit,
not from training loss or model size alone.
