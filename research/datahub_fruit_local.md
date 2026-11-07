# DataHub local edit gate: commands and interpretation

This experiment predicts only the replacement for one old source sentence.
The gold target supplies that sentence's location, so this is an
**oracle-locus upper bound**, not a deployable end-to-end updater. The
publisher's train shard 00000 and development shard 00009 remain separate;
the later-period gold test is untouched. Copying untouched sentences is a
deterministic reconstruction step. The FRUIT labels are silver and do not by
themselves establish support from the supplied evidence.

## Install the small code update

Upload `revproof_fruit_local_gate.zip` to the DataHub home directory. Then:

```bash
cd ~/revproof
unzip -o ~/revproof_fruit_local_gate.zip -d ~/revproof
.venv_blackwell/bin/python -m unittest test_fruit_local.py test_fruit_data.py test_fruit_gate.py -v
test -s data/fruit/train_00000.jsonl
test -s data/fruit/dev_00009.jsonl
```

This zip contains only source code and documentation; it leaves your dataset,
virtual environment, checkpoints, and prior results in place.

## Phase 1: extract and inspect (CPU, under a minute)

```bash
cd ~/revproof
.venv_blackwell/bin/python -u fruit_local.py \
  --train data/fruit/train_00000.jsonl \
  --dev data/fruit/dev_00009.jsonl \
  --output data/fruit/local_gate | tee runs/fruit_local_extract.log
cat data/fruit/local_gate/manifest.json
```

Expected on the converted files used locally: 2,940/11,342 training and
2,955/11,559 development rows kept. The converter validates every kept
row's full-document reconstruction and creates `dev_audit_40.csv`. Inspect
that CSV before interpreting any evaluation: mark `evidence_support`,
`meaningful_change`, and `silver_quality` as `yes`, `no`, or `unclear`, with
short notes. Judge support using the provided evidence, not your own world
knowledge. For example, development ID 47804210 contains malformed markup
(`clubĐồng Tháp`); a structurally valid label can still be bad supervision.
Keep the unmodified audit sample and your annotated copy as separate files.

## Phase 2: training mechanics positive control (GPU)

This deliberately trains on eight short training rows and evaluates those
*same* rows. It only checks whether the implementation can memorize a tiny
set; any score here is **not** held-out generalization. Existing FLAN-T5-XL
cache and `.venv_blackwell` can be reused.

```bash
cd ~/revproof
set -o pipefail
export HF_HOME="$HOME/.cache/huggingface"
.venv_blackwell/bin/python -u train_fruit_local.py \
  --train data/fruit/local_gate/local_train_00000.jsonl \
  --dev data/fruit/local_gate/local_dev_00009.jsonl \
  --positive-control --max-steps 120 --logging-steps 5 \
  --output runs/fruit_local_control120 2>&1 | tee runs/fruit_local_control120_train.log
.venv_blackwell/bin/python -u evaluate_fruit_local.py \
  --data data/fruit/local_gate/local_train_00000.jsonl \
  --adapter runs/fruit_local_control120/adapter \
  --positive-control --output runs/fruit_local_control120_compare \
  2>&1 | tee runs/fruit_local_control120_eval.log
cat runs/fruit_local_control120_compare/summary.json
```

If exact replacements stay near zero on these eight overlapping rows, stop.
Inspect `predictions.jsonl`, prompt and target serialization, adapter loading,
and decoding before more GPU steps. Training loss alone is insufficient.

## Phase 3: held-out gate, only after Phase 2 works

Freeze the 40-row audit sample before choosing a held-out subset. The same
FLAN-T5-XL backbone, prompt, quantization, and LoRA modules are used as in
the mechanics control. This changes the target from full article to local
replacement; it does not claim a model-family improvement.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u train_fruit_local.py \
  --train data/fruit/local_gate/local_train_00000.jsonl \
  --dev data/fruit/local_gate/local_dev_00009.jsonl \
  --max-train-rows 2000 --max-steps 500 --logging-steps 5 \
  --output runs/fruit_local_gate500 2>&1 | tee runs/fruit_local_gate500_train.log
.venv_blackwell/bin/python -u evaluate_fruit_local.py \
  --data data/fruit/local_gate/local_dev_00009.jsonl \
  --adapter runs/fruit_local_gate500/adapter \
  --max-rows 100 --output runs/fruit_local_gate500_compare \
  2>&1 | tee runs/fruit_local_gate500_eval.log
cat runs/fruit_local_gate500_compare/summary.json
```

Report paired copy/base/editor results on exactly the same token-eligible
development IDs. Inspect their replacement texts and manually adjudicate
whether newly asserted facts are supported. `replacement_exact`, `full_exact`,
word LCS, and added-token F1 are silver-reference diagnostics only. The
`full_exact` result here receives the oracle location, so it cannot be
compared as an end-to-end score with the earlier unrestricted article task.
If this stage works, the next experiment learns to locate the sentence and
abstain when evidence is insufficient.

For sharing summaries/predictions without multi-gigabyte model files:

```bash
cd ~/revproof
tar -czf ~/revproof_fruit_local_results.tgz \
  data/fruit/local_gate/manifest.json \
  data/fruit/local_gate/dev_audit_40.csv \
  runs/fruit_local_control120/protocol.json \
  runs/fruit_local_control120/train_metrics.json \
  runs/fruit_local_control120_compare/summary.json \
  runs/fruit_local_control120_compare/predictions.jsonl
```

Add the `runs/fruit_local_gate500*` summary and prediction paths to that
archive only if Phase 3 was actually run. Keep the adapter checkpoint on
DataHub and do not include it in the lightweight results archive.
