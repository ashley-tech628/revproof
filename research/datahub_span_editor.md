# DataHub experiment: locate a short edit, generate a phrase, copy the rest

## Research question and limits

The previous oracle-locus FLAN-T5-XL editor achieved 5/95 exact replacement
on a fixed FRUIT-WIKI silver development slice. Its five exact examples came
from the 31 single-span gold edits in that slice. Several nonexact edits
nevertheless conveyed a useful evidence-supported change, while other edits
were wrong or damaged Unicode. See the [case analysis](fruit_local_gate500_analysis.md).

This experiment asks whether changing the *output representation* improves
supported local edits while keeping the 3B FLAN-T5-XL backbone. The infiller
emits only one missing phrase. The program copies everything else. A separate
RoBERTa QA locator predicts where to edit. This follows the general
detect-then-correct idea and copy-heavy editing work, but this implementation
is our own experiment, not a reproduction of their reported systems:
[FRUIT](https://aclanthology.org/2022.naacl-main.269/),
[EdiT5](https://aclanthology.org/2022.findings-emnlp.156/),
[LaserTagger](https://aclanthology.org/D19-1510/),
[Detector–Corrector](https://aclanthology.org/2024.eamt-1.18/), and
[SQuAD 2.0](https://arxiv.org/abs/1806.03822).

The strict converter keeps 1,096/2,940 train and 1,082/2,955 dev rows from
the earlier one-source-sentence local dataset. It requires a nonempty old and
new phrase of at most 80 characters, at least 40% of the old sentence
retained, stable boundary whitespace, and exact sentence/full-article
roundtrips. Thus the new task covers a **subset** of local edits; it cannot
measure performance on omitted long, multispan, or insertion-only cases.
All FRUIT labels in this experiment are silver. The later-period gold test is
not used. The locator and infiller training targets are gold-derived; only
the final `pipeline` evaluation is end-to-end at inference.

## 1. Update the DataHub project and check disk/GPU

Upload `revproof_span_editor_v1.zip` to your DataHub home directory using
JupyterLab. Then run in a **GPU pod terminal**:

```bash
cd ~/revproof
unzip -o ~/revproof_span_editor_v1.zip -d ~/revproof
command -v nvidia-smi && nvidia-smi
df -h . /tmp
du -sh ~/.cache/huggingface .venv_blackwell runs 2>/dev/null
.venv_blackwell/bin/python -m unittest test_span_editor.py -v
```

If `nvidia-smi` is unavailable, restart DataHub with a GPU allocation before
training. The existing `.venv_blackwell` and cached FLAN-T5-XL can be reused.
Keep at least about 2 GB of free course-workspace space for the locator,
adapter, checkpoints, predictions and logs. RoBERTa downloads use `/tmp`
via `--cache-dir` so they do not fill the course quota. `/tmp` is ephemeral;
the final locator is saved in `runs/`.

## 2. Convert the existing FRUIT local data

If `data/fruit/local_gate/` already contains the two local JSONL files, run
only the second command. Otherwise reconstruct them from your downloaded
FRUIT shards first:

```bash
cd ~/revproof
.venv_blackwell/bin/python fruit_local.py \
  --train data/fruit/train_00000.jsonl \
  --dev data/fruit/dev_00009.jsonl \
  --output data/fruit/local_gate
.venv_blackwell/bin/python -u fruit_span.py \
  --train data/fruit/local_gate/local_train_00000.jsonl \
  --dev data/fruit/local_gate/local_dev_00009.jsonl \
  --output data/fruit/span_v1
```

The converter writes `manifest.json` with input/output SHA-256 hashes and
filter counts. Do not use the same output directory for a changed extraction
rule. The `span_*` files include gold location labels for training and
evaluation diagnostics; the end-to-end inference prompt does not read them.

## 3. Small overlapping training control

This checks that the new infiller can learn the short output format. It
trains on eight eligible train examples and evaluates on those *same eight*
examples; do not report it as generalization. It takes roughly minutes to
tens of minutes depending on the DataHub GPU and cache state.

```bash
cd ~/revproof
set -o pipefail
export TOKENIZERS_PARALLELISM=false
.venv_blackwell/bin/python -u -m span_editor.train_infiller \
  --train data/fruit/span_v1/span_train_00000.jsonl \
  --dev data/fruit/span_v1/span_dev_00009.jsonl \
  --positive-control --max-steps 120 --logging-steps 5 \
  --output runs/span_infill_control120 2>&1 | tee runs/span_infill_control120.log
.venv_blackwell/bin/python -u -m span_editor.evaluate \
  --data data/fruit/span_v1/span_train_00000.jsonl \
  --adapter runs/span_infill_control120/adapter \
  --positive-control --output runs/span_control_eval \
  2>&1 | tee runs/span_control_eval.log
cat runs/span_control_eval/summary.json
```

Inspect several `predictions.jsonl` rows. If the trained oracle infiller
cannot outperform the unchanged model on the overlapping control, diagnose
the prompt/labels before spending GPU time on the full run.

## 4. Train the two components

The locator uses fast-tokenizer character offsets, one gold positive sentence
and up to three hard negative source sentences per example. It learns a
SQuAD-style no-span label on negative sentences. Its output includes the
sentence index and character interval. Training saves one final model, not
full optimizer checkpoints, to reduce course storage.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.train_locator \
  --train data/fruit/span_v1/span_train_00000.jsonl \
  --dev data/fruit/span_v1/span_dev_00009.jsonl \
  --output runs/span_locator_v1 --epochs 3 \
  --cache-dir /tmp/revproof_locator_cache \
  2>&1 | tee runs/span_locator_v1.log
.venv_blackwell/bin/python -u -m span_editor.train_infiller \
  --train data/fruit/span_v1/span_train_00000.jsonl \
  --dev data/fruit/span_v1/span_dev_00009.jsonl \
  --max-steps 500 --logging-steps 10 \
  --output runs/span_infiller_v1 \
  2>&1 | tee runs/span_infiller_v1.log
```

Each training directory contains `protocol.json`, `train_metrics.json`, and
`model/` or `adapter/`. The infiller also keeps at most two resumable
checkpoints. Training loss is an optimization diagnostic, not an edit-quality
or factual-support score. If a run is interrupted, resume the infiller with
`--resume runs/span_infiller_v1/checkpoints/checkpoint-N` using the *same*
arguments and output directory. The locator has no intermediate checkpoint;
rerun it if interrupted.

## 5. Evaluate on development IDs excluded from the earlier pilot

`--exclude-local-dev` removes the first 100 hash-ranked rows of the earlier
local-development file **by ID** before selecting new span rows. This is more
precise than skipping the first 100 span rows, because the two filters create
different rank lists. Keep the same 100 selected examples for every model
condition. The evaluator records the IDs, file hashes, raw predictions,
locator accuracy (including a lexical sentence-selection baseline), exact
matches, changed count, and edit-content overlap. Metric version 2 does not
award perfect edit-content credit to unchanged output when the only silver
change is punctuation or case; the earlier eight-row control's edit-content
F1 used version 1. Exact-match counts are unchanged by this correction.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.evaluate \
  --data data/fruit/span_v1/span_dev_00009.jsonl \
  --exclude-local-dev data/fruit/local_gate/local_dev_00009.jsonl \
  --exclude-local-rows 100 --max-rows 100 \
  --adapter runs/span_infiller_v1/adapter \
  --locator runs/span_locator_v1/model \
  --output runs/span_v1_fresh_dev \
  2>&1 | tee runs/span_v1_fresh_dev.log
cat runs/span_v1_fresh_dev/summary.json
.venv_blackwell/bin/python -m span_editor.export_audit \
  --predictions runs/span_v1_fresh_dev/predictions.jsonl \
  --output runs/span_v1_fresh_dev/audit_40_blank.csv --count 40
```

The conditions have different meanings:

| Name | Location used at inference | What it measures |
| --- | --- | --- |
| `copy` | none | Old document unchanged |
| `base_oracle` | gold sentence and span | Untrained FLAN phrase generation upper bound |
| `editor_oracle` | gold sentence and span | Trained infiller upper bound |
| `pipeline` | trained locator prediction | End-to-end system |

An exact `editor_oracle` match is not an end-to-end success. Compare
`pipeline` against `copy`, inspect `locator_scores`, and manually annotate
the fixed audit CSV for factual support, preservation and silver target
quality. The script does not automatically call any lexical check a factual
verification. The 40 audit rows are selected by a stable ID hash and include
both exact and nonexact predictions.

To send results for analysis without uploading large weights or checkpoints:

```bash
cd ~/revproof
tar -czf ~/revproof_span_v1_results.tgz \
  data/fruit/span_v1/manifest.json \
  runs/span_infiller_v1/protocol.json runs/span_infiller_v1/train_metrics.json \
  runs/span_locator_v1/protocol.json runs/span_locator_v1/train_metrics.json \
  runs/span_v1_fresh_dev/summary.json \
  runs/span_v1_fresh_dev/predictions.jsonl \
  runs/span_v1_fresh_dev/audit_40_blank.csv
ls -lh ~/revproof_span_v1_results.tgz
```

The audit CSV is blank by design. Human review should record whether the
update is supported by *the provided evidence*, whether untouched facts
survive, and whether the silver target itself is sound. Also record the
fraction of the original local dataset retained by the span filter; do not
present a high subset score as performance on all document updates.
