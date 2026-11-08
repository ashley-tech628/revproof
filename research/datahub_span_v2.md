# DataHub: reference-planned span editor v2

This is a runnable experiment, not a promised improvement. The official
FRUIT train shards 0–8 are training data. Shard 9 is development. The
later-period `gold_test` remains untouched. V1 and v2 must run on the same
frozen 100 development IDs for a valid paired comparison. Silver exact match
does not establish factual support.

Run these commands in a **DataHub GPU pod terminal** in order. Upload
`revproof_span_v2.zip` into `~` with JupyterLab first. The archive contains
code, this runbook and the frozen ID manifest; it does not contain model
weights or FRUIT publisher data.

## 1. Install code and verify the environment

```bash
cd ~/revproof
unzip -o ~/revproof_span_v2.zip -d ~/revproof
test -x .venv_blackwell/bin/python
.venv_blackwell/bin/python -m unittest test_span_editor.py test_span_editor_v2.py -v
nvidia-smi
df -h . /tmp
mkdir -p runs data/fruit
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
set -o pipefail
```

If `nvidia-smi` fails, this is a CPU pod; launch a GPU pod before training.
Keep the existing FLAN cache and `.venv_blackwell`. The RoBERTa download
cache is directed to `/tmp` by the trainers; their final models live in
`runs/`. Check free workspace space before every long run.

## 2. Get all remaining official training shards

The original `data/fruit/train_00000.jsonl` and `dev_00009.jsonl` from v1
must still be present. These are the processed FRUIT TFRecords converted by
`fruit_data.py`, not the unrelated raw JSONL format.

```bash
test -s data/fruit/train_00000.jsonl
test -s data/fruit/dev_00009.jsonl
.venv_blackwell/bin/python -c 'import tfrecord; print("tfrecord OK")'
for i in {1..8}; do
  s=$(printf '%05d' "$i")
  if [ ! -s "data/fruit/train_${s}.jsonl" ]; then
    curl -fL --retry 5 --retry-delay 5 \
      "https://storage.googleapis.com/gresearch/FRUIT/dataset/train/article_pairs.update.diff.all.text.reference.tfrecords-${s}-of-00010" \
      -o "data/fruit/train_${s}.tfrecord.part" || break
    mv "data/fruit/train_${s}.tfrecord.part" "data/fruit/train_${s}.tfrecord"
    .venv_blackwell/bin/python fruit_data.py \
      --input "data/fruit/train_${s}.tfrecord" \
      --output "data/fruit/train_${s}.jsonl.part" --split train || break
    mv "data/fruit/train_${s}.jsonl.part" "data/fruit/train_${s}.jsonl"
  fi
done
ls -lh data/fruit/train_0000[0-8].jsonl
```

Do not proceed if any of the nine train JSONL files is missing or empty.
The `.part` files are kept if an operation fails so a completed file cannot
be mistaken for a partial conversion. The publisher's official download
location and processed-record format are documented in its
[README](https://github.com/google-research/language/tree/master/language/fruit).

## 3. Convert once, retaining evidence references

```bash
cd ~/revproof
.venv_blackwell/bin/python -u -m span_editor.v2_data \
  --train data/fruit/train_0000[0-8].jsonl \
  --dev data/fruit/dev_00009.jsonl \
  --output data/fruit/span_v2 \
  2>&1 | tee runs/span_v2_data.log
cat data/fruit/span_v2/manifest.json
```

One-shard local verification retained the same 1,096 short-span train rows
and 1,082 development rows as v1, recovering usable heuristic evidence
references for 1,087 and 1,065 of them respectively. Counts on nine train
shards will be higher and must be read from the new manifest. The 100 frozen
IDs are present in the resulting development file; no frozen row content was
inspected for model tuning.

## 4. Train the three v2 components

Run each command in a GPU pod. The default training cap is 12,000 original
short-span rows, enough for the anticipated nine-shard conversion; if the
manifest shows more, increase all three `--max-train-rows` values to the same
number and retain the changed protocols. These are separate v2 output
directories and do not overwrite v1 weights.

```bash
cd ~/revproof
set -o pipefail
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
.venv_blackwell/bin/python -u -m span_editor.train_selector \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --max-train-rows 12000 --epochs 2 \
  --output runs/span_selector_v2 \
  2>&1 | tee runs/span_selector_v2.log
```

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.train_locator_v2 \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --max-train-rows 12000 --epochs 3 \
  --output runs/span_locator_v2 \
  2>&1 | tee runs/span_locator_v2.log
```

Before the long infiller run, measure candidate recall on the **already
inspected** v1 development IDs. This probe generates no replacement text and
must not use `next_dev_100_frozen.json`.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.probe_v2 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --seen-summary results/span_v1_fresh_dev/summary.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --output runs/span_v2_seen_probe \
  2>&1 | tee runs/span_v2_seen_probe.log
cat runs/span_v2_seen_probe/summary.json
```

The v1 locator reached the gold sentence on 80/100 and exact character span
on 30/100. The v2 top-k candidate recall should comfortably exceed these
values before investing in a full infiller run. Top-k recall is easier than
v1's top-1 exact span result, so this is only a preliminary screen: a generated
phrase and final candidate ranking can still fail. If candidate recall is low, send
the probe summary for analysis before running the next training command.
The probe uses the same eight-candidate budget as the final evaluator;
the earlier probe version without this cap measured up to 18 candidates
and must not be used as the eight-candidate result.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.train_infiller_v2 \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --max-train-rows 12000 --max-steps 3000 \
  --save-steps 500 --logging-steps 25 \
  --output runs/span_infiller_v2 \
  2>&1 | tee runs/span_infiller_v2.log
```

The infiller checkpoint is written every 500 steps; at most two are retained.
If it is interrupted, find the latest complete `checkpoint-N` under
`runs/span_infiller_v2/checkpoints/` and rerun the **same** infiller command
with `--resume runs/span_infiller_v2/checkpoints/checkpoint-N`. Do not change
the step count or dataset while resuming. Selector and locator save final
models only; rerun an interrupted component.

Monitor from another terminal with `tail -f runs/span_infiller_v2.log` and
`nvidia-smi`. A decreasing training loss does not imply correct edits.

## 5. Paired evaluation on the frozen next 100 IDs

First run the old system on the **same** IDs. The selection below was checked
locally against `next_dev_100_frozen.json`: excluding the prior 100 local IDs
and then `--skip-rows 100` selects exactly the frozen next 100.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.evaluate \
  --data data/fruit/span_v1/span_dev_00009.jsonl \
  --exclude-local-dev data/fruit/local_gate/local_dev_00009.jsonl \
  --exclude-local-rows 100 --skip-rows 100 --max-rows 100 \
  --adapter runs/span_infiller_v1/adapter \
  --locator runs/span_locator_v1/model \
  --output runs/span_v1_next100 \
  2>&1 | tee runs/span_v1_next100.log
```

Then run v2. This will generate up to eight candidates per row, so allow
substantially more time than v1 evaluation. `v2_oracle_gold_location_and_references`
is a component diagnostic using gold labels. The deployable conditions are
`v2_forced` and `v2_gated` only.
Each completed row is written to `runs/span_v2_next100/completed_rows/`.
If the pod stops, rerun the exact v2 command: matching completed rows are
reused and the evaluator refuses a changed protocol in the same directory.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.evaluate_v2 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --ids-file results/span_v1_fresh_dev/next_dev_100_frozen.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --adapter runs/span_infiller_v2/adapter \
  --output runs/span_v2_next100 \
  2>&1 | tee runs/span_v2_next100.log
.venv_blackwell/bin/python -m span_editor.compare_v2 \
  --v1 runs/span_v1_next100/predictions.jsonl \
  --v2 runs/span_v2_next100/predictions.jsonl \
  --output runs/span_v1_v2_paired
cat runs/span_v1_v2_paired/paired_summary.json
.venv_blackwell/bin/python -m span_editor.compare_v2 \
  --v1 runs/span_v1_next100/predictions.jsonl \
  --v2 runs/span_v2_next100/predictions.jsonl \
  --condition v2_forced --output runs/span_v1_v2_forced_paired
cat runs/span_v1_v2_forced_paired/paired_summary.json
```

The comparison refuses unequal IDs, source articles, or silver targets. Its
`audit_blind.csv` randomizes v1/v2 output order per ID; score factual support,
old-fact removal, other-fact preservation, and unsupported additions before
opening `audit_key_keep_separate.json`.

## 6. Send back the results

```bash
cd ~/revproof
tar -czf ~/revproof_span_v2_results.tgz \
  data/fruit/span_v2/manifest.json \
  runs/span_selector_v2/protocol.json runs/span_selector_v2/train_metrics.json \
  runs/span_locator_v2/protocol.json runs/span_locator_v2/train_metrics.json \
  runs/span_v2_seen_probe/summary.json \
  runs/span_infiller_v2/protocol.json runs/span_infiller_v2/train_metrics.json \
  runs/span_v1_next100/summary.json runs/span_v1_next100/predictions.jsonl \
  runs/span_v2_next100/summary.json runs/span_v2_next100/predictions.jsonl \
  runs/span_v1_v2_paired/paired_summary.json \
  runs/span_v1_v2_forced_paired/paired_summary.json \
  runs/span_v1_v2_paired/audit_blind.csv
ls -lh ~/revproof_span_v2_results.tgz
```

Send this archive for a paired error analysis. Keep trained weights on
DataHub; the result archive deliberately omits them.
