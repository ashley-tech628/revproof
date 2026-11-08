# DataHub: hard-negative boundary ranker follow-up

The first v3 probe used the same 100 already inspected development examples
and eight-candidate budget as v2. It found the silver gold span in 65/100
v2 candidates, 77/100 of the unranked v3 proposal pools, 56/100 v3 top
eight, and 69/100 in a four-v2/four-v3 hybrid. V3 additionally gained five
span hits and lost 14 relative to v2. The learned ranking, rather than the
proposal pool, is the main reason v3 missed its screen. These are candidate
recall diagnostics, not final edit accuracy or factual-support measurements.

The original boundary model was trained on perturbed gold spans in a gold
sentence with a reference evidence item. Inference instead compares spans
from several predicted sentence/evidence pairs. This follow-up mines actual
selector/QA proposals **only from training rows**, then trains a new ranker
against their mistakes. The original v3 weights remain in place. The next
frozen 100 development IDs remain unopened until the seen-row screen passes.

Upload `revproof_span_v3_hard.zip` into `~` through JupyterLab. Run these in
a DataHub GPU pod terminal, not the DSMLP login shell. The archive contains
code and this runbook, not model weights or publisher data.

## 1. Install and check

```bash
cd ~/revproof
unzip -o ~/revproof_span_v3_hard.zip -d ~/revproof
.venv_blackwell/bin/python -m unittest test_span_editor_v3.py -v
test -s runs/span_selector_v2/model/config.json
test -s runs/span_locator_v2/model/config.json
test -s runs/span_boundary_v3/model/config.json
test -s data/fruit/span_v2/span_train_v2.jsonl
test -s data/fruit/span_v2/span_dev_v2.jsonl
nvidia-smi
df -h . /tmp
mkdir -p runs
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
set -o pipefail
```

## 2. Mine hard negatives on training rows

The script saves each completed row under
`runs/span_boundary_hard_mine/completed_rows/`. Rerun the **identical**
command after interruption; completed rows are reused and the protocol
rejects incompatible settings. It writes `negatives.jsonl` only after all
rows finish. This can be slower than training because it applies the
selector and QA locator to every training example.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.mine_boundary_hard \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --max-train-rows 12000 \
  --output runs/span_boundary_hard_mine \
  2>&1 | tee runs/span_boundary_hard_mine.log
test -s runs/span_boundary_hard_mine/negatives.jsonl
cat runs/span_boundary_hard_mine/summary.json
```

## 3. Fine-tune the existing boundary model

This uses the existing v3 ranker as initialization. The new weights go to
`runs/span_boundary_hard/model`, leaving v3 untouched. Loss decreasing is
not evidence of improvement; the next probe is decisive.

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.train_boundary_hard \
  --train data/fruit/span_v2/span_train_v2.jsonl \
  --dev data/fruit/span_v2/span_dev_v2.jsonl \
  --mine runs/span_boundary_hard_mine/negatives.jsonl \
  --model runs/span_boundary_v3/model \
  --max-train-rows 12000 --epochs 2 \
  --output runs/span_boundary_hard \
  2>&1 | tee runs/span_boundary_hard.log
test -s runs/span_boundary_hard/model/config.json
cat runs/span_boundary_hard/train_metrics.json
```

## 4. Screen on the already inspected 100 rows

```bash
cd ~/revproof
set -o pipefail
.venv_blackwell/bin/python -u -m span_editor.probe_v3 \
  --data data/fruit/span_v2/span_dev_v2.jsonl \
  --seen-summary results/span_v1_fresh_dev/summary.json \
  --selector runs/span_selector_v2/model \
  --locator runs/span_locator_v2/model \
  --boundary runs/span_boundary_hard/model \
  --output runs/span_boundary_hard_seen_probe \
  2>&1 | tee runs/span_boundary_hard_seen_probe.log
.venv_blackwell/bin/python -c "import json; s=json.load(open('runs/span_boundary_hard_seen_probe/summary.json')); print({k:s[k] for k in ('gold_span_in_v2_candidates','gold_span_in_pool','gold_span_in_candidates','gold_span_in_hybrid','v3_only_gold_span','v2_only_gold_span')})"
```

The old boundary model scored 56/100; v2 reached 65/100; the old
four-plus-four hybrid reached 69/100. Proceed to a new frozen evaluation
only if the new ranker beats **69/100** under the same eight-candidate
budget. If it does not, send the probe summary and stop this branch.
Even 70+/100 span recall does not ensure a large final exact-match gain:
replacement generation is a separate bottleneck.

If the screen passes, run the v2 baseline and v3 evaluation in
[`datahub_span_v3.md`](datahub_span_v3.md), replacing the v3 `--boundary`
path with `runs/span_boundary_hard/model` and using a new v3 output directory.
Pair the outputs with `span_editor.compare_v3`. Keep the frozen ID file and
all other settings identical between systems.

## Next model change if span recall rises but exact edits do not

The prior frozen 100-row analysis found 27 errors where the exact silver
span was already proposed but the FLAN infiller did not generate the silver
replacement; numeric edits were 0/15 exact. The next isolated experiment
should add an evidence-extraction replacement candidate and compare it with
the current FLAN output, with a support and grammar audit. Enlarging the
boundary ranker cannot fix missing or unsupported replacement facts.
