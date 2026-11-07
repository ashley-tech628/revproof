# DataHub FRUIT 3B gate

This is an early development diagnostic for `google/flan-t5-xl` on 30 fixed
FRUIT-WIKI silver development examples. It compares copying the source,
the unchanged base model, and the 500-step LoRA editor on identical rows.
Exact reconstructed text, word-sequence overlap, added-token overlap, and
reconstruction syntax checks are diagnostics; none establishes factual support.
The official gold test remains untouched.

## Restore DataHub storage

On 2026-11-07, the ECE285 course workspace was at 30.7/30 GB (102%). The
notebook pod was repeatedly restarting. The earlier `google/flan-t5-xl`
download failed with `Disk quota exceeded`. See UCSD's
[DataHub FAQ](https://support.ucsd.edu/its?id=kb_article&sysparm_article=KB0030470)
for SSH access when the notebook will not start. From a computer terminal:

```bash
ssh xil277@dsmlp-login.ucsd.edu
workspace -c ECE285_FA26_002
pwd
du -h --max-depth=1 . | sort -h
du -sh .cache/huggingface/hub/models--google--flan-t5-xl .cache/huggingface/xet revproof/.venv_blackwell revproof/.venv_datahub revproof/.venv 2>/dev/null
```

Verify that `pwd` is the ECE285 course home before deleting anything. Retain
`revproof/.venv_blackwell`, data, and runs. `revproof/.venv_datahub` was the old
environment whose PyTorch 2.2.1 could not execute on the Blackwell GPU. If it
is present and no longer needed, remove only that exact environment. The
interrupted FLAN-T5-XL cache can also be removed after inspecting its size;
it will be downloaded again. Do not clear unrelated model caches or result
folders. Keep about 18 GB course-space headroom before a fresh 11.4 GB model
download. The separate 10 GB personal pool cannot hold the full source
checkpoint. After reclaiming space, use Services > manual-resetter and reopen
DataHub.

## Install updated experiment code

Upload `revproof_fruit_datahub_gate.zip` into `~/revproof`, then:

```bash
cd ~/revproof
unzip -o revproof_fruit_datahub_gate.zip
.venv_blackwell/bin/python -m unittest test_fruit_data.py test_fruit_gate.py -v
test -s data/fruit/train_00000.jsonl
test -s data/fruit/dev_00009.jsonl
```

If the two JSONL files are absent, follow the download and conversion commands
in `research/fruit_editor_runbook.md`. Set Hugging Face cache to course space,
not `~/private`, which has a 10 GB quota in this account:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

## Train 500 steps and compare baselines

The 500 optimizer steps cover roughly 2,000 sample presentations because the
gradient accumulation factor is 4. One pass over 2,000 selected train rows is
an early signal, not a converged result. The development rows are selected by
ID hash; 30 are selected before token-length filtering.

```bash
set -o pipefail
mkdir -p runs
.venv_blackwell/bin/python -u train_fruit_editor.py --train data/fruit/train_00000.jsonl --dev data/fruit/dev_00009.jsonl --model google/flan-t5-xl --max-train-rows 2000 --max-dev-rows 30 --max-steps 500 --stage train --output runs/fruit_xl_gate500 2>&1 | tee runs/fruit_xl_gate500_train.log
```

The adapter and Trainer checkpoint are
saved under `runs/fruit_xl_gate500`. Then run paired evaluation:

```bash
.venv_blackwell/bin/python -u evaluate_fruit_gate.py --dev data/fruit/dev_00009.jsonl --model google/flan-t5-xl --adapter runs/fruit_xl_gate500/adapter --max-dev-rows 30 --output runs/fruit_xl_gate500_compare 2>&1 | tee runs/fruit_xl_gate500_eval.log
```

Inspect `runs/fruit_xl_gate500_compare/summary.json` and
`dev_predictions.jsonl`. Send both files for analysis. A gain in added-token
overlap versus base and copy, together with valid reconstructions and a manual
sample audit, is the minimum reason to scale the run. No one metric alone is a
success claim.
