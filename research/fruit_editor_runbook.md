# FRUIT-WIKI large generative editor pilot

## Research decision

The first large-model experiment is a **3B encoder-decoder editor** based on
`google/flan-t5-xl`, trained to produce the FRUIT publisher's sentence-copy and
evidence-reference format. This is an implementable analogue of FRUIT's
**EDIT5** output objective, not a reproduction of their exact T5X architecture
or hyperparameters. FRUIT EDIT5 and the separate EdiT5 tagging model are
different papers despite similar names.

The [FRUIT paper](https://aclanthology.org/2022.naacl-main.269/) reports
UpdateROUGE-L 41.1 for its EDIT5-3B and 36.8 for evidence-input T5-Large on
its gold set, but only 50/100 sampled EDIT5-3B outputs were fully grounded
updates by their manual audit. These are **published comparator results**, not
RevProof results. The public [dataset release](https://github.com/google-research/language/tree/master/language/fruit)
provides official training shards 0-8, development shard 9, and a later-period
gold test. We will tune on shard 9 and leave gold test unopened until final.

Why this has a stronger chance than the CoDocBench pilot: the input contains
the old text and explicit new evidence; output uses source-sentence copy
markers and edited sentences instead of predicting numeric character offsets;
the official training set is much larger. Transfer from Wikipedia to technical
release notes is still unproven. A second candidate is Qwen3-8B in 4-bit,
evaluated on exactly the same development examples as a prompt-only challenger
before any fine-tuning. [Qwen's model card](https://huggingface.co/Qwen/Qwen3-8B)
documents its 8.2B parameters and non-thinking chat mode; 4-bit fit on the
24GB MIG still needs a memory preflight. We will not attribute any gain solely
to parameter count.

## Pilot protocol on UCSD DataHub

Upload `revproof_fruit_pilot.zip` to `~`, then run in the existing 1-GPU
environment. The zip contains code and this runbook, **not** the external
dataset or adapters.

```bash
cd ~/revproof
unzip -o ~/revproof_fruit_pilot.zip
.venv_blackwell/bin/python -m pip install tfrecord
.venv_blackwell/bin/python -m unittest discover -p 'test_*.py' -q
mkdir -p data/fruit
curl -L --fail --silent --show-error 'https://storage.googleapis.com/gresearch/FRUIT/dataset/train/article_pairs.update.diff.all.text.reference.tfrecords-00000-of-00010' -o data/fruit/article_pairs.update.diff.all.text.reference.tfrecords-00000-of-00010
curl -L --fail --silent --show-error 'https://storage.googleapis.com/gresearch/FRUIT/dataset/train/article_pairs.update.diff.all.text.reference.tfrecords-00009-of-00010' -o data/fruit/article_pairs.update.diff.all.text.reference.tfrecords-00009-of-00010
.venv_blackwell/bin/python fruit_data.py --input data/fruit/article_pairs.update.diff.all.text.reference.tfrecords-00000-of-00010 --output data/fruit/train_00000.jsonl --split train
.venv_blackwell/bin/python fruit_data.py --input data/fruit/article_pairs.update.diff.all.text.reference.tfrecords-00009-of-00010 --output data/fruit/dev_00009.jsonl --split dev
```

Local inspection of these same official files found 11,345 training-shard
records and 11,561 development-shard records; the conservative reconstruction
parser retained 11,342 and 11,559 respectively, rejecting five malformed or
ambiguous numbered-source cases across both shards. SHA-256 hashes of the
downloaded files should be recorded in the run protocol. This pilot uses only
one of the publisher's nine training shards and is therefore a data-budget
screen, not a full published-benchmark reproduction.

First run an **overlapping 8-row positive control**. Its score is not a
generalization result. This downloads the 3B model; the first run needs time
and disk. If the model fails to fit or train on the 24GB MIG, stop and inspect
memory/version errors before changing methods.

```bash
.venv_blackwell/bin/python train_fruit_editor.py --train data/fruit/train_00000.jsonl --dev data/fruit/dev_00009.jsonl --positive-control --max-steps 120 --output runs/fruit_t5xl_control
```

After inspecting its `dev_summary.json` and predictions, run the frozen pilot.
The hash-based development subsample is deterministic. Checkpoints save every
500 steps. The development split is used for model selection; do not call its
score an untouched test result.

```bash
.venv_blackwell/bin/python train_fruit_editor.py --train data/fruit/train_00000.jsonl --dev data/fruit/dev_00009.jsonl --max-train-rows 10000 --max-dev-rows 200 --max-steps 2500 --output runs/fruit_t5xl_2500
```

To resume after interruption, pass `--resume runs/fruit_t5xl_2500/checkpoints/checkpoint-N`
with the same other arguments and existing output directory. Do not change
`--max-steps` while resuming because the saved scheduler was planned for that
budget.

The script reports strict reconstructed-text exact match, valid copy markers,
and copy-source exact match. These are diagnostics. Before a research claim,
run the publisher's UpdateROUGE/entity metrics and a blinded factual-support
audit on fixed examples, then evaluate on the untouched gold test and a
separate technical-document holdout. If training barely beats copy-source,
inspect edits and evidence use before scaling model size.

Upload only `protocol.json`, `train_metrics.json`, `dev_summary.json`, and
`dev_predictions.jsonl` for analysis; adapters are not needed in the results
archive.
