# RevProof

RevProof studies evidence-grounded revision: given an old document and new
evidence, identify the outdated text and write a supported update while
preserving unrelated content. The current large-scale development data are
FRUIT-WIKI silver examples. The small source-audited Node.js/Flask set is a
separate technical-domain pilot. Neither silver exact match nor string
overlap establishes factual support.

## Current experiment: locate, infill, copy

The [span-editor runbook](research/datahub_span_editor.md) gives the exact
DataHub commands. The pipeline trains two components:

1. A RoBERTa QA locator chooses a sentence and character span from the old
   source using the evidence. Its supervision comes from gold-derived spans.
2. A 4-bit `google/flan-t5-xl` LoRA infiller produces only the replacement
   phrase. The program copies every character outside that span.

`fruit_span.py` creates strict short-span examples from `fruit_local.py`
outputs. `span_editor/train_locator.py`, `span_editor/train_infiller.py`, and
`span_editor/evaluate.py` implement training and paired evaluation. The
evaluation separates oracle-span upper bounds from the end-to-end pipeline.
The first 100 hash-ranked local-development examples were selected during an
earlier pilot. The evaluator explicitly excludes those IDs before choosing
new span-development rows. The later-period FRUIT gold test remains unopened.

The latest completed GPU result predates this architecture: a one-sentence
oracle-locus FLAN-T5-XL editor achieved **5/95** exact replacement on
held-out FRUIT silver development examples after 500 steps, versus **1/95**
for the unchanged model and **0/95** for copying. The record and manual
case analysis are in [research/fruit_local_gate500_analysis.md](research/fruit_local_gate500_analysis.md).
The new span-editor code has passed local contract checks and data conversion;
GPU training and evaluation are pending.

## Repository map

| Location | Purpose |
| --- | --- |
| [`span_editor/`](span_editor/) and [`fruit_span.py`](fruit_span.py) | Current locator, infiller and copy-constrained evaluation |
| [`research/datahub_span_editor.md`](research/datahub_span_editor.md) | DataHub setup, commands, protocol and interpretation |
| [`research/README.md`](research/README.md) | Research decisions and experiment history |
| [`results/README.md`](results/README.md) | Archived output index; training artifacts are excluded |
| [`data/`](data/) | Small checked-in datasets, audit notes and protocols |
| [`notebooks/`](notebooks/) | Historical Colab notebooks |
| Root legacy `train_*.py`, `evaluate_*.py`, `run_*.py` | Reproducible earlier pilots; see the research index |

The older full project guide is retained as
[`research/README_legacy_2026-11-07.md`](research/README_legacy_2026-11-07.md).
Historical root scripts remain in place so existing DataHub commands and
Python imports continue to work.

## Local checks

```bash
python -m unittest discover -p 'test_*.py' -v
python fruit_span.py --train data/fruit/local_gate/local_train_00000.jsonl \
  --dev data/fruit/local_gate/local_dev_00009.jsonl \
  --output data/fruit/span_v1
```

The FRUIT files are large external inputs and are not checked into Git. See
[`research/datahub_fruit_local.md`](research/datahub_fruit_local.md) for the
publisher-data conversion. The current runbook starts after that conversion.
