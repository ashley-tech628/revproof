# RevProof

RevProof studies evidence-grounded revision: given an old document and new
evidence, identify the outdated text and write a supported update while
preserving unrelated content. The current large-scale development data are
FRUIT-WIKI silver examples. The small source-audited Node.js/Flask set is a
separate technical-domain pilot. Neither silver exact match nor string
overlap establishes factual support.

## Current experiment: boundary refinement v3

The [v3 DataHub runbook](research/datahub_span_v3.md) trains one RoBERTa
classifier to score complete character-span boundaries around existing QA
proposals. It reuses the v2 evidence selector and FLAN-T5-XL infiller, then
compares v2 and v3 on a newly frozen 100-row development slice. The first
already-inspected candidate probe reached 56/100 exact silver spans, below
v2's 65/100 under the same eight-candidate limit. A pool-versus-ranking
diagnostic is in progress; no v3 final-edit accuracy result exists yet. The
later-period gold test remains untouched.

## Completed experiment: reference-planned span editor v2

The [v2 DataHub runbook](research/datahub_span_v2.md) gives the new training
and paired evaluation commands. It trains a sentence/evidence selector on
the publisher's heuristic reference markers, a RoBERTa character-span
locator on selected evidence items, and a 4-bit `google/flan-t5-xl` LoRA
infiller on short replacement phrases. Inference proposes multiple evidence
and span candidates, checks for unsupported new numbers, and copies all
unmodified text. A candidate recall probe on already inspected development
rows ran before infiller training, and the next 100 IDs were frozen before
the paired comparison.

The completed paired evaluation on those 100 frozen FRUIT silver short-span
rows found **9/100** full exact for v1, **20/100** for v2 forced editing,
and **17/100** for v2 with a fixed abstention gate. The [case analysis](results/span_v2_frozen_dev/ANALYSIS.md)
shows that span boundaries and replacement generation remain the main
bottlenecks. Those 100 rows are now inspected; future model selection needs
another untouched slice. Exact silver text does not establish factual support.

## Previous experiment: locate, infill, copy

The [v1 span-editor runbook](research/datahub_span_editor.md) gives the exact
DataHub commands for the completed baseline. That pipeline trains two components:

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

An earlier GPU result predates this architecture: a one-sentence
oracle-locus FLAN-T5-XL editor achieved **5/95** exact replacement on
held-out FRUIT silver development examples after 500 steps, versus **1/95**
for the unchanged model and **0/95** for copying. The record and manual
case analysis are in [research/fruit_local_gate500_analysis.md](research/fruit_local_gate500_analysis.md).
The completed v1 span editor achieved **31/100** exact with the gold edit
span and **15/100** end-to-end exact on a new, fixed FRUIT silver short-span
development slice. The locator selected the gold sentence on 80/100 and the
exact character span on 30/100. Read the [per-example analysis](results/span_v1_fresh_dev/ANALYSIS.md)
before interpreting these numbers: the short-span filter excludes most
publisher examples, and silver exact match is not factual support.

## Repository map

| Location | Purpose |
| --- | --- |
| [`span_editor/`](span_editor/) and [`fruit_span.py`](fruit_span.py) | V1–v3 locator, boundary ranker, infiller and copy-constrained evaluation |
| [`research/datahub_span_v3.md`](research/datahub_span_v3.md) | Current v3 DataHub training and paired evaluation commands |
| [`research/datahub_span_v2.md`](research/datahub_span_v2.md) | Completed v2 DataHub setup and commands |
| [`research/datahub_span_editor.md`](research/datahub_span_editor.md) | V1 setup, commands and protocol |
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
publisher-data conversion. The v2 runbook also shows how to download the
remaining official train shards.
