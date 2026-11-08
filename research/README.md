# Research index

## Current direction

- [DataHub span-editor runbook](datahub_span_editor.md): strict short-span
  FRUIT subset, trained locator, FLAN-T5-XL infiller, copy-constrained
  reconstruction, and a fresh development slice.
- [Span-editor v1 result analysis](../results/span_v1_fresh_dev/ANALYSIS.md):
  31/100 gold-span and 15/100 end-to-end exact on a fixed silver development
  slice, with per-example error analysis and limits.
- [Span-editor v2 literature decision](span_editor_v2_literature_decision.md):
  evidence-reference supervision, larger FRUIT training coverage, candidate
  boundaries, and paired evaluation without gold-location leakage.
- [DataHub span-editor v2 runbook](datahub_span_v2.md): complete download,
  conversion, three-component training, frozen paired evaluation and audit.
- [Span-editor v2 paired result analysis](../results/span_v2_frozen_dev/ANALYSIS.md):
  v1 9/100, v2 forced 20/100 and v2 gated 17/100 on the same newly inspected
  FRUIT silver development IDs; breakdown of remaining failures.
- [Local-editor case analysis](fruit_local_gate500_analysis.md): the completed
  500-step oracle-locus experiment and provisional human review that motivate
  the new target.
- [One-window experiment decision](local_edit_next_step.md): pretraining
  rationale and structural-filter feasibility. Its stages predate the new
  implementation; use the span-editor runbook for current commands.

## Completed pilot records

| Topic | Record |
| --- | --- |
| FRUIT one-sentence positive control | [fruit_local_control120_analysis.md](fruit_local_control120_analysis.md) |
| FRUIT full-article generator | [fruit_xl_gate500_analysis.md](fruit_xl_gate500_analysis.md) |
| FRUIT data and initial gate | [fruit_editor_runbook.md](fruit_editor_runbook.md), [datahub_fruit_gate.md](datahub_fruit_gate.md), [datahub_fruit_local.md](datahub_fruit_local.md) |
| Code change silver pretraining | [codoc_300_analysis.md](codoc_300_analysis.md), [codoc_training_runbook.md](codoc_training_runbook.md) |
| Earlier sparse/oracle experiments | [architecture_decision.md](architecture_decision.md), [evidence_grounded_revision_reset.md](evidence_grounded_revision_reset.md) |
| Dataset survey | [dataset_sources.md](dataset_sources.md) |

The [previous root README](README_legacy_2026-11-07.md) is archived for
historical commands and context. Its status section is superseded by the
current root README and the experiment records above.
