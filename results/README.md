# Archived results

These are checked-in summaries, selected predictions, and analyses. Large
model weights, training checkpoints, raw external datasets, and transient
`runs/` output are excluded from Git.

| Experiment | Location | Interpretation |
| --- | --- | --- |
| FRUIT local editor, 500 steps | [`fruit_local_gate500/`](fruit_local_gate500/) | 5/95 exact on silver dev with gold edit location; [analysis](../research/fruit_local_gate500_analysis.md) |
| FRUIT short-span editor v1 | [`span_v1_fresh_dev/`](span_v1_fresh_dev/) | 31/100 oracle-span and 15/100 end-to-end exact on a fresh silver short-span development slice; [analysis](span_v1_fresh_dev/ANALYSIS.md) |
| FRUIT full-article gate, 500 steps | [`fruit_xl_gate500/`](fruit_xl_gate500/) | Earlier generation objective, not paired with the local subset |
| Earlier sparse/oracle pilots | [`sparse_15c6e2f/`](sparse_15c6e2f/), [`oracle_flan_8007a70/`](oracle_flan_8007a70/) | Diagnostic pilots |
| Original DataHub pilot | [`datahub_223b776/`](datahub_223b776/) | Small source-checked technical-domain pilot |
| Local template/retrieval checks | [`pilot_local/`](pilot_local/) | Baselines and pipeline checks |

New span-editor runs should remain under ignored `runs/` until their protocol,
denominator, and predictions have been reviewed. An exact match with FRUIT
silver text is not a factual-support judgment.
