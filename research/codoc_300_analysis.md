# CoDocBench 300-step pilot: diagnosis and next experiment

Analyzed the user-provided `revproof_codoc_300_results.tgz` (SHA-256
`a477ae9f9015fcc01df90713453a31056aa5530785213d400cb2063af0c7f7d9`).
The two runs used identical eligible rows from the official CoDocBench **train**
split, project-disjoint internal train/dev sets (1,773/254 rows from 87/10
projects), the same Qwen2.5-1.5B-Instruct base and 300 optimizer steps. The
CoDocBench official test split was not used. The code diff is **silver**
evidence: a co-change does not prove that the diff entails a docstring edit.

| Objective | Exact new docstring | Parseable/applicable edit plan | Training loss |
| --- | ---: | ---: | ---: |
| Generate full docstring | 7/254 (2.8%) | N/A | 0.5313 |
| Generate character-offset edit plan | 0/254 (0%) | 242/254 (95.3%) | 1.1974 |

The plan failures were eight invalid/overlapping offsets and four empty edits.
Among the 242 applicable plans, only nine placed **any** edit start at a gold
start; zero matched any complete gold `(start, end)` span. This is a location
failure, even when JSON parsing succeeds. For example, `StoredData.__init__`
should remove `pickle ` at character offsets 10:17, but the model inserted
`Home Assistant` at offset 1 and added a generic parameter block. For
`YetiBinarySensor.is_on`, the target inserts ` True` at offset 6; the model
inserted whitespace inside existing words. The full-text model copied the old
docstring verbatim for 53/254 rows despite every target differing from its
source. Its median output was 145.5 characters versus a 69-character median
target. Exact match is strict and not a semantic-quality measure; we need a
small blinded human audit for support, preservation, and usefulness.

The 300 steps consumed roughly 600 example presentations, or 0.34 of one
epoch, and the default linear schedule decayed the learning rate to zero at
step 300. Thus this pilot does **not** establish that either objective would
fail after sufficient training. The 3-row positive controls already reached
3/3 training-example exactness for both objectives; they only establish that
the pipeline can learn and reload a small set.

The dev set is also uneven: 163/254 rows are from project `core`. Report both
micro averages and per-project macro averages in later experiments. One seed
and a single internal split are insufficient for a publication-level claim.

## Decision

Keep the full-text model as the trained baseline. Do not cite 95.3% plan
validity as revision quality. The next controlled experiment should train the
two existing objectives for at least one full epoch with matched steps and
evaluate on the same project split, while retaining this 300-step artifact as
an early-training checkpoint. The current code saves only final adapters, so
rerunning with a new output directory is required. Also add a copy-source
baseline, no-adapter prompt baseline, character edit distance and per-project
metrics; exact match alone hides failure modes.

For the architectural revision, replace free-form **numeric character offset
generation** with a learned localization stage over source spans/lines,
followed by a conditional infiller and deterministic copy of unchanged text.
This factorization itself is established prior art (FELIX and EdiT5); the
research contribution must be the evidence-conditioned localization,
preservation/abstention constraints, and an evaluation that tests them.
An oracle-location infiller is only a component positive control and must not
be counted as an end-to-end result. Use held-out project data to compare this
factorization with the full-text baseline at the same training budget. Audit
whether code diffs support the requested edits before claiming factual
grounding. The real versioned-document task still requires independently
verified evidence and a non-CoDocBench evaluation set.

Prior work: [FELIX](https://aclanthology.org/2020.findings-emnlp.111/),
[EdiT5](https://arxiv.org/abs/2205.12209), and the
[CoDocBench dataset paper](https://arxiv.org/abs/2502.00519).
