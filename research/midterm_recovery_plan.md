# ECE285 midterm recovery plan (2026-11-06)

## What the submitted report must establish

The instructor asks for roughly five pages excluding references: revised
proposal, detailed mathematical/model/algorithmic method, and discussion of
novelty, advantages, and limitations. Experiments and conclusions are the
focus of the later final report. The midterm can include preliminary failures,
but should never describe them as successful results.

The original proposal's central question remains: can verified, versioned
document interventions improve evidence selection for questions whose answers
may change across releases? CoDocBench code/docstring updates are an auxiliary
silver-data experiment, not evidence that this central question has been
answered. Its code co-change does not independently verify factual support.

## Revised contribution

Use an evidence-conditioned, copy-constrained **multi-span editor** to propose
revisions, followed by explicit claim verification and abstention. Use only
verified edits in an intervention-consistent reranker. Factorizing editing into
localization and infilling follows FELIX/EdiT5 and is not itself novel. The
research hypothesis is that coupling this factorization to release-specific
evidence, preservation constraints, and selective retrieval yields more
reliable version-aware answers than unconstrained generation or retrieval
baselines. This is a hypothesis, not a measured gain.

Let `x` be the old passage, `e` an independently sourced target-release fact
card, `q` the question, `v` the requested version, `z` one or more disjoint
source-token spans, and `r` the generated replacements. Predict

`p(z,r | x,e,q,v) = p_phi(z | x,e,q,v) p_theta(r | x,e,q,v,z)`.

The deterministic renderer `y = Apply(x,z,r)` copies all tokens outside `z`
verbatim. The locator is a token-level BIO or start/end predictor; it does not
generate free-form numeric character offsets. A pretrained encoder can model
`p_phi`; a pretrained encoder-decoder model can infill `p_theta`. The evidence
card specifies changed value, unchanged protected values, release, branch,
and source provenance. Train the locator with diff-derived span supervision
and the infiller with replacement-only negative log likelihood. Add a
verifier/abstention gate: apply a candidate only when release scope matches,
the new fact is supported, and protected facts remain intact. Gold spans at
evaluation are an oracle upper bound, never an end-to-end system score.

For the downstream task, retain the proposal's supervised reranker and
answer-preserving / answer-changing consistency losses. The intervention
mapping is used only for audited one-to-one changes. The sufficiency head
calibrates abstention when version provenance is missing or contradictory.
The fixed answer model isolates the effect of evidence selection.

## Evidence and evaluation hierarchy

1. **Real target domain:** source-linked software release documentation with
   independently audited old/new facts and version scope. This is the only
   set that can support the core RevProof factual-validity claim. Freeze a new
   repository/document-family holdout before architecture selection.
2. **FRUIT-WIKI:** an external evidence-grounded editing benchmark with
   large distantly supervised training data and a separately verified gold
   evaluation set. Useful for editor pretraining and an external benchmark,
   but Wikipedia performance is not software-documentation performance.
3. **CoDocBench:** 2,300 official train examples currently converted to
   code-diff-conditioned docstring edits. Use as a diagnostic or optional
   domain-adaptation source, clearly labeled unaudited silver data. Its
   internal project-held-out dev set has 254 eligible examples; it is not the
   final target-domain test.

Primary editor measures: factual support of every changed claim, preservation
of protected claims, unsupported addition rate, edit-location accuracy,
abstention coverage/risk, and per-repository uncertainty. Exact match and
edit distance are secondary diagnostics. Retrieval measures: applicable
evidence Recall@k, stale-answer rate, version-specific answer accuracy, and
invariance/equivariance violations. Compare against deterministic edits,
prompt-only and full-generation baselines, and retrieval with exact version
filtering. Match candidate sets and available provenance.

## Honest preliminary result

Qwen2.5-1.5B-Instruct QLoRA trained on CoDocBench silver data scored 7/254
exact for complete-docstring generation and 0/254 for generated character-
offset plans after both 300 and 900 steps. At 900 steps, 248/254 plans could
be applied; applicability did not translate into exact revisions. The 900-step
numbers currently come from screenshots. Per-example files are needed for
support and location diagnostics. This experiment motivates learned source
localization and better evidence; it does not establish that the revised method
works. Three-row overlapping positive controls are pipeline checks, not
generalization evidence.

## Five-page structure

1. **Problem and revision to proposal (0.8 page):** version-conditional
   evidence, why a document edit can change or preserve an answer; narrow the
   generation component to evidence-grounded local edits.
2. **Related work and gap (0.7 page):** version-aware retrieval, FRUIT,
   FELIX/EdiT5; identify the combined evidence/verification/retrieval
   hypothesis without claiming the component methods as novel.
3. **Data and formal setup (0.8 page):** provenance, release scope, changed
   and protected facts, split protocol, silver versus audited evidence.
4. **Model and algorithm (1.6 pages):** locator, infiller, copy renderer,
   verifier/abstention, reranker loss; train and inference pseudocode.
5. **Preliminary findings and discussion (1.1 pages):** report the failed
   CoDocBench baseline in a compact table; explain format-versus-correctness,
   advantages, limitations, planned ablations and real-domain evaluation.

This is a content plan, not a finished report. Do not promise a grade or a
publication. The strongest defensible midterm is a precise method and honest
protocol with correctly labeled preliminary evidence.

## Immediate work before submission

- Inspect the two 900-step prediction files, calculate location and source-
  preservation metrics, and manually audit a fixed sample. Avoid another
  identical QLoRA run before this diagnosis.
- Implement and unit-test the **locator data construction** (source-token
  offsets and diff-derived labels) and deterministic reconstruction. This
  preprocessing is now implemented in `token_edit_data.py`: all 2,300 local
  silver pairs round-trip exactly, and the full 28-test suite passes. It is
  **not** a trained locator or a model-quality result. Among these rows, the
  median source has 90 lexical/whitespace tokens, the median alignment deletes
  two tokens and inserts text at one gap. If a meaningful locator pilot is
  completed, report it; otherwise state clearly which components are proposed
  and which are implemented.
- Draft the five-page report with the original proposal as the starting point.
  Keep experiment outcomes and future hypotheses grammatically distinct.

## Primary references

- [FRUIT](https://aclanthology.org/2022.naacl-main.269/)
- [FELIX](https://aclanthology.org/2020.findings-emnlp.111/)
- [EdiT5](https://aclanthology.org/2022.findings-emnlp.156/)
- [CoDocBench](https://arxiv.org/abs/2502.00519)
