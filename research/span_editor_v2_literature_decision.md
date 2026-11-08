# Span editor v2 research decision (2026-11-07)

This is an experiment design, not a measured v2 result. The measured v1 record is
[`../results/span_v1_fresh_dev/ANALYSIS.md`](../results/span_v1_fresh_dev/ANALYSIS.md).

## Decision

Keep the copy-constrained FLAN-T5-XL infiller as the first generator baseline.
Build an evidence-planned editor around it: (1) select a source sentence and
specific evidence item, (2) predict a *small set* of character-span candidates,
(3) generate a replacement for each candidate from the selected evidence,
(4) rerank or abstain using explicit evidence-role and preservation checks,
(5) copy all other source characters. Train on substantially more FRUIT train
shards. Compare each component in a paired ablation before trying a bigger
generator. This design is inspired by prior work but is not a reproduction of
any one published model.

## Why this experiment, rather than just a larger language model?

The v1 100-example development slice gives 0 exact for unchanged FLAN with a
gold span, 31 exact for trained FLAN with a gold span, and 15 exact end to end.
The locator found the right sentence 80 times but the exact character span 30
times. Among 31 gold-span successes, 17 were lost in the pipeline: six wrong
sentences, three abstentions with correct sentences, and eight accepted wrong
spans. A different span yielded one pipeline exact when the gold-span output
was nonexact, so 31 is *not* a formal ceiling for a new locator. Still, merely
fixing locations cannot be expected to solve every remaining generator error.
Among 39 acted-on rows with the correct sentence but wrong span, 33 spans
overlap the gold span and 16 contain it; this motivates candidate boundary
testing, but does not prove that the right boundary appears in the model's
top-k predictions.

Training used only 1,014 token-eligible short-span rows from one publisher
training shard and 500 steps (~1.97 epochs). The FRUIT publisher reports 114K
silver articles in its 2019–2020 corpus, of which it uses 10% for validation;
the processed training partition has nine shards. This is a data
coverage gap, although the strict short-span conversion retains only a
fraction of those articles. The current locator puts the entire concatenated
evidence into a question and truncates that side at 384 tokens. The 100
inspected development rows have median evidence length 772.5 characters and
23/100 exceed 1,500 characters. The exact rate of evidence truncation has not
yet been measured, but important cells can be lost. We need an instrumented
token-coverage report before changing the input policy.

Crucially, the official processed FRUIT target includes parenthesized
*evidence-reference indices* before edited sentences. The publisher says
these training references are heuristic. `fruit_local.extract` currently
removes the references while making one-sentence rows. Retaining them offers
direct, if noisy, supervision for evidence selection. The v2 converter must
verify references align with the one literal replacement and never pass them
to inference. Evidence items are delimited by `(0)`, `(1)`, etc. in the input.

In the inspected sample only 33/100 gold replacement spans occur literally
within the concatenated evidence, case-insensitively. This does not mean only
33 are supported: aliases, inflection, and larger paraphrases exist. It does
mean a hard evidence-copy-only decoder cannot reproduce all silver targets.
Use exact copying as a candidate for names, dates, numbers and simple entity
substitution; retain generation for grammatical or multi-fact changes.

## What the literature actually supports

| Primary source | Relevant finding | Consequence for RevProof |
| --- | --- | --- |
| [FRUIT / EDIT5](https://aclanthology.org/2022.naacl-main.269/) and [official data format](https://github.com/google-research/language/tree/master/language/fruit) | EDIT5 compresses copied sentences and prefixes evidence-reference tokens to edits. Its ablation credits both choices; reference tokens have a larger effect. Oracle control codes improve its *UpdateROUGE* substantially. EDIT5-Base, Large, and 3B have relatively small differences in UpdateROUGE-L (39.7, 39.6, 41.1) on its gold task. | Preserve and predict evidence references; test source/evidence/edit planning before spending on model size. These publisher scores are **not exact match** and are not comparable with ours. Oracle control codes are not deployable inference performance. |
| [EdiT5](https://aclanthology.org/2022.findings-emnlp.156/), [FELIX](https://aclanthology.org/2020.findings-emnlp.111/), [LaserTagger](https://aclanthology.org/D19-1510/) | Tagging or selecting retained text and generating insertions can help copy-heavy editing, particularly with limited data. | Keep deterministic copying; test token-boundary tagging or candidate spans rather than article-level regeneration. EdiT5 is a distinct paper/system from FRUIT's EDIT5. |
| [Fact-based Text Editing](https://aclanthology.org/2020.acl-main.17/) | Explicit fact memory and copying outperform a plain encoder-decoder on its table-to-text editing tasks. | Preserve table row and column roles instead of flattening a whole table into an undifferentiated string. Its reported gains do not transfer automatically to FRUIT. |
| [Evidence-based Factual Error Correction](https://aclanthology.org/2021.acl-long.256/) | Evidence-conditioned T5 correction can make more accurate local revisions than the compared pointer-copy system in that paper. | Do not force every new phrase to be copied; allow evidence-conditioned infilling when it is needed. |
| [RARR](https://aclanthology.org/2023.acl-long.910/) | Evidence attribution and preservation need explicit checking; irrelevant evidence can cause unsupported edits. | Add a support check and a no-edit/abstain decision; do not use word overlap as a truth score. |
| [ToTTo input-fidelity analysis](https://aclanthology.org/2024.naacl-long.408/) | Table-to-text errors can arise from malformed or ambiguous input representation. | Keep cell headers, row identity, entity role, and provenance through evidence selection and generation. |
| [CoEdIT](https://aclanthology.org/2023.findings-emnlp.350/), [PEER](https://openreview.net/forum?id=KbYevcLjnc), [EditEval](https://aclanthology.org/2024.conll-1.7/) | Instruction-based editors are capable of broad revisions, but factual updating and evaluation remain difficult. | Use them as comparison/auxiliary editing experience, not a substitute for evidence-aligned FRUIT training. |

FRUIT reports that only about one third of updates in its silver corpus are
substantiated by evidence under its distant-supervision rule. Its 914-item
gold set was human corrected so updates are supported. Our exact silver
matches therefore cannot be called factual correctness. Case `13019633`
exactly matches a silver publisher claim that the provided acquisition
evidence does not establish; case `43511708` gives the supported club with a
different Unicode spelling and loses exact credit. These are separate
evaluation failures.

## Build order and falsifiable tests

1. **Audit input and labels before GPU use.** Parse original FRUIT target
   reference tokens, split evidence items, count broken/ambiguous alignments,
   measure tokenizer truncation by item and by known train reference. Keep
   `train` shards 0–8 for training and shard 9 for development. Do not inspect
   the later-period gold test while designing the model.
2. **Scale the existing infiller with controlled data.** Convert additional
   training shards, deduplicate by publisher ID and article identity where
   available, retain source article groups,
   and compare 1-shard vs 9-shard on the same fixed development IDs and same
   FLAN architecture. Include all current short-span examples that pass the
   converter, then separately test higher-confidence evidence-supported
   subsets. Report the number and type of excluded rows. Do not merge
   ungrounded CoEdIT or unaudited CoDocBench rows into this target.
3. **Predict evidence before location.** Train a sentence/evidence-item
   cross-encoder or dual encoder with the heuristic reference indices as
   partial labels. Include same-article distractors and a no-support option.
   Compare reference recall@1 and @3, source-sentence accuracy, and evidence
   truncation against v1. Table candidates must retain row/column context.
4. **Replace single best QA span with candidate boundaries.** Use a token
   start/end or BIO tagging head on the selected sentence; keep the top-k
   valid boundaries and a no-edit candidate. At training, score candidate
   spans with gold-derived labels; at inference, never pass gold offsets.
   Measure gold-span recall@k and incorrect-span acceptance. This is the
   clearest test of the observed 30/100 boundary bottleneck.
5. **Generate and check.** Condition the same infiller on one or a few
   selected evidence items and a candidate span. For simple entity/number
   changes, propose a verbatim evidence-value candidate too. Rerank complete
   edited sentences using generator likelihood, evidence support, and
   preservation. Reject candidates that change text outside the proposed
   span, mix table columns, or leave a contradictory old qualifier. Calibrate
   abstention on inspected development examples; report coverage and error
   tradeoff instead of maximizing exact by indiscriminate editing.
6. **Evaluate paired.** Run v1 and each v2 ablation on the *same* frozen next
   100 development IDs in `../results/span_v1_fresh_dev/next_dev_100_frozen.json`.
   Do not compare a v2 score there with v1's 15/100 from a different 100.
   Export predictions, sentence accuracy, span recall@k, selected-evidence
   recall, end-to-end exact and edit-content metrics. Blindly adjudicate a
   fixed sample for target-fact support, old-fact removal, protected-fact
   preservation, unsupported additions, and silver-target quality. Later
   evaluate once on untouched later-period gold data with a declared scope.

The first success criterion is a *paired* increase in supported end-to-end
edits without increasing harmful edits, with exact match as a secondary
diagnostic. A large exact gain is possible but cannot be promised. If
evidence selection improves while exact remains flat, inspect unsupported
silver targets and output-form errors. If gold-span infilling improves while
pipeline exact remains flat, focus on locator and reranking. If both stay
flat despite more data, revisit the one-span task definition before changing
to a larger backbone.
