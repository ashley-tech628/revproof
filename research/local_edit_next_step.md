# RevProof next experiment: one-window evidence-conditioned infilling

## Why the current gate is insufficient

The 500-step FLAN-T5-XL pilot on FRUIT-WIKI learned source-copy syntax but
underperformed copying the old article on 29 silver development rows. See
`fruit_xl_gate500_analysis.md`. The next experiment changes the prediction
target while keeping the 3B backbone fixed, so differences are attributable
to the task formulation more cleanly than a simultaneous model swap.

The [FRUIT paper](https://aclanthology.org/2022.naacl-main.269/) studies
evidence-grounded article updating and reports gains from more explicit edit
controls. The separate [EdiT5 paper](https://aclanthology.org/2022.findings-emnlp.156/)
decomposes text editing into keep/reorder/insert decisions. Their published
architectures are related motivation, not claims about our model.

## Offline feasibility check, 2026-11-07

We parsed the already converted official FRUIT-WIKI files using their
sentence-copy markers. A strict single-window row has exactly one generated
literal block; all other source sentence indices appear exactly once and in
order; omitted source indices form one contiguous region; the generated block
occupies that region. With exactly one omitted sentence, old and replacement
spans each at most 400 characters:

| Split | All converted rows | Strict one-sentence, short replacement |
| --- | ---: | ---: |
| Train shard 00000 | 11,342 | 2,937 |
| Development shard 00009 | 11,559 | 2,956 |

Across these structurally filtered examples, an unordered-token check finds
that roughly 58% of gold-added word occurrences also occur somewhere in the
provided evidence (train mean 0.578, dev mean 0.579). This is **not** a factual
support rate: paraphrases lower lexical overlap, while incidental token
overlap inflates it. The silver labels and evidence relevance require manual
audit. The later-period publisher gold test stays unopened.

## Staged experiment

1. **Build and validate local targets without GPU.** Parse the copy markers
   into one old sentence, one gold replacement, neighboring sentence context,
   and the original evidence. Rebuild every extracted full article by replacing
   only that sentence; require exact equality with the converted gold article.
   Keep train and development shards separate and record the filter denominator.
2. **Audit a fixed sample.** Inspect 30-50 development examples for whether
   the replacement is actually supported by the supplied evidence and whether
   it changes a meaningful fact. Flag unsupported silver rows. This audit must
   be recorded before model selection.
3. **Oracle-locus upper bound.** Give the generator the gold location of the
   old sentence to revise, plus the old sentence and evidence. Train it to
   output only the replacement. The gold location is unavailable in a real
   deployment and must be prominently labeled as an oracle control. A tiny
   overlapping positive control should overfit before large training starts.
4. **Paired held-out evaluation.** On the same audited development rows,
   compare copying the old sentence, the unchanged backbone, and LoRA-trained
   local infilling. Measure replacement exact match, edit-content overlap,
   preservation of untouched context (guaranteed by deterministic copying),
   duplicate/deletion errors, and blinded factual support. Report both all
   structurally valid silver rows and the audited supported subset.
5. **Remove the oracle only if step 4 succeeds.** Train a separate evidence-
   conditioned locator to choose the old sentence/window, then evaluate the
   end-to-end system and abstention. Finally test on source-audited technical
   version changes; Wikipedia development performance cannot establish that
   transfer.

## Stop/go decision

Do not scale the existing full-article objective to 2,500 steps yet. A local
editor must first pass extraction roundtrips and the tiny positive control.
Then it must beat the copy baseline on changed content and show supported
updates in manual audit. If an oracle-locus local editor cannot do this, the
next work should diagnose evidence quality and target ambiguity rather than
increase model size. A successful oracle-locus result is an upper bound, not
the final RevProof result.
