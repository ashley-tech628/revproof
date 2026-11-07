# FRUIT local editor: held-out 500-step gate

## Provenance and scope

User-run UCSD DataHub archive `revproof_fruit_local_gate500_results.tgz`,
SHA-256 `9f335088de638beeafbf00721b406585080fd25d2c2c329ddef2ca7bd7b1b9f1`,
received 2026-11-07. The exact summary, protocol, training metrics, and
per-example predictions are preserved in `results/fruit_local_gate500/`.
The input data came from official FRUIT train shard 00000 and development
shard 00009. Structural filtering retained 2,940/11,342 and 2,955/11,559
converted rows respectively. The run hash-selected 2,000 train rows, 1,847
of which fit token limits, and trained for 500 optimizer steps (1.082 epochs,
25.1 minutes). For development, 100 rows were selected before token filtering,
and the same 95 eligible rows were used for every condition. Model:
`google/flan-t5-xl` with NF4 4-bit quantization and q/v LoRA (9,437,184
trainable parameters). The later-period FRUIT gold test was not used.

This experiment provides the **gold-derived edit location** and asks the
model to regenerate one old source sentence's replacement from evidence.
The rest of the article is copied deterministically. It is an oracle-locus
component study, not a deployable updater or a technical-document result.
The FRUIT development labels are silver. The
[FRUIT paper](https://aclanthology.org/2022.naacl-main.269/) distinguishes
distantly supervised training data from its separately audited gold test.

## Paired development results

| Condition | Exact replacement | Mean replacement word LCS F1 | Mean edit-content token F1 | Changed output |
| --- | ---: | ---: | ---: | ---: |
| Copy old sentence | 0/95 | 0.737 | 0.000 | 0/95 |
| Unchanged FLAN-T5-XL | 1/95 | 0.268 | 0.088 | 75/95 |
| 500-step LoRA local editor | 5/95 | 0.753 | 0.239 | 53/95 |

All five editor exact matches are on a token-level *single-span* gold edit;
the 95 rows comprise 31 one-span, 30 two-span, and 34 three-or-more-span
cases (Unicode word/punctuation tokenizer, `difflib.SequenceMatcher`,
`autojunk=False`). Thus exact is 5/31 in the simple stratum and 0/64 in the
other strata. Against copying, editor word LCS is higher on 28 rows, lower
on 17, and tied on 50. Its output is exactly the old sentence on 42/95.
Against the base model, edit-content token F1 is higher on 38 rows and lower
on 32. The only base exact example is among the editor's five exact examples.
The local editor's exact advantage over base on this fixed sample is four
rows. `full_exact` also equals 5/95 because the surrounding article is
deterministically copied; it is not an independent whole-document ability.

The 8-row positive control had 4/8 exact after training and evaluating on
those same rows, showing the pipeline can learn some local outputs but not
that it generalizes. It must not be mixed with the 95-row development score.
The tokenizer's 512-token warning was nonfatal in this completed run, which
used a 1,024-token acceptance limit. The training loss 0.702 is not a quality
or support metric.

## What the predictions show

All five exact IDs: 39246450, 51423380, 44118054, 24483433, 7471706.
The evidence directly supports the spelling change for 39246450, the
Aldershot Town and Chicago Fire transfers for 44118054/24483433, and the
Lee Enterprises newspaper listing for 7471706. ID 51423380 exactly matches
the silver target `current senior whip is Kieran McAnulty`, but the evidence
calls him `Chief Government Whip`; equivalence of the role and the temporal
word `current` require independent verification. Exact match is therefore
not proof of factual validity.

Several non-exact edits convey the main evidence-supported change:

- ID 60294758 changes the job to Leeds vice-chancellor, retaining the old
  birth-date format; gold changes the date format too. Its unordered edit F1
  is 1.0 even though the strings differ.
- ID 15496245 changes the website owner to Red Ventures and is grammatical;
  the silver target has `that currently maintained`.
- IDs 36743800 and 53785730 use `Banijay Group` and `Atalanta B.C.` where
  silver uses `Banijay` and `Atalanta`.
- ID 59714894 updates the football club but keeps older nationality wording,
  which the provided evidence does not adjudicate.

Other changes are unsafe or wrong. ID 37690762 turns a table's `163,000`
distance into a claim that a cluster contains 163,000 stars. ID 37875939
names Lori instead of the evidenced transfer destination Gandzasar Kapan.
ID 679630 transfers the old `Serenity Inc.` attribution to ArcaOS. The
model also damages names with non-ASCII characters: `Nguyễn` becomes
`Nguyn` (47804210), `Šego` becomes `ego` (57778301), and Cyrillic text is
partly lost (32312079). Among 27 rows whose old sentence contains non-ASCII
characters, 19 editor outputs change the sentence and 9 of those have fewer
non-ASCII characters than both old and gold. This count is a screening proxy;
the named examples confirm actual damage.

Silver target quality varies. ID 47804210's gold contains broken markup
(`clubĐồng Tháp`); ID 37690762's gold has missing numerical fields. ID
23728791's model prediction `caretaker manager` is supported by multiple
provided 2020 snippets, while the silver target adds a career-history claim.
These are reasons to audit both predictions and targets, not to retroactively
change exact-match denominators.

## Fixed-sample preliminary human review

`results/fruit_local_gate500/provisional_audit_40.csv` records a single
reviewer's initial evidence-based judgement of the 40 hash-selected audit
IDs, selected before viewing model outputs. Three were token-ineligible,
leaving 37 reviewed predictions. Categories are deliberately coarse:

| Initial category | Count/37 | Meaning |
| --- | ---: | --- |
| `supported_core` | 9 | Main intended change appears supported; other sentence details may differ |
| `partial` | 3 | Some supported change, but an important part is missed |
| `different_evidence_supported` | 1 | Evidence supports the model's alternative change, not the silver target |
| `unchanged` | 9 | The model leaves the old sentence unchanged |
| `incorrect_or_corrupt` | 12 | Contradictory fact, unsupported attribution, or damaged text |
| `uncertain` | 3 | Evidence or target ambiguity prevents a confident judgement |

This is **not** a formal factual-support rate: the sample is small, the
evidence is silver/retrieved, the categories are single-reviewer and not
blinded, and preserved old claims have not all been independently checked.
For a credible report, a second reviewer and source-linked technical-document
audit are needed. The 9 `supported_core` rows also include a spelling-only
edit; they are not all new factual updates.

## Decision

The local objective improves over the unchanged backbone on exact and
edit-content diagnostics, but 5/95 exact with an oracle edit location and
12/37 clearly problematic predictions is not sufficient for an autonomous
updater. More steps or a larger backbone alone would not fix target ambiguity
or the damage caused by regenerating unchanged names and numbers.

The next controlled model should copy untouched tokens from the old sentence
and generate only the changed phrase, with a location/tagging step and an
evidence-linked validation or abstention step. On this silver shard, 1,118/
2,940 train and 1,135/2,955 development rows have exactly one token-level
changed span; 1,982/2,940 and 1,988/2,955 have at most two. The one-span
subset offers a limited diagnostic before attempting multi-span edits.
This is consistent with the decomposition into token preservation and
insertion in [EdiT5](https://aclanthology.org/2022.findings-emnlp.156/) and
keep/delete/phrase insertion in
[LaserTagger](https://aclanthology.org/D19-1510/), without claiming that our
current model implements either published architecture. Oracle span
positions must be removed in an end-to-end evaluation. The next gate should
measure factual support, preservation of unchanged text, and abstention on
an audited set, then test transfer to versioned technical documents.
