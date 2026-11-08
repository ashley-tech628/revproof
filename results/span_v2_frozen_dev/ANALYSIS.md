# FRUIT short-span editor v2: paired development analysis

## Provenance

User-run UCSD DataHub archive `revproof_span_v2_eval_results.tgz`, SHA-256
`73e6e5432f7cc09aa8f3a1fbb677cb397b06dc7ee0848c95f462c6a514a3fde9`,
received 2026-11-08. The seven original outputs are preserved here with
descriptive filenames and checksums in `manifest.json`. Recalculation from
both prediction files agrees with both paired summaries. All 100 IDs, source
articles, and silver target articles match between v1 and v2. These examples
were the previously frozen next 100 development IDs; they are **now inspected**
and must not be used again as an untouched test set.

The archive contains inference summaries and predictions, but no training
protocols or dataset manifest. The DataHub terminal output reported 3,000
v2 infiller steps and 9,965 token-eligible training rows. Exact model and
training-data provenance should be supplemented by the DataHub protocols.

## Paired results

| Condition | Silver full exact | Changed article | Mean edit-content word F1 |
| --- | ---: | ---: | ---: |
| v1 pipeline | 9/100 | 54/100 | 0.309 |
| v2 forced edit | **20/100** | 100/100 | 0.497 |
| v2 gated edit | **17/100** | 69/100 | 0.418 |
| v1 gold-location infiller diagnostic | 29/100 | 90/100 | 0.590 |
| v2 gold-location and gold-reference diagnostic | 44/100 | 97/100 | 0.754 |

The v2 forced condition gains 11 percentage points over v1 on identical
examples: 12 v2-only exact cases and one v1-only exact case. The gated
condition gains eight points: nine v2-only and one v1-only case. The two
gold-label diagnostics are **not deployment results** and differ in evidence
conditioning as well as training, so their difference does not isolate the
effect of generator weights. Full exact and replacement exact happen to agree
on these 100 copy-constrained examples; they are not independent outcomes.

Forced v2 edits every article; 80/100 outputs are therefore nonexact edits.
The fixed gate edits 69 articles, 52 of which remain nonexact. A nonexact
silver match is not necessarily factually wrong, and an exact match is not
necessarily factually supported. The blank blinded audit sheet has not been
scored, so no factual-support rate is reported.

## Where the v2 forced pipeline loses silver exact match

The selector's six sentence/evidence pairs contain the gold sentence for
93/100 rows and a silver reference pair for 89/100. Among at most eight
generated candidates, 52/100 have exactly the silver gold character span.
Twenty-five rows have the exact gold replacement phrase on that exact span;
the final selection is exact for 20 of these.

The 80 nonexact rows partition by the *first* unmet condition:

| First unmet condition | Rows |
| --- | ---: |
| Gold sentence absent from the six selected pairs | 7 |
| Gold sentence present, exact gold span absent from eight candidates | 41 |
| Exact gold span present, gold phrase absent at that span | 27 |
| Exact gold span and phrase present, but another candidate selected | 5 |

This is a diagnostic decomposition against the silver reference edit, not a
classification of factual mistakes. A different span can sometimes produce
the same correct final article. Of the 41 rows lacking the exact gold span,
35 have at least one predicted span overlapping it and 20 have a predicted
span containing it. Boundary refinement is thus a more promising next target
than merely increasing the number of sentence/evidence pairs. On this
frozen slice, correct phrase generation and final ranking are also material
bottlenecks; a better reranker alone can recover at most the five rows where
an exactly correct gold-span candidate was present but not selected.

## Edit-type sensitivity

This simple text partition is descriptive, not a semantic edit taxonomy.

| Silver edit text | Rows | v1 exact | v2 forced exact |
| --- | ---: | ---: | ---: |
| Case-only | 5 | 0 | 3 |
| One-word nonnumeric | 18 | 3 | 8 |
| Multiword nonnumeric | 62 | 6 | 9 |
| Contains a digit | 15 | 0 | 0 |

Three of the 12 v2-only exact wins are case-only changes. The gain on the
other 95 examples is still positive (17 vs 9), but the system has not solved
numeric updates. The single v1-only case is ID `62152797`; v2 generated the
gold phrase at different character boundaries, producing a nonexact article.
Examples of v2-only exact changes include IDs `57877408` (a club change) and
`52387657` (another club change), alongside capitalization and spelling edits.
The supplied FRUIT evidence references are heuristic; these examples have
not been independently fact-checked.

## Next experiment

Text-editing studies such as [FELIX](https://aclanthology.org/2020.findings-emnlp.111/)
and [EdiT5](https://aclanthology.org/2022.findings-emnlp.156/) separate
identifying retained/deleted source tokens from generating inserted text.
Our results suggest refining that identification step before increasing the
generator size. Train a **joint boundary refiner** on the nine training shards. Give it the
source sentence, one selected evidence item, and nearby QA span proposals;
train it to score complete start/end pairs, including expanded and contracted
token-boundary alternatives. Keep the copy-constrained infiller and selector
fixed initially, so a change in accuracy can be attributed to localization.
Use the previously inspected development rows for model selection, freeze a
new untouched set of IDs before training, and compare on that new set once.
The next 100 IDs are already frozen in `next_dev_100_v3_frozen.json` by the
same hash-ranking rule at ranks 200–299 after the initial local-dev exclusion;
the earlier rank ranges were checked against the archived v1 and v2 ID lists.
No v3 predictions for these IDs have been examined.
Also report nonexact but potentially useful edits through blinded factual
audit. Do not tune thresholds or boundary rules against the 100 rows here.
