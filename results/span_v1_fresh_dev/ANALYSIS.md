# FRUIT short-span editor v1: held-out silver development analysis

## Provenance and design

User-run UCSD DataHub archive `revproof_span_v1_results.tgz`, SHA-256
`7497cc22972af0a0b6933752cb0627d2084d8f11fd530fbfd1681a36c875a3d4`,
received 2026-11-07. The archive's original manifest, training protocols,
metrics, summary, 100 predictions, and fixed 40-row blank audit sample are
preserved in this directory. Recalculation from `predictions.jsonl` agrees
with `summary.json` on exact counts and the error partitions below.

The strict short-span converter retained 1,096/2,940 local train rows and
1,082/2,955 local development rows from the first publisher training shard
and its development shard. Relative to the full converted shards, this is
1,096/11,342 train and 1,082/11,559 development rows. Long, multispan,
insertion-only, and structurally ambiguous changes were excluded. The
infiller had 1,014 token-eligible train rows and ran for 500 steps (1.97
epochs; 23.1 minutes) with `google/flan-t5-xl` 4-bit NF4 plus q/v LoRA. A
`FacebookAI/roberta-base` QA locator was trained for three epochs (2,645
positive/negative sentence examples; 54 seconds). Its QA head started with
newly initialized weights. See `infiller_protocol.json` and
`locator_protocol.json` for file hashes and settings.

The development selection first excludes by ID the 100 hash-ranked local
development rows from the prior pilot, then takes the first 100 hash-ranked
span rows. All conditions below use those same 100 IDs. These rows were new
to this experiment when evaluated. They are now inspected development data,
so subsequent tuning must be evaluated on a different fixed slice.

## Paired results

| Condition | Exact full/replacement | Changed | Mean edit-content word F1 |
| --- | ---: | ---: | ---: |
| Copy source | 0/100 | 0 | 0.000 |
| Unadapted FLAN, gold span | 0/100 | 35 | 0.130 |
| Adapted infiller, gold span | **31/100** | 91 | 0.634 |
| Trained locator + adapted infiller | **15/100** | 67 | 0.417 |

The gold-span rows are **oracle-location component studies**. The 15/100
pipeline condition chooses the edit location from source and evidence at
inference. Every accepted edit copies characters outside its predicted
span, but the predicted span can include the wrong fact. Exact full text
equals exact replacement in this sample because reconstruction is
deterministic; these are not independent model abilities. Word F1 is a
lexical diagnostic and can reward a partly wrong claim. None of these
numbers is a factual-support rate. This short-span sample is different from
the earlier 5/95 one-sentence experiment, so those exact-match counts are
not a paired head-to-head comparison.

## Where the end-to-end system loses accuracy

The raw locator chose the gold sentence on 80/100, versus 64/100 for a
simple evidence-word-overlap sentence baseline. It chose the exact gold
character span on 30/100. The score threshold caused 29/100 abstentions;
13 of these had the correct sentence, and two even had the exact span. A
raw locator hit is counted in the 80/30 figures even if the pipeline
subsequently abstains.

The 100 runs partition as follows:

| Raw locator and action | Rows | Pipeline exact |
| --- | ---: | ---: |
| Wrong sentence | 20 | 0 |
| Correct sentence, abstained | 13 | 0 |
| Correct sentence, wrong span, acted | 39 | 4 |
| Correct sentence, exact span, acted | 28 | 11 |

Of the 31 gold-span infiller exact examples, 14 are also pipeline exact.
The other 17 are lost through six wrong-sentence choices, three abstentions
with the correct sentence, and eight wrong spans with an accepted edit.
One pipeline exact match occurs with a different, wider span even though
the infiller's gold-span output was not exact. Thus exact span boundaries
are not always necessary to obtain the correct final text. On the 20
wrong-sentence rows, the pipeline abstained 16 times; three of the four
accepted candidates made a nontrivial wrong-sentence change.

## Case review against the *provided* evidence

The fixed 40-row audit export was inspected together with additional
pipeline-exact examples. This was a **provisional, unblinded case review**,
not an independently sourced fact-check or an overall support rate. Examples:

- **Directly supported exact updates:** ID `178038` changes Dovre's county
  from Oppland to Innlandet, matching a supplied municipality table. ID
  `60921263` changes Frederik Ibsen's destination from FC Copenhagen to
  Kolding IF, matching a departure table. ID `33803642` changes Anthony
  Tolliver's team to Sacramento Kings, matching supplied transaction text.
- **Useful nonexact changes:** ID `43511708` selects the evidence-supported
  club CSM București but uses `ş` where the silver target has `ș`. ID
  `54988405` changes Cardiff City to Solihull Moors while omitting the
  silver target's league label; the supplied transfer row supports the
  destination but does not itself establish that label. ID `54157915`
  selects Sligo Rovers but omits the league. ID `60255338` changes the club
  to AS Ain M'lila as spelled in the supplied transfer tables; the silver
  target uses `Aïn`. Exact match counts all of these as wrong.
- **Unsafe changes:** ID `36699509` replaces the band Red Krayola with
  Rachel Williams, who the evidence identifies as the cover model. ID
  `493361` replaces a region name with a war title from another evidence
  passage. ID `54483771` changes the representative to Scott Wyatt but
  leaves `since 2006`, transferring the old representative's date to the
  new person. ID `8561226` appends `So Vicente` to the club Imperatriz,
  apparently copying a neighboring table field. These show why high word
  overlap is not enough.
- **Silver-target/evidence tension:** ID `13019633` exactly matches
  `published by Rossel`, while the provided evidence says Rossel led a
  consortium that *acquired* the magazine. That passage alone does not
  establish the publisher relationship. ID `8170373` has the typo `ad`
  in the silver replacement. ID `55196637` has a nationality change in
  the silver target that is not established by the provided loan table;
  another supplied passage calls the player Moroccan. These cases require
  independent review before a support claim.

The longer source article is reconstructed from the old text, so unchanged
sentences do not suffer generative paraphrasing. An overbroad predicted
span can still alter unrelated facts *within that sentence*.

## Next experiment

The next design should improve boundary selection and evidence-role
handling while retaining deterministic copy and the same FLAN backbone.
Candidate spans from the locator can be reranked or trained with a more
explicit edit-boundary objective; entity strings should be checked against
evidence with attention to Unicode and table-column roles. Abstention
should be calibrated on this inspected slice only, then evaluated once on
the pre-frozen next 100 development IDs in `next_dev_100_frozen.json`.
Those IDs are an additional development slice, not the untouched FRUIT
later-period gold test. Training on additional publisher shards is a
separate data-scale ablation and must retain shard 9 as development.
