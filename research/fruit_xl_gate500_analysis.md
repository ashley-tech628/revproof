# FLAN-T5-XL FRUIT early gate: 500 optimizer steps

## Provenance and scope

User-run DataHub Blackwell GPU pilot, archived in
`revproof_fruit_xl_gate500_results.tgz` (SHA-256:
`3ea73f6c257c055f2d99443fbebd546c9655c9df657544f9ccc90ea9b1a13710`)
on 2026-11-07. The extracted summary, per-row predictions, and training
metrics are retained in `results/fruit_xl_gate500/`. Model:
`google/flan-t5-xl`, NF4 QLoRA with LoRA on `q` and `v`. Training selected
2,000 examples from official FRUIT-WIKI train shard 0; 500 optimizer steps with
gradient accumulation 4 yielded 1.075 epochs over the token-eligible subset.
The fixed, hash-selected silver development subset came from official train
shard 9: 30 selected, 29 within token limits. Development JSONL SHA-256:
`f11fc3eeb08f5e7207a0da0f1515da3f45379b7aae37244da8606c20fa8d6259`.
The later-period publisher gold test was not used. The official
[FRUIT release](https://github.com/google-research/language/tree/master/language/fruit)
describes the shard split and the distant supervision. This is a small
development diagnostic, not the published benchmark.

The three conditions use exactly the same 29 rows. `copy` returns the old
article; `base` is the unchanged 4-bit backbone; `editor` adds the 500-step
LoRA adapter. Deterministic greedy decoding and the same length budget are
used for base and editor. All scores below come from the archived JSONL and
summary, not from the [paper's](https://aclanthology.org/2022.naacl-main.269/)
reported numbers.

| Condition | Full text exact | Word LCS F1 | Added-token F1 | Changed output | Parser accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy old article | 0/29 | 0.784 | 0.000 | 0/29 | 29/29 |
| Unchanged FLAN-T5-XL | 0/29 | 0.147 | 0.019 | 28/29 | 28/29 |
| 500-step LoRA editor | 0/29 | 0.723 | 0.042 | 12/29 | 28/29 |

Paired comparison against `copy`: the editor has higher word LCS F1 on 1/29
rows, lower on 9/29, and equal on 19/29. Sixteen editor reconstructions are
exactly the old article. One is invalid because the generated `[17]` refers
outside the source. The model emitted at least one source-copy marker on 23/29
rows, so it learned the compressed output convention, but added-token F1 is
positive on only 5/29. The mean training loss was 1.112 over 1,721 seconds;
training loss is not a development quality metric.

## Output audit

- **ID 9207111, 174th Attack Wing:** evidence explicitly lists MQ-9 Reaper and
  the gold text adds that aircraft. The editor duplicated the existing first
  sentence and omitted the aircraft. This is a missed supported update.
- **ID 23556579, Dark River:** the gold changes `Sutherland Sound` to
  `Te Hāpua / Sutherland Sound`. The editor duplicated the old river sentence
  and missed the name change.
- **ID 42923480, Khovaling:** the editor added `Khatlon Region of Tajikistan`,
  producing the sole row with better word LCS F1 than copying. It did not add
  the gold population `10,601`, which appears in the evidence. This is a
  partial update, not a correct revision.
- **ID 65565, progressive metal:** the editor appended an Angband sentence
  from the context rather than performing the gold's progressive-metal edit.
  Mere evidence overlap did not make the sentence appropriate for the article.
- **ID 435802, Edwards syndrome:** output contains `[17]` with no matching
  source sentence, so reconstruction fails.
- **IDs 54299574 and 1779594:** the editor repeats source-copy indices,
  duplicating article passages. Both are accepted by the current permissive
  parser despite being undesirable revisions.

The other changed rows mostly keep old facts, make superficial changes, or
damage names/diacritics. These examples are diagnostic; the silver references
and external evidence have not been subjected to a complete independent
factual-support audit.

## Interpretation and next gate

The adapter sharply improves over the unchanged backbone's ability to emit
source-copy structure, but this checkpoint does **not** improve revision
quality over copying the old article on this development sample. More training
steps alone are not justified by the evidence. A better next experiment should
isolate changed sentences and relevant evidence, generate only local
replacements, and deterministically copy the untouched remainder. Evaluate
against the same dev rows with a paired copy baseline, explicit edit coverage,
duplicate/malformed marker checks, and a manually audited factual-support
sample. Transfer from Wikipedia to audited technical version changes remains
unproven.

The current `reconstruction_valid` field means only that our lightweight
parser found no out-of-range source-copy marker. Plain text with no copy
markers and duplicated in-range markers can pass. Word LCS F1 mostly rewards
retaining unchanged text. Added-token F1 compares unordered token counts and
does not verify context, order, or factual support. Do not describe these as
grounding or official FRUIT UpdateROUGE scores.
