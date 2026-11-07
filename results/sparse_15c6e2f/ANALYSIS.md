# Sparse editor GPU pilot: failure analysis

Archive received from DataHub: `revproof_sparse_compare_15c6e2f.tgz`, SHA-256 `3e1e6fb48b111bbc2b88493ef330bfa3707f52293ff78cb15dfa4b2a7c26f198`. The dataset hash in both protocols is `2f1c9f060639d6b2f72d12d1bd29589c590032e543c1178e0a48e6c352c3d796`. The 9 dev questions represent 7 revision groups. The already inspected v2 test was excluded.

| Model and condition | JSON valid | Anchor applicable | Exact span | Literal check |
|---|---:|---:|---:|---:|
| FLAN-T5-large prompt | 0/9 | 0/9 | 0/9 | 0/9 |
| FLAN-T5-large LoRA | 0/9 | 0/9 | 0/9 | 0/9 |
| Qwen2.5-1.5B prompt | 0/9 | 0/9 | 0/9 | 0/9 |
| Qwen2.5-1.5B QLoRA | 9/9 | 4/9 | 0/9 | 2/9 |

The FLAN-T5 LoRA training completed 3 epochs in 28.3 seconds (train loss 2.278, dev teacher-forced loss 1.663). Its adapter contains nonzero LoRA-B weights, yet all nine greedy dev outputs are *identical* to the pretrained prompt-only outputs. They are headings, echoed source fragments, or fact-card text, never JSON. This demonstrates no observed decoding gain from the 17-row LoRA run; it does not establish that the adapter was unused or that the backbone cannot edit. The longest dev script target is 250 tokens, versus 98 tokens for train, making the complex Flask case especially out of range.

Qwen QLoRA training completed 3 epochs in 20.4 seconds (train loss 1.256, dev teacher-forced loss 0.655). It learned the JSON surface form but the edit operation failed:

- 3/9 scripts were no-ops with identical `old` and `new` values (HTTP, Buffer change, Flask MAX_CONTENT_LENGTH).
- 2/9 selected an ambiguous or absent source anchor (Buffer MAX_LENGTH selected `2`; Node net omitted source punctuation/markup).
- 4/9 applied. One Buffer keep-question output may express the intended conditional change but fails exact and literal protected-fact checks; it needs blinded semantic review. The two Flask Python outputs pass all literal checks while producing `Flask supports Python 3.9 or newer and newer.` Both are semantically malformed. The SERVER_NAME output replaces `subdomain` with `subdomain_matching`, leaving an incorrect explanation.

The sparse edit *script* objective is not yet an improvement over the historical full-paragraph baseline. The structural hypothesis remains untested independently of the difficult localization and JSON tasks. Next diagnostic: train a replacement-only encoder-decoder with an **oracle edit window derived from the gold paragraph**. Its dev score is an upper bound with target-location leakage, not a deployable method or fair head-to-head result. If that fails, improve data quantity/task pretraining before implementing a locator; if it succeeds, tackle localization separately. A final comparison needs a new, untouched holdout and independent semantic labels.
