# FRUIT local editor: 120-step training-overlap control

User-run UCSD DataHub result, 2026-11-07. Model: `google/flan-t5-xl` 3B,
NF4 4-bit base, LoRA `q`/`v` (9,437,184 trainable parameters). The model
received the *gold-derived old sentence location*, old/neighboring sentences,
and FRUIT evidence, and generated only the replacement. The eight selected
rows came from FRUIT training shard 00000 and were used for both training and
evaluation. This is a **mechanics/memorization control** with 60 passes over
eight rows, not a held-out result, oracle-free update, or factual-support
assessment. Training took 310.6 seconds; final average training loss was
0.627. The exact sample IDs and file hashes are in the user-run summary.

| Condition | Exact replacement | Changed output | Mean replacement word LCS F1 |
| --- | ---: | ---: | ---: |
| Copy old sentence | 0/8 | 0/8 | 0.807 |
| Unchanged FLAN-T5-XL | 0/8 | 5/8 | 0.569 |
| LoRA local editor | 4/8 | 7/8 | 0.870 |

The nonzero exact score and improved replacement overlap show that the
adapter, serialization, and decoding can learn this local target on seen
examples. Four of eight exact after 60 epochs is only partial memorization;
we need the per-row predictions to diagnose the misses. It gives a reason
to test held-out generalization, not evidence that the model improves
real-world technical documentation updates.

The original evaluator's `mean_added_token_f1` was 0.125 for the copy
condition because its helper returns 1 when neither prediction nor gold
adds any tokens, even if the gold deletes text. That score should not be
interpreted as a successful copy edit. The evaluator has been revised to
report addition-plus-deletion `mean_edit_content_f1`; exact match and word
LCS figures above are unaffected. The old lexical score was 0.142 for the
unchanged base and 0.659 for the editor, but it is omitted from the main
table for this reason.
