# Oracle-location infilling: FLAN-T5-large diagnostic

DataHub archive: `revproof_oracle_results.tgz`, SHA-256 `4967a9223ad854bfdc3207fd5609459bf0873fbd836e8b89f2e8d29878324e9`. The protocol and training metadata report the frozen v2 dataset SHA-256 `2f1c9f060639d6b2f72d12d1bd29589c590032e543c1178e0a48e6c352c3d796`. Both eval commands used dev only: nine questions from seven groups. All three commands exited successfully. The v2 test was excluded.

**Interpretation constraint:** the edit location was derived from `new_span` for every dev row. This is a target-leaking diagnostic upper bound, not a deployable result or a fair comparison to the sparse-script model.

| Condition | Exact replacement | Exact reconstructed paragraph | Literal checks |
|---|---:|---:|---:|
| FLAN-T5-large prompt-only | 0/9 | 0/9 | 1/9 |
| FLAN-T5-large LoRA | 0/9 | 0/9 | 1/9 |

FLAN-T5-large LoRA trained for 3 epochs on 17 real train questions from 15 groups, with learning rate 1e-4, gradient accumulation 2, and seed 42. The run took 27.7 seconds, reported training loss 2.394 and dev teacher-forced loss 2.384. Longest target lengths were 42 train tokens versus 190 dev tokens. These losses do not establish successful free generation.

The prompt and LoRA decoded replacements are identical on **8/9** dev questions. The only changed output removed a stray `/edit>` marker without producing the required edit. Outputs repeatedly copy large parts of the input paragraph or malformed `<edit>` markers instead of only the replacement. In the Flask Python change case, both conditions pass the literal checks but reconstruct the malformed sentence `Flask supports Python Flask requires Python 3.9 or newer; retain the recommendation to use the latest Python version. and newer.` Exact match and literal checks therefore need a separate semantic audit.

Because replacement generation fails even with target-derived location, the current FLAN-T5 run cannot support the claim that localization is the sole bottleneck. This does not prove localization is unimportant. The evidence is consistent with task mismatch and only 17 train questions; it cannot distinguish those factors from model choice. The next controlled diagnostic uses the **same oracle location, data, split, and training budget with Qwen2.5-1.5B-Instruct** as a causal model. It is still an oracle diagnostic and does not address deployment or publication-level evaluation. Data scaling and a fresh untouched holdout remain necessary after the architecture decision.
