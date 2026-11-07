# RevProof architecture decision after the v2 pilot

Date: 2026-11-06. This is an experiment plan, not a measured improvement.

## Evidence from the current pilot

The v2 run used Qwen2.5-1.5B-Instruct with a 4-bit LoRA adapter and full-paragraph next-token supervision. Its 17 train questions represent only 15 independent revision groups. The fixed dev set has 9 questions from 7 groups; the four-question test set has 4 groups. The test predictions have now been inspected, so any subsequent architecture choices made from them are exploratory. Freeze a *new* holdout before claiming a final comparison.

| Method | Dev exact | Test exact |
|---|---:|---:|
| Deterministic template | 4/9 | 1/4 |
| Same-base prompt-only | 0/9 | 0/4 |
| Full-paragraph QLoRA SFT | 0/9 | 0/4 |
| SFT, four samples, literal filter | 1/9 | 0/4 |

Exact match is a diagnostic, not a semantic-validity score. One non-exact SFT test sentence appears to express the correct `ReadableStream` change; other outputs retain stale facts. Conversely, all four test template outputs pass the literal checker while only one is exact. An independent semantic audit must check every target fact, protected fact, scope, syntax, and unsupported addition.

A word-level `SequenceMatcher` comparison of one old/new paragraph per independent revision group found 548 words in train targets, of which 104 were changed (81.0% copied); 180 words in test targets, of which 29 were changed (83.9% copied). The dev copy fraction is only 46.1%, dominated by one large Flask rewrite. These are descriptive counts, not token-level training-loss contributions or a causal proof. They motivate testing an editor that generates only the changed material.

## Recommended generative structure

**Fact-conditioned, copy-constrained span infilling** is the leading hypothesis. It adapts the *tag/keep and insert* structure of FELIX and EdiT5 to versioned technical documentation. Let old paragraph be `x`, verified target fact card be `f`, and edit locations be `z`. Train a locator `p_phi(z | x,f)` and an infiller `p_theta(y_edit | x,z,f)`. Construct the final paragraph by copying every token outside `z` exactly and inserting only `y_edit` inside `z`:

`p(z, y_edit | x, f) = p_phi(z | x, f) p_theta(y_edit | x, z, f)`.

Use a bidirectional encoder-decoder infiller (first `google/flan-t5-large`, then a larger checkpoint only if a memory preflight succeeds). The encoder reads left and right context plus the complete fact card; the decoder emits replacement text. Deterministic reconstruction locks untouched text. The locator cannot use the gold `new_span` at inference. Gold old/new diffs may supervise locator training, but an *oracle-located* score must be labeled an upper bound. For a first, simpler deployable version, predict a compact edit script containing old and new substrings, apply it only when the old substring matches uniquely, and abstain otherwise. This tests sparse generation without an oracle location.

This is a better-matched structure, **not** a guaranteed improvement. It directly targets stale old facts, dropped protected clauses, and wasted generation on copied text. A long multi-sentence rewrite may require multiple edit windows; if the target facts are incomplete, no architecture can reconstruct the missing information reliably.

## Comparison that can identify the cause of any gain

1. Keep the existing full-paragraph Qwen run as the historical baseline.
2. Train/evaluate **Qwen edit-script output** with the same fact cards, groups, and candidate budget. This isolates sparse-output factorization from a backbone change.
3. Train/evaluate **FLAN-T5-large edit-script output** under the same protocol. This tests the encoder-decoder backbone.
4. Optionally evaluate `grammarly/coedit-large` as an editing-pretrained comparator. Its model card states a CC BY-NC license, so it is a research comparison, not a commercial deployment default.
5. If the smaller models show a useful signal and hardware permits, try `google/flan-t5-xl` (3B) with LoRA. Do not infer that size alone caused a gain unless the sparse-output and full-paragraph ablations are controlled.

Report exact match and edit distance as secondary diagnostics. Primary evaluation: independently adjudicated target-fact correctness, protected-fact preservation, unsupported additions, syntax/format, and an abstention-coverage curve. Count independent revision groups, not merely questions, in uncertainty estimates. One human should review model outputs without seeing model names. Expand the source-linked corpus across independent releases and projects before making generalization or publication claims.

## Why this choice rather than a masked diffusion LLM

Masked diffusion offers bidirectional generation but would add substantial implementation and evaluation risk here. The present failure is sparse factual editing with only 15 train groups, and FELIX/EdiT5 give a directly relevant decomposition. A diffusion model can be a later ablation once a credible copy-constrained editor and a larger benchmark exist.

## Compute decision

The current UCSD DataHub RTX PRO 6000 Blackwell 24 GB MIG has already run the QLoRA pipeline successfully. Use it for the first 770M-class encoder-decoder and sparse-output ablations. Colab Pro is useful if its *actually assigned* GPU has enough free memory for the 3B experiment or if DataHub availability is a practical problem. Google does not guarantee a particular Colab Pro GPU, runtime length, or unlimited compute; do not assume that buying Pro means an A100. On each Colab session, record `nvidia-smi` output, free GPU memory, PyTorch/CUDA, allocated compute units, dataset hash, checkpoint, seed, and package versions. Persist checkpoints and predictions outside the transient runtime.

## Primary sources

- FELIX: https://aclanthology.org/2020.findings-emnlp.111/
- EdiT5: https://aclanthology.org/2022.findings-emnlp.156/
- CoEdIT: https://aclanthology.org/2023.findings-emnlp.350/
- FLAN-T5 XL model card: https://huggingface.co/google/flan-t5-xl
- CoEdIT large model card: https://huggingface.co/grammarly/coedit-large
- Colab resource FAQ: https://research.google.com/colaboratory/faq.html
