# Research reset: evidence-grounded, version-aware technical-document editing

Date: 2026-11-06. This is a literature-grounded research design, **not a measured improvement**. It supersedes the model-selection recommendation in `architecture_decision.md` and pauses the next GPU run until the data and evaluation are repaired.

## 1. What the current 0/9 actually says

The oracle-location FLAN-T5 experiment gave the model a target-derived edit window. On nine dev questions, neither the pretrained model nor its 17-row, three-epoch LoRA produced an exact replacement. Eight decoded replacements were identical before and after tuning. The only literal-check pass reconstructed a grammatically and semantically malformed Flask sentence. Thus the run shows failure of this **model + training setup + output contract** even when location is supplied. It does **not** prove that generative editing is impossible, nor isolate localization as irrelevant.

The present benchmark has additional design problems:

1. **Insufficient and unbalanced supervision.** Training has 17 questions/15 independent revision groups; dev has 9/7. Gold replacement length has median 47 characters and maximum 128 in train versus median 103 and maximum 669 in dev. The 669-character Flask rewrite is outside the observed train target range. The three-epoch run made only 27 optimizer steps. The literature's editing models are generally trained on far larger datasets; even EdiT5's explicit 450-example low-resource condition is much larger than 17 examples. These facts make undertraining plausible but do not prove it is the only cause.
2. **The single-window target is unstable.** `gold_script` uses a longest common prefix/suffix and expands to word boundaries. Distant factual changes collapse into one long replacement; a one-sentence insertion may ask the decoder to output an entire clause. This target is an artifact of a character diff, not a natural semantic edit plan. It also forces one location when the revision may require several.
3. **The fact card is not an exact target.** A fact card may say “Flask requires Python 3.9 or newer” while the reference says “Flask supports Python 3.9 and newer.” Many valid phrasings cannot exactly match the single reference. Yet the 0/9 also includes clearly malformed outputs, so exact-match mismatch is only part of the story.
4. **The intended inference task is mixed.** The handwritten v2 fact cards were produced after reading the target revisions. That is a legitimate *controlled editing* input, but it is not independent evidence retrieval or prediction of unknown release changes. The oracle experiment additionally derives the edit location from the target. Neither result should be described as a deployable evidence-grounded updater.
5. **Training mechanics have not been positively validated.** Training/evaluation commands exited successfully and LoRA changed the previous sparse model's weights, but the oracle run does not include a free-generation result on train examples or a 1–3-example memorization check. Before interpreting more zero scores as a model limitation, check that the exact serialized input, target tokens, EOS behavior, adapter loading, and decoding are consistent and that the model can learn a tiny controlled case. Do this before expensive architecture search.

## 2. What prior work already covers

| Primary paper | Directly relevant result | Consequence for RevProof |
|---|---|---|
| [Fact-based Text Editing (ACL 2020)](https://aclanthology.org/2020.acl-main.17/) | Edits a draft from facts with keep/drop/generate actions and a copy mechanism. | “Fact-conditioned edit actions” alone are established prior art. |
| [FRUIT (NAACL 2022)](https://aclanthology.org/2022.naacl-main.269/) | Updates old text from new evidence; FRUIT-WIKI has over 170K distantly supervised items and a 914-instance evidence-audited evaluation. Its EDIT5 emits copied sentence IDs and updated text with evidence references. | This is the closest task formulation. RevProof must distinguish technical documentation, version provenance, conditional API semantics, and its evaluation design. |
| [FELIX (EMNLP 2020)](https://aclanthology.org/2020.findings-emnlp.111/) and [EdiT5 (EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.156/) | Decompose editing into preserving source tokens and inserting missing text; EdiT5 also handles reordering. EdiT5 beats plain T5 in its reported 450-example low-resource sentence-fusion condition. | Copy-constrained generation is well motivated, but our two-field, single-window JSON script is not an implementation of these architectures. |
| [CoEdIT (EMNLP 2023)](https://aclanthology.org/2023.findings-emnlp.350/) | Instruction-tuned text editor trained on 82K editing instructions. | Stronger editing pretraining is a relevant baseline; 17 examples cannot be treated as comparable supervised training. |
| [FineEdit (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.118/) | Studies precise edits in structured domains using a benchmark of over 30K tasks. | Technical formatting and exact symbol handling deserve dedicated tests, not only general semantic scores. |

FRUIT explicitly shows that whole-text overlap metrics can reward copying an obsolete source and that model updates can hallucinate numbers/dates. Those are directly relevant to versioned API docs. Its findings support evaluating changed claims and evidence attribution, not relying on ROUGE or a single exact reference.

## 3. Revised research question and model

**Research question:** Given a versioned old technical paragraph `x` and independently sourced target-version evidence `e`, can a model make the *smallest supported set of edits* while preserving unchanged API symbols, conditions, and Markdown/reStructuredText syntax, and abstain when `e` does not justify a safe update?

Let `z` be a sequence of `KEEP`, `DELETE`, and `INSERT` actions over source tokens, potentially spanning several clauses. Let `r` be generated insertions. The model factorization is

`p(z,r | x,e) = p_phi(z | x,e) p_theta(r | x,e,z)`.

A deterministic composer produces `y = Apply(x,z,r)`, copying every `KEEP` span byte-for-byte. For the first practical system, `p_phi` can be a source-token tagger or span start/end predictor and `p_theta` an instruction-tuned generative model. A later integrated model can share an encoder. Evidence IDs and target version should be explicit inputs. Each generated claim should map to a supporting evidence span; unsupported or contradictory evidence should trigger abstention. A verifier can score candidate edits for fact coverage, protected-fact retention, unsupported additions, and valid document syntax. This verifier must be evaluated separately and must not silently turn string heuristics into “semantic truth.”

The intended learning objective is multi-task: plan cross-entropy for `z`, autoregressive negative log likelihood for `r` conditioned on the plan and evidence, and a contrastive/verification loss on deliberately corrupted versions or facts. Exact symbolic constraints (API identifiers, version numbers, code spans, markup delimiters) are best enforced by the composer and validators, not merely encouraged through a soft loss. A calibrated reject option should be measured by a coverage-versus-validity curve.

**The contribution to test**, rather than claim in advance, is whether combining *version-specific evidence provenance + multi-span copy-constrained generation + abstention* improves factual correctness on technical-document updates. FRUIT, FactEditor, FELIX, and EdiT5 already establish the general editing components. The domain-specific combination and a rigorously audited benchmark could be novel if demonstrated across independent projects and release transitions.

## 4. Repair data before selecting a larger backbone

1. Define two tasks separately. **Controlled revision** receives a human-approved complete change request; **evidence-grounded updating** receives release notes, changelog or PR text independently authored before the target documentation edit. Do not combine their scores.
2. For each training example retain old text, target text, evidence provenance, version pair, group ID, and human-audited support labels. Generate high-precision silver edits from official tagged histories, then manually audit a stratified sample. Exclude edits whose new facts cannot be recovered from the allowed evidence, or label them `insufficient_evidence` for abstention. Do not use the target paragraph or a target-derived fact card as inference evidence.
3. Split by **release transition and project**, not by question, and deduplicate near-identical edits before splitting. Keep a new untouched final holdout; the existing v2 test has already influenced architecture decisions.
4. Align edits at sentence/clause/token level and allow multiple disjoint edit spans. Keep literal API tokens and markup intact. Inspect a stratified set of alignments manually; treat ambiguous alignment as a label-quality problem rather than forcing a single character-diff answer.
5. Build a modest but real scale-up before claiming model gains: hundreds of high-confidence training revisions and a separately audited held-out set, followed by learning curves. The exact achievable count depends on evidence-support filtering. Do not fabricate fact cards to inflate sample size, and do not treat duplicate questions as independent revisions.

## 5. Predeclared comparison and evaluation

First pass the **positive control**: a tiny model/training slice must learn a few audited edits and reconstruct them on train. Then compare on a new frozen holdout using the *same allowed evidence* and candidate budget:

1. Copy old text / no-op baseline, with an explicit abstain option.
2. Deterministic exact-value replacement for simple scalar edits.
3. Strong instruction-tuned full-paragraph editor, with and without task fine-tuning.
4. Sentence-level compressed edit output with copy IDs, inspired by FRUIT EDIT5.
5. Proposed multi-span tagger + generative infiller + deterministic composer.
6. Proposed system plus evidence-grounded verifier/abstention.

Keep oracle locations as diagnostic-only conditions. Report fact coverage, retained protected facts, unsupported claims, condition/negation/numeric accuracy, technical syntax validity, edit sparsity, and acceptance coverage. Exact match and changed-span similarity remain secondary. Blind human review should include literal-check passes and failures. Report denominators by **independent revision group** and uncertainty by group-level bootstrap; four held-out groups are too few for a strong generalization claim. Include separate results for simple scalar, conditional, and long multi-clause revisions.

## 6. Immediate decision

Pause the planned Qwen oracle GPU run. It is useful as a limited model-family diagnostic but cannot fix the data, target, and evaluation problems above. Before further GPU work, inspect serialization and run the positive control, curate additional evidence-supported revisions, and freeze the new benchmark. In the mid-term report, present the existing zero results honestly as a failure analysis and motivate the revised method mathematically; do not claim the proposed architecture has already improved accuracy. The report can discuss this design without waiting for a new training run.
