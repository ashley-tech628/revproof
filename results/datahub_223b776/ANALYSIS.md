# Real-data GPU pilot: diagnostic analysis

This is an analysis of the user's DataHub artifact archive `raw_datahub_artifacts.tgz` (SHA-256 `7dca0819c630c014f0b42f07ee38f64d14b6624ad26cf930a0dbfe58fff05f0d`). The run used dataset SHA-256 `ae9862b22dd5cb7f7733a7d4df8631551de2c660a7c6eacb45ce2fc7119df54f` and code commit `223b776`. `diagnostics.json` was produced by `analyze_pilot_artifacts.py`; that script also regenerates `selected_predictions_for_human_audit.csv` from the archived raw outputs. The CSV's human-validity fields are intentionally blank pending an independent, explicit annotation pass.

## Setup and measured outcomes

Qwen2.5-1.5B-Instruct was trained with 4-bit QLoRA for three epochs on 17 question rows from 15 Node.js revision groups. Dev had five questions from four later Node.js groups; test had four questions from three Flask groups. The DataHub GPU was an RTX PRO 6000 Blackwell 24 GB MIG. The final teacher-forced train loss was 0.126 and dev loss 0.154. These losses measure token prediction with the gold prefix, not the correctness of free-running generation.

| Generator | Dev exact / 5 | Test exact / 4 | Test selected literal checks / 4 |
|---|---:|---:|---:|
| Template | 2 | 2 | 4 |
| Prompt-only, greedy | 0 | 0 | 1 |
| QLoRA SFT, greedy | 1 | 0 | 2 |
| Prompt-only, four sampled then filtered | 0 | 0 | 3 |
| QLoRA SFT, four sampled then filtered | 1 | 0 | 3 |

The generator does **not** outperform the deterministic template on this pilot. The literal-check column is not a validity measure: two of four template test predictions and three of four filtered SFT test predictions pass it despite differing from the reference. Non-exactness alone is also not proof of semantic invalidity; those cases require human review. Among the four sampled SFT candidates for `flask_python_min_keep`, candidate 3 exactly matches the gold span, but the filter selects candidate 2 because it is the first to pass the weaker literal checks. Thus the filter can discard a demonstrably exact candidate.

The real-only MiniLM reranker trained for two epochs on 34 query-document pairs from 15 train revision groups. Its dev ranking did not improve over the pretrained cross-encoder. The fixed pools have only eight candidates per dev query and six per test query, with four and three independent revision groups respectively.

| Retrieval method | Dev top-1 / 5 | Test top-1 / 4 |
|---|---:|---:|
| BM25 | 1 | 2 |
| Pretrained cross-encoder | 4 | 3 |
| Real-only trained cross-encoder | 4 | 3 |
| Version BM25 / version cross-encoder | 5 | 4 |

Version filtering already saturates this small pool. The reranker experiment has **no generated training documents**, so it does not measure whether generated edits improve retrieval. A larger, harder fixed pool and an audited generated-data arm are needed for that claim.

## Concrete failure modes from the selected spans

1. **Missing information in the task specification.** The Flask `MAX_CONTENT_LENGTH` row gives the generator only `It can be set on a specific` as the new-value cue, but the target paragraph adds a 413 error, `None` behavior, WSGI stream conditions, and a view-specific override. The model cannot infer every target fact from this cue alone. A short output, `It can be set on a specific request.`, passes the current literal check while deleting most of the paragraph. This is a data-contract problem, not simply a model-capacity issue.
2. **Lost conditions.** The Node `Buffer.concat` gold distinguishes combined length *exceeding* versus being *less than* `totalLength`. The template retains only the exceeding branch; SFT says the remaining space is filled with zeros without stating the less-than condition. The current checks reject these dev outputs, but the example shows why conditional edits need explicit condition-level evaluation.
3. **Lost qualifiers or preserved facts.** The SFT HTTP span places the `rejectNonStandardBodyWrites` condition after a largely unconditional error statement. The Flask Python span omits the recommendation to use the latest Python version. The `SERVER_NAME` span omits why the setting enables extraction of a subdomain. Several sampled prompt-only outputs contain code fences or entire surrounding sections, violating the replacement-span format.
4. **Weak candidate selection.** On the Flask keep question, the sampled SFT set contains the exact target but the first-pass filter chooses a non-exact replacement with an extra heading. This is a direct selection failure even under the strict exact-match diagnostic.

## Research decision

Keep this pilot immutable and report the negative result. The test set has now been inspected, so future code or prompt changes evaluated on these four Flask rows are exploratory, not a fresh unbiased test. For the next study, first provide complete target-fact specifications and preservation constraints, then freeze new revision groups before training. Evaluate semantic fact coverage and document structure through independent human annotations; use exact match and literal checks only as supporting diagnostics. Add substantially more independent revisions before claiming cross-version or cross-project generalization. The retrieval experiment needs many same-version distractors and a separately audited generated-document training arm with the same model and training budget as its real-only control.
