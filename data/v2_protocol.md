# Fact-card revision pilot: frozen protocol before new GPU test

The first real-data run (commit `223b776`) revealed that several target paragraphs contained facts absent from the short `new_value` cue. V2 keeps the old dataset untouched and adds a manually written `change_spec` with all intended new facts. The input still does not contain the exact gold replacement span. This evaluates **controlled editing from a complete change request**, not prediction of undocumented release changes. The cards were written after inspecting the v1 failures, so v1/v2 scores are not a clean ablation of prompt wording alone.

`data/real_v2.jsonl` SHA-256: `2f1c9f060639d6b2f72d12d1bd29589c590032e543c1178e0a48e6c352c3d796`.

| Split | Revision groups | Question rows | Source |
|---|---:|---:|---|
| Train | 15 | 17 | Node v16→v17, v17→v18, v19→v20 |
| Dev | 7 | 9 | Node v21→v22; previously inspected Flask 3.0.3→3.1.0 |
| Fresh test | 4 | 4 | Node v22.14.0→v23.11.0, previously untested in this project |

The four test edits cover a byte-oriented file stream, a conditional symlink type, an authentication-tag rule, and a TypeScript feature default. They were selected and their fact cards frozen **before** any v2 generation was run on them. The old/new excerpts and SHA-256 hashes are in `v2_new_test_sources.jsonl`. `verify_pilot_sources.py --include-v2` verifies all 26 selected source changes against old and new official tagged files. These are documentation facts; runtime behavior was not independently tested. The file-stream heading is neutralized in the constructed context because the upstream heading signature changed outside the marked paragraph.

The predeclared generator comparison uses the same Qwen2.5-1.5B-Instruct backbone, seed 42, three SFT epochs, gradient accumulation 2, maximum length 1024, and four candidates for the filtered arms. Methods are deterministic template, prompt-only, QLoRA SFT, and equal-budget filtered prompt/SFT. The model prompt includes the full fact card and old span. All methods use the same frozen rows; the template uses only the `old_value`/`new_value` fields. The output directory must be new, such as `runs/v2`. No hyperparameter or method selection uses the fresh test scores.

Exact replacement-span match is the fully automated reference metric. Report question-level and independent-group counts. Literal checks and exact-in-candidates are diagnostics only. The primary semantic analysis requires a separate human audit of factual coverage, conditions, retention of still-correct information, document structure, and unsupported claims. The audit must include both literal-accepted and rejected predictions and leave uncertain cases explicit. With only four independent test groups and 17 training rows, this remains a pipeline pilot, not evidence of general performance.

Run on the existing Blackwell virtual environment:

```bash
.venv_blackwell/bin/python run_datahub_pilot.py --data data/real_v2.jsonl --output runs/v2 --prompt-version v2 --stage all
```

The run automatically creates `runs/v2/generator_table.csv`, a per-example diagnostic JSON, and an unlabelled CSV for human semantic audit. Do not run DPO until independent train-only candidate preferences are annotated. The v1 reranker pool saturates under version filtering; downstream generated-data utility needs a new, larger retrieval corpus and an audited generated-document arm.
