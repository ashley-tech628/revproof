# Local CPU pilot, 2026-11-06

These files are reproducible deterministic baselines on the committed `data/real.jsonl` (SHA-256 `ae9862b22dd5cb7f7733a7d4df8631551de2c660a7c6eacb45ce2fc7119df54f`). They are **not** GPU model-training results. The generator table was produced with `evaluate_generator.py --template-only` and `summarize_pilot.py`; retrieval pools were built by `make_retrieval_pool.py` and evaluated by `retrieval_eval.py`.

| Split | Independent revision groups | Questions | Template exact | Template literal checks | BM25 top-1 | Version BM25 top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Dev | 4 | 5 | 2/5 | 2/5 | 1/5 | 5/5 |
| Test | 3 | 4 | 2/4 | 4/4 | 2/4 | 4/4 |

The test literal checker accepts incomplete edits, so literal pass rates cannot stand in for semantic validity. The pools have 8 candidates per dev query and 6 per test query, including same-version distractors. The tiny sample and selected candidates make the rates unsuitable for generalization. No real-data SFT, DPO, or trained reranker scores are in this folder.
