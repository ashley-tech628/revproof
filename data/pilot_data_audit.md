# Real revision pilot: construction and limits

The generator pilot contains 26 question rows from 22 local documentation revisions. Fifteen groups from official [Node.js tagged API docs](https://github.com/nodejs/node/tree/main/doc/api) are train; four groups from a later Node.js release transition are dev; three groups from [Flask 3.0.3 to 3.1.0](https://github.com/pallets/flask/compare/3.0.3...3.1.0) are test. Multiple questions about one edit stay in the same group. The Flask min-Python group contributes a `change` and a `keep` question, but remains one test revision.

`mine_node_revisions.py` downloaded old and new `doc/api/*.md` files by immutable tag and aligned nearby prose paragraphs. Of 348 unreviewed candidates across 11 API modules and seven major-version transitions, 20 were selected for manual fact/condition annotation. The selected old/new source excerpts and SHA-256 hashes are in `node_selected_sources.jsonl`. `verify_pilot_sources.py` checks each selected span and hash against the official tag URL. Two additional Flask `config.rst` hunks were selected from a Git diff and checked against both official tags. The [Flask 3.1.0 release notes](https://flask.palletsprojects.com/en/stable/changes/) corroborate the new per-request `MAX_CONTENT_LENGTH` and `SERVER_NAME` behavior. All 22 selected source changes passed the verifier on 2026-11-06.

The dataset builder adds question-specific old/new answers, a structured fact cue, edit type, and exact facts to preserve. It copies only one marked paragraph or hunk into the task context, with a document heading; line wraps are normalized. For additive edits, `allow_old_value_in_new_span` is explicitly annotated because the old fact remains true. `read_records` rejects any gold span that fails the row's literal checks. A source-linked excerpt is evidence of what the tagged documentation says; no claim is made that every API's runtime behavior was independently tested.

The predefined split is:

| Split | Source transitions | Groups | Question rows |
|---|---|---:|---:|
| Train | Node v16→v17, v17→v18, v19→v20 | 15 | 17 |
| Dev | Node v21→v22 | 4 | 5 |
| Test | Flask 3.0.3→3.1.0 | 3 | 4 |

This is a **pilot** with only four dev and three test groups. Training, prompt selection, and DPO labels must not use test outputs. The simple replacement template is exact on 2/5 dev rows and 2/4 test rows. It passes literal checks on all four test rows, but at least one incomplete Flask edit also passes; literal acceptance is not semantic validity. Evaluation must report question counts, independent revision-group counts, and a manually reviewed error table. A final study needs more release transitions, more document families, independently reviewed annotations, and confidence intervals by revision group.
