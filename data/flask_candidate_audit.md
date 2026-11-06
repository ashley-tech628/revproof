# Flask 3.0.3 → 3.1.0 documentation audit (pilot)

Source: [official Git comparison](https://github.com/pallets/flask/compare/3.0.3...3.1.0) and [3.1.0 release note](https://github.com/pallets/flask/releases/tag/3.1.0). `collect_candidates.py` found 13 changed hunks in `docs/`. I checked the old and new tagged lines. This is a small pilot audit, not a representative sample or model result.

| Candidate | File | Decision | Reason |
|---:|---|---|---|
| 0 | `config.rst` | Defer | `SERVER_NAME` condition is clarified; semantic change needs code or release-note verification. |
| 1 | `config.rst` | Defer | `url_for` context wording is clarified; a changed answer is not established. |
| 2 | `config.rst` | Defer | `MAX_CONTENT_LENGTH` behavior is more detailed and conditional; needs careful field and scope annotation. |
| 3 | `deploying/asgi.rst` | Exclude | Hypercorn repository URL moved; not a versioned Flask behavior fact. |
| 4 | `deploying/waitress.rst` | Exclude | Typo correction. |
| 5 | `extensiondev.rst` | Exclude as duplicate | Same Python support threshold as candidate 7. |
| 6 | `index.rst` | Exclude | Introductory description added without a paired answer-changing fact. |
| 7 | `installation.rst` | Include | Minimum supported Python changed 3.8 → 3.9, also stated in the 3.1.0 release note. |
| 8–10 | `quickstart.rst` | Exclude | Example template variable renamed `name` → `person`; no general API behavior change. |
| 11 | `testing.rst` | Defer | Example adds `follow_redirects=True`; could be a useful code-edit task, but the documentation hunk alone does not establish a changed default. |
| 12 | `web-security.rst` | Defer | Large rewritten section introduces several facts; requires smaller alignment and independent checks. |

Candidate 7 yields two *questions* about one real edit: a `change` question about the minimum supported version and a `keep` question about the recommendation to use the latest Python. Both are in `flask_pilot_audited.jsonl` with the **same group and dev split**. The template baseline exactly matches the two target spans; the two questions are one revision, not two independent data points. This trivial replacement is a strong negative control for the hypothesis that learning is useful.

## Prompt-only pilot and verifier audit (2026-11-06)

A user-run DataHub evaluation of `Qwen/Qwen2.5-1.5B-Instruct` without an adapter generated the following replacement spans (greedy decoding, one candidate per question):

| Question | Generated span | Exact target? | Manual judgment |
|---|---|---|---|
| Minimum supported Python (`change`) | `Python Version` | No | Invalid: omits the new supported version and the recommendation. |
| Recommended Python (`keep`) | `Python Version: 3.9` | No | Invalid: drops the recommendation to use the latest version and does not state the support relationship. |

The template was exact on both. The initial string checks accepted the `keep` output because it contained `3.9` and the example had no annotated protected facts. This is a **false acceptance**, not a valid edit. We subsequently annotated `We recommend using the latest version of Python.` as a protected fact in both records. Under that updated annotation, both observed model outputs fail the literal preservation check. This correction does not turn the checker into a semantic verifier: it still cannot establish whether an arbitrary statement about support, scope, or question answer is true. The original model predictions remain unchanged; summary rates computed before and after the annotation must be labeled with the annotation version.

This is one revision group with two correlated questions and no real-data fine-tuning. It illustrates a failure mode and a verifier limitation, not an estimate of population-level performance or evidence that learning is worse than templates generally. The raw DataHub prediction JSONL has not yet been archived in this repository; the strings above were transcribed from the user's terminal output.

The main bottleneck is fact-level annotation, not diff extraction. A wider FastAPI comparison (0.115.0 → 0.116.0) produced 780 English-documentation hunks, many of which change code-snippet markup rather than behavior. Raw hunk count must not be reported as training-set size. The next audit should prioritize conditional behavior and cross-sentence consistency and reserve entire release transitions for dev/test.
