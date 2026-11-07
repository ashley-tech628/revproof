# Large-data plan for RevProof

Verified against the dataset publishers' pages on 2026-11-06. These sources serve different training roles; **do not concatenate them into one grounded evaluation set**. No external data or GPU training has been run by adding this plan.

**Local ingestion check:** The official CoDocBench `dataset/train.jsonl` was downloaded to workspace `work/external_data/codocbench_train.jsonl` (17,020,744 bytes; SHA-256 `9eef860afedcbf059a4f2e408cafbaa6c61ffbf4369e687a9b45ae80f7ac7c6d`). `prepare_codocbench.py` converted all 2,300 rows from 100 projects into `work/external_data/codocbench_train_silver.jsonl`; zero are marked audited. These raw and converted files are deliberately outside the Git repository. CoEdIT and FRUIT have not been downloaded or run.

| Source | Published scale and fields | RevProof role | Important limitation |
|---|---|---|---|
| [FRUIT-WIKI](https://github.com/google-research/language/tree/master/language/fruit) / [paper](https://aclanthology.org/2022.naacl-main.269/) | Over 170K distantly supervised old-article/evidence/update examples; publisher provides TFRecords and raw JSONL, plus 914 human-verified gold items. | Best available large-scale **evidence-grounded editing** pretraining and an external benchmark. | Wikipedia is not API documentation. Silver evidence support is heuristic; its gold evaluation must stay separate. Google hosts the dataset at `gs://gresearch/FRUIT/dataset`; inspect terms and storage before importing. |
| [CoDocBench](https://github.com/kunpai/codocbench) / [paper](https://arxiv.org/abs/2502.00519) | 4,573 coupled code/docstring revisions from 200 Python projects. Publisher's project split: 2,300 train, 2,273 test. Includes old/new docstrings and a code diff. | Closest **technical-domain** source. Code diff is candidate evidence for a docstring update. | Co-change does not prove the code diff supports every docstring statement. `prepare_codocbench.py` labels rows unaudited silver; review support before grounded training. Preserve the project's split; do not leak test projects. Verify repository-level licensing before redistributing text. |
| [CoEdIT](https://huggingface.co/datasets/grammarly/coedit) / [paper](https://aclanthology.org/2023.findings-emnlp.350/) | Official public train split has about 69.1K rows (`src`, `tgt`, `task`) and lists Apache-2.0. The paper reports 82K training instructions, while the public release is smaller. | General **editing instruction pretraining** or a pretrained editor baseline. `prepare_coedit.py` streams only train and marks all rows ungrounded. | Editing tasks are mostly grammar/style and do not provide independent new-version evidence. Never count these rows as factual-update supervision. |
| [FineEdit / InstrEditBench](https://huggingface.co/datasets/YimingZeng/FineEdit_bench) / [paper](https://aclanthology.org/2025.findings-emnlp.118/) | Paper reports over 30K structured editing tasks, including code and LaTeX; card lists CC-BY-4.0. | Candidate auxiliary pretraining/format evaluation after schema inspection. | Publisher's Hugging Face preview currently reports a dataset-generation error. Do not promise a working `load_dataset` adapter until files are inspected directly. It is not version-grounded evidence. |

## Immediate ingestion contract

`multispan_edit.py` derives multiple exact-copy edit blocks from old/new text and rejects overlapping plans. Its `validate_grounded_record` demands nonempty independently sourced evidence, matching target/evidence version, and an explicit support audit. Gold diffs are training labels, never inference inputs. `prepare_codocbench.py` and `prepare_coedit.py` deliberately do **not** set `support_audited=true`.

Run CoEdIT conversion only when ready to store its rows (it streams the public training split; no validation/test rows are used):

```bash
python prepare_coedit.py --output data/external/coedit_train.jsonl
```

For CoDocBench, obtain the publisher's repository and convert **only its train file**:

```bash
python prepare_codocbench.py --input /path/to/codocbench/dataset/train.jsonl --output data/external/codocbench_train_silver.jsonl
```

Do not train the grounded model on that silver file until evidence support and the source projects' terms have been checked. FRUIT's official instructions use `gsutil cp -R gs://gresearch/FRUIT/dataset ...`; processed records are TFRecords, raw records JSONL. A FRUIT adapter should be written after inspecting an actual downloaded shard and should preserve the publisher's train/dev/gold split. This avoids guessing its raw schema.

## Data hierarchy

1. **Editing pretraining:** CoEdIT train (instruction adherence), optionally structured FineEdit rows after validation.
2. **Grounded pretraining:** FRUIT train with source/evidence/target; keep its dev/gold evaluation separate from RevProof.
3. **Technical-domain adaptation:** CoDocBench train candidates after support filtering and audit. Code diffs, commit messages, and release notes are separate evidence types.
4. **Primary RevProof evaluation:** Newly curated official API docs from projects/release transitions absent from all adaptation data; at least a stratified human audit, group-level reporting, and no target-derived evidence. The current 4-row v2 test has been inspected and is exploratory only.

This hierarchy supports a claim about transfer from large grounded editing to versioned technical documentation. Dataset scale alone cannot fix missing evidence or an invalid evaluation contract.
