# Running RevProof on UCSD DataHub / DSMLP

The ECE285 course notice shows `ECE285_FA26_002` with a `scipy-ml-notebook` **1 GPU** option. Log in at [datahub.ucsd.edu](https://datahub.ucsd.edu), select that course and GPU environment, then open **New → Terminal**. The exact GPU model and free memory are not specified in the notice; check them in your session.

## 1. Inspect the actual runtime

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('torch:', torch.__version__)"
```

Record the GPU model, total memory, and Python/PyTorch versions in your experiment notes. If CUDA is false in the selected GPU environment, resolve that before installing models.

## 2. Upload and install on the RTX PRO 6000 Blackwell

Use Jupyter's file browser to upload the small `revproof.zip` from your computer into your **course home directory**. In Terminal, from the directory where it was uploaded:

```bash
unzip -o revproof.zip
cd revproof
python -m venv .venv_blackwell
.venv_blackwell/bin/python -m pip install -r requirements-datahub.txt
.venv_blackwell/bin/python -m unittest discover -p test_pipeline.py -v
.venv_blackwell/bin/python check_datahub_runtime.py
```

[UCSD's DataHub FAQ](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=b096442e9718f650992771611153afed) recommends a virtual environment for custom packages. This GPU reports compute capability `sm_120`; the course image's PyTorch 2.2.1+cu121 is too old to run its kernels. Create `.venv_blackwell` **without** `--system-site-packages` so the old PyTorch, torchvision and torchaudio cannot leak into the new environment. Leave earlier `.venv` and `.venv_datahub` directories alone. The unit tests do not exercise CUDA; proceed only after `check_datahub_runtime.py` prints both CUDA operations as OK. If pip cannot resolve packages or the preflight fails, share its exact final error and the output of `nvidia-smi`. Keep the project and `runs/` in the course home directory so outputs survive notebook container restarts; check your storage quota before downloading large model weights.

## 3. GPU smoke run (fictional data)

```bash
.venv_blackwell/bin/python make_toy_data.py
.venv_blackwell/bin/python train_generator.py --data data/toy.jsonl --output runs/toy_sft --epochs 1 --allow-synthetic-smoke
.venv_blackwell/bin/python evaluate_generator.py --data data/toy.jsonl --split dev --adapter runs/toy_sft/adapter --output runs/toy_sft_dev.jsonl
```

This tests 4-bit model loading, training, checkpoint saving, and inference. The toy data are fictional; **never report toy scores as research results**.

## 4. Real pilot experiment

The current zip includes `data/real.jsonl`: 15 Node.js train revision groups, 4 Node.js dev groups from a held-out release transition, and 3 Flask test groups. Source excerpts were checked against old/new official tags; the sample is still small. Upload the **latest zip with a distinct filename**, unzip with `-o`, and confirm `data/real.jsonl` is present. Then run:

```bash
.venv_blackwell/bin/python -m unittest discover -p test_pipeline.py -v
.venv_blackwell/bin/python run_datahub_pilot.py --stage all
.venv_blackwell/bin/python run_retrieval_pilot.py
```

The generator runner performs a preflight, template and prompt-only baselines, real-data QLoRA SFT, greedy dev/test evaluation, and equal-budget four-candidate filtered comparisons. It writes predictions, summaries, training run metadata, a dataset hash, and `runs/pilot/generator_table.csv`. The retrieval runner trains the real-only MiniLM cross-encoder and evaluates the same fixed candidate pools for all methods. These pilots have few held-out groups; assess generated edits manually before making semantic claims. DPO requires completed train-only human preference labels and is not run automatically.

## 5. Revised fact-card pilot with a fresh test transition

The first GPU pilot is archived in `results/datahub_223b776/`. Its complex targets were sometimes impossible to infer from the short input cue. The [v2 protocol](data/v2_protocol.md) adds complete, manually reviewed change facts and reserves a previously untested Node release transition. After uploading the latest project zip, use the existing `.venv_blackwell` environment:

```bash
cd ~/revproof
.venv_blackwell/bin/python -m unittest discover -p test_pipeline.py -v
.venv_blackwell/bin/python run_datahub_pilot.py --data data/real_v2.jsonl --output runs/v2 --prompt-version v2 --stage all
```

Keep `runs/v2` separate from `runs/pilot`; the two tasks have different inputs and held-out groups. The runner produces an unlabelled audit CSV under `runs/v2/diagnostics/`. Inspect selected outputs for all fact-card requirements before making semantic claims. The new test should be evaluated only once under this frozen configuration.

The Colab notebook contains `google.colab.files` and `/content/...` paths, so use these Terminal commands on DataHub. If DataHub is congested or its assigned GPU cannot fit your selected model, upload the same `revproof.zip` and notebook to Colab and run the Colab workflow there.

UCSD notes that course workspaces provide storage but busy cluster periods can lead to eviction notices. Colab also has variable GPU availability and session limits. Keep checkpoints and copied results outside a temporary runtime.

## 6. Exploratory sparse-editor experiment

The new [architecture plan](research/architecture_decision.md) tests a single-window edit-script objective. It predicts a unique source anchor and replacement; the program copies all text outside that anchor exactly. The v2 test has already been inspected, so `run_sparse_pilot.py` evaluates **dev only**. After uploading the latest committed project zip and using the working Blackwell environment:

```bash
cd ~/revproof
.venv_blackwell/bin/python -m unittest discover -p 'test_*.py' -v
.venv_blackwell/bin/python run_sparse_pilot.py --backend seq2seq --model google/flan-t5-large --output runs/sparse_flan_large --stage all
.venv_blackwell/bin/python run_sparse_pilot.py --backend causal --model Qwen/Qwen2.5-1.5B-Instruct --output runs/sparse_qwen_1p5b --stage all
```

Each run writes `protocol.json`, `commands.jsonl`, the LoRA adapter, raw dev predictions, summaries, and `generator_table.json`. Run the FLAN condition first. Do not infer semantic correctness from exact match or literal checks; audit the actual revised text. If you use Colab Pro instead, follow `notebooks/revproof_sparse_colab.ipynb` and inspect the actual GPU allocation before training. For the current span-editor experiment, use `research/datahub_span_editor.md`.

## 7. Oracle-location infilling diagnostic

The first sparse-editor GPU pilot failed to reconstruct any of the nine dev paragraphs. See [the per-case analysis](results/sparse_15c6e2f/ANALYSIS.md). This next experiment exposes the **gold edit location**, derived from the target paragraph, and asks FLAN-T5-large to generate only replacement text. It separates replacement generation from location prediction and JSON formatting. Its scores are an **oracle upper bound with target-location leakage**, not deployable accuracy. It uses dev only; the previously inspected test is excluded.

Upload the new project zip under a distinct filename, then from the DataHub Terminal:

```bash
cd ~
unzip -o revproof_oracle.zip
cd ~/revproof
.venv_blackwell/bin/python -m unittest discover -p 'test_*.py' -v
.venv_blackwell/bin/python run_oracle_infill_pilot.py --output runs/oracle_flan_large --stage all
```

This writes `runs/oracle_flan_large/protocol.json`, prompt-only and LoRA dev predictions and summaries, the adapter, and `generator_table.json`. The runner records the dataset hash and refuses to reuse an output directory with a different protocol. Compare the two dev predictions per case; teacher-forced loss alone does not demonstrate successful generation. Keep this table separate from the deployable sparse-editor comparison.

The [FLAN oracle diagnostic](results/oracle_flan_8007a70/ANALYSIS.md) obtained 0/9 exact dev replacements despite the gold location; 8/9 prompt and LoRA strings were identical. To check whether this is specific to FLAN-T5 on the same small dataset, run the Qwen causal condition with the **newer project zip** and a separate directory:

**Research update:** [The literature and failure analysis](research/evidence_grounded_revision_reset.md) now recommends pausing this Qwen run until the training positive control, evidence provenance, and evaluation set are repaired. The command remains here only as a reproducible optional diagnostic.

```bash
cd ~/revproof
.venv_blackwell/bin/python -m unittest discover -p 'test_*.py' -v
.venv_blackwell/bin/python run_oracle_infill_pilot.py --backend causal --model Qwen/Qwen2.5-1.5B-Instruct --output runs/oracle_qwen_1p5b --stage all
```

This Qwen condition still receives target-derived edit locations, so its dev accuracy is diagnostic only. Do not run it on the already inspected v2 test.

To return a compact archive for analysis:

```bash
cd ~/revproof
tar -czf ~/revproof_oracle_results.tgz runs/oracle_flan_large/protocol.json runs/oracle_flan_large/commands.jsonl runs/oracle_flan_large/generator_table.json runs/oracle_flan_large/prompt_dev.jsonl runs/oracle_flan_large/prompt_dev.summary.json runs/oracle_flan_large/sft_dev.jsonl runs/oracle_flan_large/sft_dev.summary.json runs/oracle_flan_large/sft/run.json
```
