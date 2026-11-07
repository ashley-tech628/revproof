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

[UCSD's DataHub FAQ](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=b096442e9718f650992771611153afed) recommends a virtual environment for custom packages. This GPU reports compute capability `sm_120`; the course image's PyTorch 2.2.1+cu121 is too old to run its kernels. Create `.venv_blackwell` **without** `--system-site-packages` so the old PyTorch, torchvision and torchaudio cannot leak into the new environment. Leave earlier `.venv` and `.venv_datahub` directories alone. The eight unit tests do not exercise CUDA; proceed only after `check_datahub_runtime.py` prints both CUDA operations as OK. If pip cannot resolve packages or the preflight fails, share its exact final error and the output of `nvidia-smi`. Keep the project and `runs/` in the course home directory so outputs survive notebook container restarts; check your storage quota before downloading large model weights.

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

The Colab notebook contains `google.colab.files` and `/content/...` paths, so use these Terminal commands on DataHub. If DataHub is congested or its assigned GPU cannot fit your selected model, upload the same `revproof.zip` and notebook to Colab and run the Colab workflow there.

UCSD notes that course workspaces provide storage but busy cluster periods can lead to eviction notices. Colab also has variable GPU availability and session limits. Keep checkpoints and copied results outside a temporary runtime.
