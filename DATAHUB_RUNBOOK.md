# Running RevProof on UCSD DataHub / DSMLP

The ECE285 course notice shows `ECE285_FA26_002` with a `scipy-ml-notebook` **1 GPU** option. Log in at [datahub.ucsd.edu](https://datahub.ucsd.edu), select that course and GPU environment, then open **New → Terminal**. The exact GPU model and free memory are not specified in the notice; check them in your session.

## 1. Inspect the actual runtime

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('torch:', torch.__version__)"
```

Record the GPU model, total memory, and Python/PyTorch versions in your experiment notes. If CUDA is false in the selected GPU environment, resolve that before installing models.

## 2. Upload and install

Use Jupyter's file browser to upload the small `revproof.zip` from your computer into your **course home directory**. In Terminal, from the directory where it was uploaded:

```bash
unzip revproof.zip
cd revproof
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-colab.txt
.venv/bin/python -m unittest discover -p test_pipeline.py -v
```

[UCSD's DataHub FAQ](https://support.ucsd.edu/its?id=kb_article_view&sys_kb_id=b096442e9718f650992771611153afed) recommends a virtual environment for custom packages. `--system-site-packages` lets it see the course image's CUDA-enabled PyTorch if present. If dependency installation changes or breaks PyTorch, stop and inspect the environment instead of continuing a GPU run blindly. Keep the project and `runs/` in the course home directory so outputs survive notebook container restarts; check your storage quota before downloading large model weights.

## 3. GPU smoke run (fictional data)

```bash
.venv/bin/python make_toy_data.py
.venv/bin/python train_generator.py --data data/toy.jsonl --output runs/toy_sft --epochs 1 --allow-synthetic-smoke
.venv/bin/python evaluate_generator.py --data data/toy.jsonl --split dev --adapter runs/toy_sft/adapter --output runs/toy_sft_dev.jsonl
```

This tests 4-bit model loading, training, checkpoint saving, and inference. The toy data are fictional; **never report toy scores as research results**.

## 4. Real experiment

Audit versioned documents and upload `data/real.jsonl` with disjoint train/dev/test revision groups. The included Flask file has just one dev revision group and cannot train a research model. Then run the commands in [README.md](README.md): SFT; template, prompt-only, and SFT dev evaluation; optional train-only audit and DPO; and fixed-candidate retrieval baselines. Keep the same backbone for all learned generator conditions. Save the JSONL predictions, `.summary.json`, `run.json`, and human annotation CSV with your report materials.

The Colab notebook contains `google.colab.files` and `/content/...` paths, so use these Terminal commands on DataHub. If DataHub is congested or its assigned GPU cannot fit your selected model, upload the same `revproof.zip` and notebook to Colab and run the Colab workflow there.

UCSD notes that course workspaces provide storage but busy cluster periods can lead to eviction notices. Colab also has variable GPU availability and session limits. Keep checkpoints and copied results outside a temporary runtime.
