"""Check the GPU training stack before downloading any model weights."""
import importlib
import sys
from importlib.metadata import version


def main():
    torch = importlib.import_module("torch")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; built for CUDA {torch.version.cuda}")
    if torch.__version__.split("+")[0] != "2.2.1":
        raise RuntimeError("Expected the ECE285 DataHub torch 2.2.1 build; use a fresh .venv_datahub with requirements-datahub.txt")
    for package, module in (
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("trl", "trl"),
        ("datasets", "datasets"),
    ):
        importlib.import_module(module)
        print(f"{package}: {version(package)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the DataHub 1 GPU environment and PyTorch install")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # A tiny 4-bit layer exercises the QLoRA backend without fetching a model.
    from bitsandbytes.nn import Linear4bit

    layer = Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                       quant_type="nf4").to("cuda")
    with torch.no_grad():
        output = layer(torch.zeros(1, 64, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    print(f"4-bit CUDA operation: OK; output shape {tuple(output.shape)}")


if __name__ == "__main__":
    main()
