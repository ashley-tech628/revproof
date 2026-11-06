"""Check the GPU training stack before downloading any model weights."""
import importlib
import sys
from importlib.metadata import version


def main():
    torch = importlib.import_module("torch")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the DataHub 1 GPU environment and PyTorch install")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    supported = torch.cuda.get_arch_list()
    print(f"GPU: {torch.cuda.get_device_name(0)}; architecture {architecture}")
    print(f"PyTorch CUDA architectures: {supported}")
    if architecture not in supported:
        raise RuntimeError(f"This PyTorch build lacks {architecture}; use a CUDA 12.8+ build in a fresh .venv_blackwell")
    with torch.no_grad():
        result = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")
    torch.cuda.synchronize()
    print(f"PyTorch CUDA matrix multiply: OK; first value {result[0, 0].item()}")
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
