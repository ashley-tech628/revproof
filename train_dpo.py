"""Optional DPO stage, using only independently audited train preferences."""
import argparse
import json
from pathlib import Path


def read_pairs(path):
    pairs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not pairs:
        raise ValueError("empty preference dataset")
    if any(p.get("split") != "train" for p in pairs):
        raise ValueError("DPO preferences must be train-only")
    for pair in pairs:
        if not all(pair.get(k) for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"incomplete pair: {pair.get('id')}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="output of prepare_preferences.py")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit DPO requires a CUDA GPU")
    pairs = read_pairs(args.pairs)
    dataset = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bf16 = torch.cuda.is_bf16_supported()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                               device_map="auto")
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = DPOConfig(output_dir=str(output), num_train_epochs=args.epochs,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8,
                       learning_rate=args.lr, beta=args.beta, max_length=args.max_length,
                       logging_steps=1, save_strategy="no", report_to="none",
                       bf16=bf16, fp16=not bf16, optim="paged_adamw_8bit", seed=args.seed)
    # With ref_model=None, TRL freezes the initial policy as the DPO reference.
    trainer = DPOTrainer(model=model, ref_model=None, args=config,
                         train_dataset=dataset, processing_class=tok)
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tok.save_pretrained(str(output / "adapter"))
    (output / "run.json").write_text(json.dumps({
        "model": args.model, "sft_adapter": args.sft_adapter, "pairs": args.pairs,
        "pair_count": len(pairs), "revision_groups": len({p["group"] for p in pairs}),
        "metrics": result.metrics, "beta": args.beta, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
