"""Train a fixed cross-encoder reranker for downstream data-utility comparison."""
import argparse
import json
from pathlib import Path


def read_pairs(path, split):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(r.get("split") != split for r in rows):
        raise ValueError(f"{path}: expected nonempty {split} rows only")
    if any(r.get("label") not in (0, 1) for r in rows):
        raise ValueError("labels must be binary")
    return rows


class PairDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        for row in rows:
            encoded = tokenizer(row["query"], row["document"], truncation=True,
                                max_length=max_length, add_special_tokens=True)
            self.items.append({**encoded, "labels": float(row["label"])})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    train = read_pairs(args.train, "train")
    dev = read_pairs(args.dev, "dev")
    if {r["group"] for r in train} & {r["group"] for r in dev}:
        raise ValueError("same revision group in train and dev")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    class BCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits.view(-1), labels)
            return (loss, outputs) if return_outputs else loss

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", seed=args.seed,
        remove_unused_columns=False)
    collator = DataCollatorWithPadding(tok)
    trainer = BCETrainer(model=model, args=training_args,
                         train_dataset=PairDataset(train, tok, args.max_length),
                         eval_dataset=PairDataset(dev, tok, args.max_length),
                         data_collator=collator, processing_class=tok)
    result = trainer.train()
    dev_metrics = trainer.evaluate()
    trainer.save_model(str(output / "model"))
    tok.save_pretrained(str(output / "model"))
    (output / "run.json").write_text(json.dumps({
        "model": args.model, "train_count": len(train), "dev_count": len(dev),
        "train_groups": len({r["group"] for r in train}),
        "dev_groups": len({r["group"] for r in dev}),
        "train_sources": sorted({r["source"] for r in train}),
        "train_metrics": result.metrics, "dev_metrics": dev_metrics, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(json.dumps(dev_metrics, indent=2))


if __name__ == "__main__":
    main()
