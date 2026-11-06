"""Create fictional examples for pipeline testing only; never report as research data."""
import json
from pathlib import Path


def row(name, split, old, new, question, before, after):
    return {"id": name + "_" + ("change" if before != after else "keep"),
            "group": name, "split": split, "source_type": "synthetic_toy",
            "source_url": "FICTIONAL_PIPELINE_TEST", "question": question,
            "answer_before": before, "answer_after": after,
            "prefix": "Configuration notes. ", "old_span": f"The retry limit is {old}.",
            "new_span": f"The retry limit is {new}.",
            "suffix": " The timeout is 10 seconds.", "field": "retry limit",
            "old_value": old, "new_value": new,
            "protected_facts": ["The timeout is 10 seconds."], "edit_type": "scalar",
            "version_before": "v1", "version_after": "v2", "scope": name}


def build_rows():
    examples = []
    for name, split, old, new in [
        ("toy_a", "train", "3", "2"), ("toy_b", "train", "5", "4"),
        ("toy_c", "dev", "7", "6"), ("toy_d", "test", "9", "8")]:
        examples.append(row(name, split, old, new, "What is the retry limit?", old, new))
        examples.append(row(name, split, old, new, "What is the timeout?", "10 seconds", "10 seconds"))
    return examples


if __name__ == "__main__":
    path = Path(__file__).parent / "data" / "toy.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in build_rows()) + "\n", encoding="utf-8")
    print(path)
