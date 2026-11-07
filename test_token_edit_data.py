import json
import unittest
from pathlib import Path

from token_edit_data import apply_token_edit, gold_token_edit


class TokenEditDataTest(unittest.TestCase):
    def test_lossless_alignment_on_distinct_edit_types(self):
        cases = [
            ("", "new"), ("old", ""), ("old", "old"),
            ("Return if on.", "Return True if on."),
            ("a b a b", "a c a b"),
            ("value: 3\n\told", "value: 2\n\tnew"),
            ("π = 3", "π = 3.14"),
            ("one two three", "three one two"),
        ]
        for source, target in cases:
            with self.subTest(source=source, target=target):
                labels = gold_token_edit(source, target)
                self.assertEqual(apply_token_edit(**labels), target)

    def test_reject_malformed_predictions(self):
        with self.assertRaises(ValueError):
            apply_token_edit(["a"], [True], [""])
        with self.assertRaises(TypeError):
            apply_token_edit(["a"], [1], ["", ""])

    def test_real_silver_rows_roundtrip_when_available(self):
        path = Path(__file__).resolve().parent.parent / "work" / "external_data" / "codocbench_train_silver.jsonl"
        if not path.exists():
            self.skipTest("local external CoDocBench data unavailable")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2300)
        for row in rows:
            labels = gold_token_edit(row["old_text"], row["new_text"])
            self.assertEqual(apply_token_edit(**labels), row["new_text"])


if __name__ == "__main__":
    unittest.main()
