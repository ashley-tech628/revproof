"""Contract tests for the new sparse-generation objective and reconstruction."""
import json
import unittest
from pathlib import Path

from evaluate_sparse_editor import assess, summarize
from revproof_data import read_records
from sparse_edit import apply_script, edit_prompt, gold_script, parse_script, script_text
from train_sparse_editor import encode_row


class TinyTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = [ord(x) % 95 + 1 for x in text]
            if add_special_tokens:
                input_ids += [self.eos_token_id]
        return Encoded()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<user>" + messages[0]["content"] + "<assistant>"


class SparseEditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_records(Path(__file__).parent / "data" / "real_v2.jsonl")

    def test_all_real_gold_scripts_reconstruct_exactly(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                script = gold_script(row["old_span"], row["new_span"])
                self.assertEqual(apply_script(row["old_span"], script), row["new_span"])
                self.assertEqual(parse_script(script_text(script)), script)
                self.assertNotIn(row["new_span"], edit_prompt(row))

    def test_insertion_deletion_and_repeated_anchor(self):
        for old, new in (("The type is Error.", "The type is TypeError."),
                         ("A B C", "A C"), ("A C", "A B C"),
                         ("x x x", "x y x")):
            self.assertEqual(apply_script(old, gold_script(old, new)), new)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            apply_script("x x", {"old": "x", "new": "y"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_script("x", {"old": "x", "new": "y", "start": 0})

    def test_invalid_generation_is_visible(self):
        row = self.rows[0]
        bad = assess(row, "```json\n{}\n```")
        self.assertFalse(bad["valid_json"])
        self.assertFalse(bad["applicable"])
        self.assertIsNone(bad["prediction"])
        missing = assess(row, json.dumps({"old": "NEVER_IN_SOURCE", "new": "something"}))
        self.assertTrue(missing["valid_json"])
        self.assertFalse(missing["applicable"])
        self.assertFalse(missing["exact_span"])

    def test_gold_assessment_and_summary(self):
        row = self.rows[0]
        result = assess(row, script_text(gold_script(row["old_span"], row["new_span"])))
        self.assertTrue(result["exact_span"])
        self.assertTrue(result["passes_string_checks"])
        item = {**result, "group": row["group"], "edit_type": row["edit_type"],
                "z": row["z"], "any_candidate_passes_string_checks": True}
        self.assertEqual(summarize([item])["all_questions_exact_rate"], 1.0)

    def test_loss_masks_only_causal_prompt(self):
        tokenizer = TinyTokenizer()
        row = self.rows[0]
        causal = encode_row(tokenizer, row, "causal", 10000, 10000)
        self.assertTrue(all(x == -100 for x in causal["labels"][:causal["source_tokens"]]))
        self.assertTrue(all(x != -100 for x in causal["labels"][causal["source_tokens"]:]))
        seq = encode_row(tokenizer, row, "seq2seq", 10000, 10000)
        self.assertEqual(len(seq["labels"]), seq["target_tokens"])
        self.assertEqual(len(seq["input_ids"]), seq["source_tokens"])
        with self.assertRaisesRegex(ValueError, "exceed limits"):
            encode_row(tokenizer, row, "seq2seq", 5, 5)


if __name__ == "__main__":
    unittest.main()
