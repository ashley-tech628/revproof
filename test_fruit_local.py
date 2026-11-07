import unittest

from evaluate_fruit_local import score
from fruit_data import normalize_fruit
from fruit_local import extract, prompt, reconstruct


def converted(source, target, split="train"):
    inputs = source + " [CONTEXT] (0) Verified update in an external source."
    old, gold = normalize_fruit(inputs, target)
    return {"id": 1, "split": split, "source_dataset": "FRUIT-WIKI",
            "input": inputs, "target": target, "source": old, "gold": gold}


class FruitLocalTest(unittest.TestCase):
    def test_middle_replacement_roundtrip_and_no_target_in_prompt(self):
        row = converted("[0] Before. [1] Old value. [2] After.",
                        "[0] (0) New value. [2]")
        local, reason = extract(row)
        self.assertIsNone(reason)
        self.assertEqual(local["edit_index"], 1)
        self.assertEqual(local["old"], "Old value.")
        self.assertEqual(local["replacement"], "New value.")
        self.assertEqual(reconstruct(local, local["replacement"]), row["gold"])
        self.assertNotIn("New value.", prompt(local))
        self.assertIn("Verified update", prompt(local))

    def test_first_and_last_replacement(self):
        cases = [("New first. [1] [2]", 0),
                 ("[0] [1] New last.", 2)]
        for target, index in cases:
            with self.subTest(index=index):
                row = converted("[0] First. [1] Middle. [2] Last.", target)
                local, reason = extract(row)
                self.assertIsNone(reason)
                self.assertEqual(local["edit_index"], index)
                self.assertEqual(reconstruct(local, local["replacement"]), row["gold"])

    def test_rejects_multiple_edits_and_duplicate_copy(self):
        source = "[0] First. [1] Middle. [2] Last."
        for target in ("New first. [1] New last.",
                       "[0] New middle. [0] [2]",
                       "[0] New middle. [2] Added after."):
            with self.subTest(target=target):
                local, reason = extract(converted(source, target))
                self.assertIsNone(local)
                self.assertIsNotNone(reason)

    def test_rejects_stored_gold_mismatch_and_wrong_dataset(self):
        row = converted("[0] Old. [1] Keep.", "New. [1]")
        row["gold"] = "wrong"
        self.assertEqual(extract(row)[1], "stored_reconstruction_mismatch")
        row["source_dataset"] = "other"
        with self.assertRaises(ValueError):
            extract(row)

    def test_scores_only_replacement_and_deterministic_context(self):
        local, _ = extract(converted("[0] Before. [1] Old. [2] After.",
                                      "[0] New. [2]"))
        self.assertEqual(score(local, "New.")["full_exact"], True)
        self.assertEqual(score(local, "Old.")["replacement_exact"], False)
        self.assertEqual(reconstruct(local, "Different."),
                         "Before. Different. After.")


if __name__ == "__main__":
    unittest.main()
