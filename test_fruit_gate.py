import unittest

from evaluate_fruit_gate import addition_f1, lcs_f1, score, summarize


class FruitGateTest(unittest.TestCase):
    def test_additions_reward_supported_update(self):
        self.assertEqual(addition_f1("A old fact", "A new fact", "A new fact"), 1)
        self.assertEqual(addition_f1("A old fact", "A old fact", "A new fact"), 0)
        self.assertEqual(addition_f1("A old fact", "A unrelated fact", "A new fact"), 0)

    def test_sequence_overlap_and_invalid_output(self):
        self.assertEqual(lcs_f1("a b", "a b"), 1)
        self.assertEqual(lcs_f1("", "a b"), 0)
        self.assertFalse(score("old", None, "new")["exact"])

    def test_summary_uses_paired_examples(self):
        rows = [{"copy": {"scores": score("old", "old", "new"), "error": None}},
                {"copy": {"scores": score("same", "same", "same"), "error": None}}]
        self.assertEqual(summarize(rows, "copy")["exact"], 1)
        self.assertEqual(summarize(rows, "copy")["n"], 2)


if __name__ == "__main__":
    unittest.main()
