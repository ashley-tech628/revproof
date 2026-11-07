"""Contract checks for copy preservation, gold extraction and locator scoring."""

import unittest

from fruit_span import apply_span, extract_span, prompt
from span_editor.evaluate import make_prediction
from span_editor.locator import best_span, candidate_sentences, question


def sample():
    old = "José joined Example FC in 2018."
    new = "José joined Example FC in 2020."
    sentences = ["The club was founded in 1901.", old, "It plays in the city league."]
    return {"id": 1, "split": "train", "source_dataset": "FRUIT-WIKI-local-oracle",
            "edit_index": 1, "source_sentences": sentences, "old": old,
            "replacement": new, "previous": sentences[0], "next": sentences[2],
            "evidence": "(0) Example FC 2020 roster: José joined in 2020.",
            "source": " ".join(sentences),
            "gold": " ".join([sentences[0], new, sentences[2]])}


class SpanEditorTest(unittest.TestCase):
    def test_short_span_roundtrip_and_prompt_has_no_gold(self):
        row, reason = extract_span(sample())
        self.assertIsNone(reason)
        self.assertEqual(row["old_span"], "2018")
        self.assertEqual(row["new_span"], "2020")
        self.assertEqual(apply_span(row, row["new_span"]), row["replacement"])
        self.assertIn("José joined", prompt(row))
        self.assertNotIn("gold", prompt(row).casefold())

    def test_copy_constrained_reconstruction(self):
        row, _ = extract_span(sample())
        pred = make_prediction(row, 1, row["span_start"], row["span_end"], "2020", 80)
        self.assertTrue(pred["scores"]["full_exact"])
        self.assertTrue(pred["full"].startswith("The club was founded"))
        self.assertIn("José joined", pred["full"])
        copy = make_prediction(row, None, None, None, "", 80)
        self.assertFalse(copy["scores"]["full_exact"])
        self.assertEqual(copy["full"], row["source"])

    def test_wrong_location_does_not_get_oracle_credit(self):
        row, _ = extract_span(sample())
        pred = make_prediction(row, 0, 4, 8, "2020", 80)
        self.assertFalse(pred["scores"]["replacement_exact"])
        self.assertFalse(pred["scores"]["full_exact"])

    def test_locator_uses_source_and_evidence_only(self):
        row, _ = extract_span(sample())
        self.assertEqual(candidate_sentences(row, 1)[0], row["edit_index"])
        self.assertNotIn("2018", question(row))
        self.assertNotIn(row["gold"], question(row))
        offsets = [(0, 0), (0, 4), (5, 9), (10, 14)]
        score, a, b = best_span([0, -1, 4, -1], [0, -1, 5, -1],
                                offsets, [1, 2, 3])
        self.assertEqual((a, b), (5, 9))
        self.assertEqual(score, 9)


if __name__ == "__main__":
    unittest.main()
