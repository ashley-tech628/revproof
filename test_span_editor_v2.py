"""Regression checks for reference parsing and label-free v2 inference."""

import unittest

from span_editor.evaluate_v2 import (choose_candidate, inference_view,
                                     phrase_checks)
from span_editor.compare_v2 import compare_rows
from span_editor.locator import top_spans
from span_editor.train_infiller_v2 import referenced_view
from span_editor.train_selector import training_pairs
from span_editor.v2_data import edit_references, extract, split_evidence


def raw_example():
    old = "José joined Example FC in 2018."
    new = "José joined Example FC in 2020."
    second = "The club was founded in 1901."
    return {"id": 1, "split": "train", "source_dataset": "FRUIT-WIKI",
            "input": (f"[0] {old} [1] {second} [CONTEXT] "
                      "(0) Another player joined Rival FC. "
                      "(1) Example FC roster: José joined in 2020."),
            "target": f"(1) {new} [1]",
            "source": f"{old} {second}", "gold": f"{new} {second}"}


class SpanEditorV2Test(unittest.TestCase):
    def test_reference_and_evidence_parsing(self):
        raw = raw_example()
        row, reason = extract(raw)
        self.assertIsNone(reason)
        self.assertEqual(row["reference_ids"], [1])
        self.assertEqual(row["new_span"], "2020")
        self.assertEqual(len(row["evidence_items"]), 2)
        self.assertEqual(edit_references(raw["target"], 0,
                                         "José joined Example FC in 2020."), [1])
        self.assertEqual(split_evidence("(0) A (2020) title (2) C"),
                         ["A (2020) title", "", "C"])

    def test_selector_positive_and_hard_negatives(self):
        row, _ = extract(raw_example())
        pairs = training_pairs(row, negatives=1, wrong_sentences=1)
        self.assertIn((0, 1, 1), pairs)
        self.assertIn((0, 0, 0), pairs)
        self.assertIn((1, 1, 0), pairs)

    def test_inference_view_excludes_gold_fields_and_references(self):
        row, _ = extract(raw_example())
        view = inference_view(row)
        for key in ("gold", "replacement", "edit_index", "span_start",
                    "span_end", "new_span", "reference_ids"):
            self.assertNotIn(key, view)
        self.assertEqual(referenced_view(row)["evidence"],
                         "(1) Example FC roster: José joined in 2020.")

    def test_top_spans_and_abstention(self):
        offsets = [(0, 0), (0, 4), (5, 9), (10, 14)]
        spans = top_spans([0, -1, 4, 2], [0, -1, 5, 1],
                          offsets, [1, 2, 3], top_k=2)
        self.assertEqual(spans[0], (9.0, 5, 9))
        valid = {"checks": [], "rank_score": 3.0, "pair_rank": 0,
                 "span_rank": 0, "selector_probability": 0.7,
                 "score_above_null": 2.0}
        forced, gated = choose_candidate([valid], 0.8, 0)
        self.assertEqual(forced, valid)
        self.assertIsNone(gated)

    def test_numeric_check_does_not_invent_dates(self):
        self.assertEqual(phrase_checks("2020", "2018", "joined in 2020",
                                       "joined in 2018", 80), [])
        self.assertIn("new_number_absent_from_context",
                      phrase_checks("2022", "2018", "joined in 2020",
                                    "joined in 2018", 80))

    def test_paired_comparison_requires_same_examples(self):
        a = {"id": 1, "source": "old", "gold_full": "new",
             "pipeline": {"scores": {"full_exact": False, "changed": False}}}
        b = {"id": 1, "source": "old", "gold_full": "new",
             "v2_gated": {"scores": {"full_exact": True, "changed": True}}}
        _, report = compare_rows([a], [b], "v2_gated")
        self.assertEqual((report["v1_exact"], report["v2_exact"],
                          report["v2_only_exact"]), (0, 1, 1))
        with self.assertRaises(ValueError):
            compare_rows([a], [{**b, "id": 2}], "v2_gated")


if __name__ == "__main__":
    unittest.main()
