"""Checks for boundary proposals, label isolation, and paired comparison."""

import unittest
from unittest.mock import patch

from span_editor.boundary_v3 import (build_examples, marked_sentence,
                                     merge_candidates, nearby_spans, propose,
                                     rank_boundary_candidates)
from span_editor.compare_v3 import compare_rows
from span_editor.evaluate_v2 import inference_view
from span_editor.mine_boundary_hard import (collect_proposals,
                                            select_hard_negatives)
from span_editor.train_boundary_hard import build_examples as build_hard_examples


class Tokenizer:
    def __call__(self, first, second, **kwargs):
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


class BoundaryV3Test(unittest.TestCase):
    def test_neighbor_contains_whole_gold_after_wide_qa_span(self):
        sentence = "José joined Example FC in 2018."
        start = sentence.index("joined")
        end = sentence.index(" FC") + len(" FC")
        gold_start = sentence.index("Example")
        self.assertIn((gold_start, end), nearby_spans(
            sentence, start, end, radius=3, max_variants=36))
        self.assertIn("[EDIT] Example FC [/EDIT]",
                      marked_sentence(sentence, gold_start, end))

    def test_training_uses_gold_only_for_labels(self):
        sentence = "José joined Example FC in 2018."
        row = {"id": 1, "source_sentences": [sentence], "edit_index": 0,
               "span_start": sentence.index("2018"),
               "span_end": sentence.index("2018") + 4,
               "evidence_items": ["José joined in 2020."], "reference_ids": [0]}
        examples, positives, skipped = build_examples(
            Tokenizer(), [row], negatives=3)
        self.assertEqual((positives, skipped), (1, 0))
        self.assertEqual(sum(x["labels"] for x in examples), 1)
        view = inference_view({**row, "source": sentence, "evidence": "(0) José joined in 2020."})
        for field in ("span_start", "span_end", "edit_index", "reference_ids"):
            self.assertNotIn(field, view)

    def test_pair_budget_limits_duplicates(self):
        candidates = [{"sentence_index": 0, "evidence_id": 0,
                       "selector_probability": 0.8, "score_above_null": 1.0,
                       "boundary_logit": float(4 - i), "pair_rank": 0,
                       "start": i, "end": i + 2} for i in range(3)]
        candidates.append({**candidates[0], "pair_rank": 1,
                           "evidence_id": 1, "boundary_logit": 0.5})
        chosen = rank_boundary_candidates(candidates, 3, 2)
        self.assertEqual(len(chosen), 3)
        self.assertEqual([x["pair_rank"] for x in chosen], [0, 0, 1])

    def test_hybrid_respects_eight_location_budget(self):
        baseline = [{"sentence_index": 0, "start": i, "end": i + 1}
                    for i in range(8)]
        refined = [{"sentence_index": 0, "start": i, "end": i + 1}
                   for i in (0, 10, 11, 12, 13)]
        hybrid = merge_candidates(baseline, refined, 4, 8)
        self.assertEqual(len(hybrid), 8)
        self.assertEqual({x["start"] for x in hybrid},
                         {0, 1, 2, 3, 10, 11, 12, 13})

    def test_proposal_path_uses_only_inference_view(self):
        sentence = "José joined Example FC in 2018."
        source = {"id": 1, "source": sentence,
                  "source_sentences": [sentence],
                  "evidence": "(0) joined in 2020",
                  "evidence_items": ["joined in 2020"]}
        pairs = [{"sentence_index": 0, "evidence_id": 0,
                  "selector_probability": 0.8}]
        a = sentence.index("2018")
        location = {"sentence_index": 0, "start": a, "end": a + 4,
                    "old_span": "2018", "score_above_null": 2.0}
        def scored(_model, _tokenizer, items, _torch, _max_length):
            return [{**item, "boundary_logit": 1.0} for item in items]
        with patch("span_editor.boundary_v3.rank_pairs", return_value=pairs), \
             patch("span_editor.boundary_v3.locate_sentence_candidates",
                   return_value=[location]), \
             patch("span_editor.boundary_v3.score_candidates",
                   side_effect=scored):
            ranked_pairs, candidates = propose(
                None, None, None, None, None, None, source, None,
                384, 384, 384, max_candidates=3)
        self.assertEqual(ranked_pairs, pairs)
        self.assertTrue(candidates)
        self.assertTrue(all("sentence" not in x and "evidence" not in x
                            for x in candidates))

    def test_hard_mining_uses_labels_after_proposals(self):
        sentence = "José joined Example FC in 2018."
        a = sentence.index("2018")
        row = {"id": 1, "source": sentence,
               "source_sentences": [sentence, "The team played in Paris."],
               "evidence": "(0) joined in 2020 (1) Paris",
               "evidence_items": ["joined in 2020", "Paris"],
               "edit_index": 0, "span_start": a, "span_end": a + 4,
               "reference_ids": [0]}
        source = inference_view(row)
        pairs = [{"sentence_index": 0, "evidence_id": 0,
                  "selector_probability": 0.8}]
        location = {"sentence_index": 0, "start": a, "end": a + 4,
                    "score_above_null": 2.0}
        with patch("span_editor.mine_boundary_hard.rank_pairs",
                   return_value=pairs), \
             patch("span_editor.mine_boundary_hard.locate_sentence_candidates",
                   return_value=[location]):
            proposals = collect_proposals(None, None, None, None,
                                          source, None, 384, 384)
        self.assertTrue(any(x["start"] == a for x in proposals))
        negatives = select_hard_negatives(row, proposals, limit=4)
        self.assertTrue(negatives)
        self.assertLessEqual(len(negatives), 4)
        self.assertFalse(any(x["sentence_index"] == 0 and
                             x["evidence_id"] == 0 and
                             x["start"] == a and x["end"] == a + 4
                             for x in negatives))
        mined = [{"id": 1, "negatives": negatives}]
        examples, positives, negative_count, skipped = build_hard_examples(
            Tokenizer(), [row], mined)
        self.assertEqual((positives, negative_count, skipped),
                         (1, len(negatives), 0))

    def test_v2_v3_pairing_and_mismatches(self):
        a = {"id": 1, "source": "old", "gold_full": "new",
             "v2_forced": {"scores": {"full_exact": False, "changed": True}}}
        b = {"id": 1, "source": "old", "gold_full": "new",
             "v3_forced": {"scores": {"full_exact": True, "changed": True}}}
        _, report = compare_rows([a], [b])
        self.assertEqual((report["v2_exact"], report["v3_exact"],
                          report["v3_only_exact"]), (0, 1, 1))
        with self.assertRaises(ValueError):
            compare_rows([a], [{**b, "gold_full": "other"}])


if __name__ == "__main__":
    unittest.main()
