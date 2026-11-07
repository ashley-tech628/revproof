"""Meaningful contracts for sparse plans and evidence provenance."""
import unittest
from pathlib import Path

from multispan_edit import apply_plan, gold_plan, validate_grounded_record
from prepare_coedit import convert_row
from prepare_codocbench import convert_row as convert_codocbench
from revproof_data import read_records


class MultiSpanTest(unittest.TestCase):
    def test_all_existing_real_revisions_roundtrip(self):
        rows = read_records(Path(__file__).parent / "data" / "real_v2.jsonl")
        self.assertEqual(len(rows), 30)
        for row in rows:
            with self.subTest(row=row["id"]):
                plan = gold_plan(row["old_span"], row["new_span"])
                self.assertEqual(apply_plan(row["old_span"], plan), row["new_span"])

    def test_disjoint_edits_copy_middle_exactly(self):
        source = "Minimum Python 3.8.  Buffer limit 4 GiB.\nKeep `API` unchanged."
        target = "Minimum Python 3.9.  Buffer limit 8 PiB.\nKeep `API` unchanged."
        edits = gold_plan(source, target)
        self.assertEqual(len(edits), 3)
        self.assertEqual(apply_plan(source, edits), target)
        self.assertEqual(source[edits[0]["end"]:edits[1]["start"]],
                         "  Buffer limit ")

    def test_insert_delete_and_reject_overlap(self):
        for old, new in (("A B", "A new B"), ("A stale B", "A B"),
                         ("A\nB", "A\nB\nC")):
            self.assertEqual(apply_plan(old, gold_plan(old, new)), new)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            apply_plan("abcdef", [{"start": 1, "end": 4, "new": "x"},
                                  {"start": 3, "end": 5, "new": "y"}])

    def test_grounding_gate(self):
        row = {"id": "1", "group": "g", "old_text": "old", "new_text": "new",
               "evidence_text": "new evidence", "evidence_url": "https://example.org/release",
               "evidence_version": "v2", "target_version": "v2",
               "evidence_origin": "release_notes", "support_audited": True}
        self.assertIs(validate_grounded_record(row), row)
        with self.assertRaisesRegex(ValueError, "target document"):
            validate_grounded_record({**row, "evidence_origin": "target_document"})
        with self.assertRaisesRegex(ValueError, "audited"):
            validate_grounded_record({**row, "support_audited": False})

    def test_coedit_remains_pretraining_only(self):
        item = convert_row({"_id": "42", "task": "gec", "src": "Fix: wrong text",
                            "tgt": "Right text"})
        self.assertFalse(item["grounded"])
        self.assertEqual(item["training_role"], "editing_pretraining_only")
        self.assertIsNone(item["evidence_text"])

    def test_codocbench_is_unverified_silver(self):
        item = convert_codocbench({
            "project": "sample", "function": "f", "diff_code": "+ return 2",
            "whitespace_only_docstring": False,
            "version_data": [{"docstring": "Return one.", "commit_sha": "a"},
                             {"docstring": "Return two.", "commit_sha": "b"}],
        })
        self.assertFalse(item["support_audited"])
        self.assertEqual(item["training_role"], "domain_silver_candidate")
        self.assertEqual(item["evidence_origin"], "code_diff")


if __name__ == "__main__":
    unittest.main()
