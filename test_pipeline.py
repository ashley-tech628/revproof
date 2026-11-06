"""Small contract checks; these do not test model quality."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from make_toy_data import build_rows
from make_retrieval_benchmark import build as build_retrieval
from make_reranker_pairs import build as build_reranker
from prepare_preferences import build_pairs
from revproof_data import check_candidate, old_document, read_records, template_edit
from retrieval_eval import bm25, score_row


class PipelineTest(unittest.TestCase):
    def save(self, rows):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "rows.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_reconstruction_and_answer_relations(self):
        rows = read_records(self.save(build_rows()))
        self.assertEqual({r["z"] for r in rows}, {"keep", "change"})
        for row in rows:
            self.assertIn(row["old_span"], old_document(row))
            self.assertTrue(check_candidate(row, template_edit(row))["passes_string_checks"])

    def test_group_split_leakage_rejected(self):
        rows = build_rows()
        rows[1]["group"] = rows[0]["group"]
        rows[1]["split"] = "dev" if rows[0]["split"] != "dev" else "test"
        with self.assertRaisesRegex(ValueError, "group leakage"):
            read_records(self.save(rows))

    def test_real_source_required(self):
        rows = build_rows()
        rows[0]["source_type"] = "real"
        rows[0]["source_url"] = ""
        with self.assertRaisesRegex(ValueError, "source URL"):
            read_records(self.save(rows))

    def test_candidate_extractor_on_tagged_docs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            repo = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(repo), *args], check=True,
                               capture_output=True, text=True)
            git("init", "-q")
            git("config", "user.name", "Pipeline Test")
            git("config", "user.email", "test@example.invalid")
            (repo / "docs").mkdir()
            doc = repo / "docs" / "config.md"
            doc.write_text("Retry limit: 3\nTimeout: 10\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "old docs")
            git("tag", "v1")
            doc.write_text("Retry limit: 2\nTimeout: 10\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "new docs")
            git("tag", "v2")
            output = repo / "candidates.jsonl"
            subprocess.run([sys.executable, str(Path(__file__).parent / "collect_candidates.py"),
                            "--repo", str(repo), "--old-tag", "v1", "--new-tag", "v2",
                            "--output", str(output)], check=True, capture_output=True, text=True)
            candidates = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["old_lines"], ["Retry limit: 3"])
            self.assertEqual(candidates[0]["new_lines"], ["Retry limit: 2"])

    def test_preferences_require_manual_train_label(self):
        row = build_rows()[0]
        row.update({"source_type": "real", "source_url": "https://example.invalid/commit/1",
                    "split": "train", "z": "change"})
        audit = [{"id": row["id"], "candidate_index": "0",
                  "candidate_span": row["old_span"], "human_valid": "false"}]
        pairs = build_pairs([row], audit)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"][0]["content"], row["new_span"])
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train split"):
            build_pairs([row], audit)

    def test_protected_fact_failure_visible(self):
        row = build_rows()[0]
        row["protected_facts"] = ["The timeout is 10 seconds."]
        row["suffix"] = ""
        row["old_span"] += " The timeout is 10 seconds."
        assessment = check_candidate(row, "The retry limit is 2.")
        self.assertFalse(assessment["checks"]["protected_facts_present"])

    def test_version_filter_is_strong_retrieval_baseline(self):
        row = build_rows()[0]
        row["z"] = "change"
        row["split"] = "dev"
        benchmark = build_retrieval([row], "dev")[0]
        ranked = score_row(benchmark, "version_bm25")
        self.assertEqual([x["version"] for x in ranked], [row["version_after"]])
        self.assertGreater(bm25("retry limit", ["retry limit", "timeout"])[0],
                           bm25("retry limit", ["retry limit", "timeout"])[1])

    def test_reranker_builder_uses_real_labels(self):
        row = build_rows()[0]
        row.update({"source_type": "real", "source_url": "https://example.invalid/commit/2",
                    "split": "train"})
        pairs = build_reranker([row], "train")
        self.assertEqual([p["label"] for p in pairs], [1, 0])
        self.assertEqual({p["source"] for p in pairs}, {"real_new", "real_old"})
        duplicate = [{"id": row["id"], "candidate_span": row["new_span"], "human_valid": "true"}]
        self.assertEqual(build_reranker([row], "train", duplicate), pairs)


if __name__ == "__main__":
    unittest.main()
