import json
import unittest

from codoc_experiment import assess, project_split, target


class CoDocExperimentTest(unittest.TestCase):
    def test_project_holdout_and_positive_control(self):
        rows = [{"group": f"project-{i}", "old_text": "old", "new_text": "new",
                 "evidence_text": "diff"} for i in range(20)]
        train, dev = project_split(rows)
        self.assertEqual(len({x["group"] for x in dev}), 2)
        self.assertFalse({x["group"] for x in train} & {x["group"] for x in dev})
        smoke_train, smoke_eval = project_split(rows, True)
        self.assertEqual(smoke_train, smoke_eval)
        self.assertEqual(len(smoke_train), 3)

    def test_gold_multispan_plan_and_invalid_output(self):
        row = {"old_text": "A 1 B 2 C", "new_text": "A 3 B 4 C"}
        gold = target(row, "plan")
        self.assertEqual(len(json.loads(gold)), 2)
        self.assertTrue(assess(row, "plan", gold)["exact"])
        self.assertFalse(assess(row, "plan", "```json\n[]\n```")["valid_plan"])
        self.assertTrue(assess(row, "full", row["new_text"])["exact"])


if __name__ == "__main__":
    unittest.main()
