import unittest
from pathlib import Path

from fruit_data import convert_tfrecord, normalize_fruit


class FruitDataTest(unittest.TestCase):
    def test_copy_and_reference_reconstruction(self):
        source, edited = normalize_fruit(
            "[0] Alpha is old. [1] Keep this. [CONTEXT] (0) New evidence",
            "(0) Alpha is new. [1]")
        self.assertEqual(source, "Alpha is old. Keep this.")
        self.assertEqual(edited, "Alpha is new. Keep this.")

    def test_invalid_copy_marker(self):
        with self.assertRaises(ValueError):
            normalize_fruit("[0] Only one. [CONTEXT] evidence", "[1]")

    def test_invalid_source_order(self):
        with self.assertRaises(ValueError):
            normalize_fruit("[1] Wrong start. [CONTEXT] evidence", "[1]")

    def test_reject_test_and_dev_as_training(self):
        for name in ["test.tfrecord", "dev_00009.tfrecord"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                convert_tfrecord(Path(name), Path("unused.jsonl"), "train")


if __name__ == "__main__":
    unittest.main()
