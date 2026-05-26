import unittest
from pathlib import Path
import importlib.util
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "prepare_mmlu_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_mmlu_data", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

rows_from_records = MODULE.rows_from_records
parse_args = MODULE.parse_args


class PrepareMMLUDataTest(unittest.TestCase):
    def test_rows_from_records_schema_and_answer_mapping(self):
        rows = rows_from_records(
            [
                {
                    "question": "What is 2+2?",
                    "subject": "elementary_mathematics",
                    "choices": ["1", "2", "3", "4"],
                    "answer": 3,
                },
                {
                    "question": "Capital of France?",
                    "subject": "global_facts",
                    "choices": ["Berlin", "Paris", "Rome", "Madrid"],
                    "answer": "B",
                },
            ],
            split="test",
            start_idx=0,
        )

        self.assertEqual(len(rows), 2)

        required = {"problem", "solution", "subject", "answer", "level", "unique_id"}
        for row in rows:
            self.assertTrue(required.issubset(row.keys()))

        self.assertEqual(rows[0]["answer"], "D")
        self.assertIn("A. 1", rows[0]["problem"])
        self.assertIn("D. 4", rows[0]["problem"])
        self.assertIn("Correct option: D", rows[0]["solution"])

        self.assertEqual(rows[1]["answer"], "B")
        self.assertIn("Correct option: B", rows[1]["solution"])

        self.assertEqual(rows[0]["unique_id"], "mmlu/all/test/000000")
        self.assertEqual(rows[1]["unique_id"], "mmlu/all/test/000001")

    def test_rows_from_records_supports_more_than_four_options(self):
        rows = rows_from_records(
            [
                {
                    "question": "Pick the fifth option.",
                    "subject": "synthetic",
                    "choices": ["zero", "one", "two", "three", "four"],
                    "answer": 4,
                }
            ],
            split="test",
            start_idx=0,
        )

        self.assertEqual(rows[0]["answer"], "E")
        self.assertIn("E. four", rows[0]["problem"])
        self.assertIn("Correct option: E", rows[0]["solution"])

    def test_parse_args_defaults_to_data_mmlu_output_dir(self):
        with patch("sys.argv", ["prepare_mmlu_data.py"]):
            args = parse_args()

        self.assertEqual(args.output_dir, Path("data/mmlu"))


if __name__ == "__main__":
    unittest.main()
