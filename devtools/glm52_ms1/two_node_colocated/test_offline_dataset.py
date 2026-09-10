"""Offline provenance, prompt isolation and strict final-answer scoring tests."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_dataset import main, prepare_dataset, score_response  # noqa: E402


class OfflineDatasetTests(unittest.TestCase):
    def test_bundled_gsm8k_ten_original_ids_no_gold_rationale_in_messages(self):
        fixture = prepare_dataset("gsm8k")
        self.assertEqual(fixture["status"], "DATASET_PREPARED")
        self.assertEqual(len(fixture["cases"]), 10)
        case = fixture["cases"][0]
        self.assertEqual(case["id"], "gsm8k-test-0000")
        self.assertEqual(case["answer"], "18")
        self.assertNotIn("<<16-3-4=9>>", case["messages"][0]["content"])
        self.assertNotIn("#### 18", case["messages"][0]["content"])
        source = Path(fixture["source"]["path"])
        self.assertEqual(
            fixture["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
        )

    def test_gsm8k_local_jsonl_limit_and_source_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {"question": f"Question {i}?", "answer": f"Reasoning\n#### {i}"}
                    )
                    for i in range(12)
                )
            )
            fixture = prepare_dataset("gsm8k", path)
            self.assertEqual(len(fixture["cases"]), 10)
            self.assertEqual(fixture["cases"][-1]["answer"], "9")
            self.assertEqual(fixture["source"]["row_count"], 12)
            with self.assertRaises(ValueError):
                prepare_dataset("gsm8k", path, limit=13)

    def test_gpqa_missing_input_is_blocked_and_no_cases(self):
        fixture = prepare_dataset("gpqa")
        self.assertEqual(fixture["status"], "BLOCKED")
        self.assertEqual(fixture["cases"], [])
        self.assertIsNone(fixture["source"])

    def test_nonexistent_path_is_blocked(self):
        self.assertEqual(
            prepare_dataset("gpqa", "/this/path/does/not/exist.csv")["status"],
            "BLOCKED",
        )

    def test_gpqa_csv_deterministic_shuffle_and_answer_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpqa_diamond.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Correct Answer",
                        "Incorrect Answer 1",
                        "Incorrect Answer 2",
                        "Incorrect Answer 3",
                    ],
                )
                writer.writeheader()
                for i in range(10):
                    writer.writerow(
                        {
                            "Question": f"Question {i}?",
                            "Correct Answer": f"Correct {i}",
                            "Incorrect Answer 1": "Wrong one",
                            "Incorrect Answer 2": "Wrong two",
                            "Incorrect Answer 3": "Wrong three",
                        }
                    )
            first, second = prepare_dataset("gpqa", path), prepare_dataset("gpqa", path)
            self.assertEqual(first, second)
            self.assertNotEqual(
                first["cases"], prepare_dataset("gpqa", path, seed=43)["cases"]
            )
            self.assertEqual(len(first["cases"]), 10)
            for i, case in enumerate(first["cases"]):
                self.assertEqual(case["choices"][case["answer"]], f"Correct {i}")
                self.assertEqual(
                    case["option_source_order"]["ABCD".index(case["answer"])],
                    "correct_answer",
                )
                self.assertEqual(
                    case["question_sha256"],
                    hashlib.sha256(case["question"].encode()).hexdigest(),
                )

    def test_gpqa_explicit_jsonl_schema_and_compact_official_columns(self):
        records = [
            {
                "question": "Q?",
                "correct_answer": "Yes",
                "incorrect_answers": ["No", "Maybe", "Unknown"],
            },
            {
                "Question": "Q?",
                "Correct Answer": "Yes",
                "Incorrect Answer1": "No",
                "Incorrect Answer2": "Maybe",
                "Incorrect Answer3": "Unknown",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpqa_diamond.jsonl"
            for row in records:
                path.write_text(json.dumps(row) + "\n")
                fixture = prepare_dataset("gpqa", path, limit=1)
                self.assertEqual(fixture["status"], "DATASET_PREPARED")
                self.assertEqual(
                    fixture["cases"][0]["choices"][fixture["cases"][0]["answer"]], "Yes"
                )

    def test_gpqa_duplicate_options_and_unsupported_schema_rejected(self):
        for row in (
            {
                "question": "Q?",
                "correct_answer": "Yes",
                "incorrect_answers": ["Yes", "No", "Maybe"],
            },
            {
                "question": "Q?",
                "choices": ["Yes", "No", "Maybe", "Unknown"],
                "answer": 0,
            },
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.jsonl"
                path.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    prepare_dataset("gpqa", path, limit=1)

    def test_wrong_parameters_rejected(self):
        for kwargs in ({"limit": True}, {"limit": 0}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_dataset("gsm8k", **kwargs)

    def test_gsm_explicit_markers_and_exact_numeric_normalization(self):
        for text, gold in (
            ("Reasoning 999\n#### 18", "18"),
            ("First \\boxed{10}; finally \\boxed{18}", "18"),
            ("Final answer: $70,000.", "70000"),
            ("Final answer is 0.5", "1/2"),
            ("\\boxed{\\frac{1}{2}}", "0.5"),
            ("#### -2", "-2"),
        ):
            with self.subTest(text=text):
                score = score_response("gsm8k", gold, text, "stop")
                self.assertTrue(score["correct"])

    def test_gsm_unmarked_or_ambiguous_last_result_never_guessed(self):
        for text in (
            "The value is 18",
            "#### 18 or 19",
            "#### 18\nFinal answer: uncertain",
            "\\boxed{18}\n\\boxed{",
            "\\boxed{18} or \\boxed{19}",
            "Final answer: 19 or \\boxed{18}",
            "#### 18 dollars",
            "#### 1,23",
            "#### 0/0",
            "print(18)",
            "#### __import__('os').system('anything')",
        ):
            with self.subTest(text=text):
                score = score_response("gsm8k", "18", text, "stop")
                self.assertEqual(score["status"], "UNPARSED")
                self.assertIsNone(score["correct"])

    def test_gsm_last_marked_correction_wins_not_last_random_number(self):
        score = score_response(
            "gsm8k", "18", "#### 12\nFinal answer: 18\nI checked it 2 times", "stop"
        )
        self.assertTrue(score["correct"])
        self.assertFalse(score_response("gsm8k", "18", "#### 19", "stop")["correct"])

    def test_gpqa_explicit_last_answer_and_no_letter_guessing(self):
        for text in (
            "Answer: B",
            "Answer: A\nFinal answer: (B).",
            "Reasoning\nAnswer: **B**",
        ):
            self.assertTrue(score_response("gpqa", "B", text, "stop")["correct"])
        for text in (
            "B",
            "I think B",
            "Answer: A or B",
            "Answer: B\nAnswer: uncertain",
        ):
            self.assertEqual(
                score_response("gpqa", "B", text, "stop")["status"], "UNPARSED"
            )
        self.assertFalse(score_response("gpqa", "A", "Answer: B", "stop")["correct"])

    def test_truncated_matching_answer_retained_but_never_correct(self):
        for dataset, answer, text in (
            ("gsm8k", "18", "#### 18"),
            ("gpqa", "A", "Answer: A"),
        ):
            score = score_response(
                dataset, answer, text, {"type": "length", "length": 1024}
            )
            self.assertEqual(score["status"], "TRUNCATED")
            self.assertTrue(score["answer_matches"])
            self.assertIsNone(score["correct"])

    def test_abort_missing_finish_and_missing_text_not_correct(self):
        for finish in (None, "abort", {"type": "abort"}, {}):
            self.assertEqual(
                score_response("gsm8k", "18", "#### 18", finish)["status"],
                "INVALID_FINISH",
            )
        self.assertEqual(
            score_response("gsm8k", "18", None, "stop")["status"], "UNPARSED"
        )

    def test_invalid_gold_cannot_silently_score_false(self):
        for dataset, gold in (("gsm8k", "ambiguous"), ("gpqa", "AB")):
            with self.assertRaises(ValueError):
                score_response(dataset, gold, "#### 18", "stop")

    def test_cli_writes_blocked_manifest_without_data_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gpqa.json"
            code = main(["prepare", "--dataset", "gpqa", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "BLOCKED")

    def test_cli_prepares_default_gsm8k(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gsm8k.json"
            self.assertEqual(
                main(["prepare", "--dataset", "gsm8k", "--output", str(output)]), 0
            )
            self.assertEqual(len(json.loads(output.read_text())["cases"]), 10)


if __name__ == "__main__":
    unittest.main()
