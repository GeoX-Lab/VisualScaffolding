"""CPU checks for the public benchmark contract."""

import os
import tempfile
import unittest
from pathlib import Path

from visual_scaffolding.benchmark import evaluate_predictions, load_samples, read_predictions
from visual_scaffolding.scoring import score_text, summarize

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path(os.environ.get("NMR_BENCH_DIR", ROOT / "benchmark"))


class EvaluationTests(unittest.TestCase):
    def test_exact_path_and_partial_credit(self):
        sample = {"sample_id": "example", "answer": r"\boxed{(A,B),(C,D)}"}
        exact = score_text("path_tracing", sample, r"<think>text</think>\boxed{(d,c),(b,a),(b,a)}")
        self.assertTrue(exact["correct"])
        partial = score_text("path_tracing", sample, r"\boxed{(A,B)}")
        self.assertFalse(partial["correct"])
        self.assertEqual(partial["cra"], 0.5)
        unknown = score_text("path_tracing", sample, r"\boxed{(A,B),(X,Y)}")
        self.assertTrue(unknown["invalid"])
        self.assertEqual(unknown["cra"], 0)

    def test_counting_denominators_and_comparison_forms(self):
        samples = [
            {"sample_id": "a", "task_form": "T1a", "answer": r"\boxed{4}"},
            {"sample_id": "b", "task_form": "T2b", "answer": r"\boxed{6}"},
            {"sample_id": "c", "task_form": "T3a", "answer": r"\boxed{small, 0}"},
            {"sample_id": "d", "task_form": "T3b", "answer": r"\boxed{safety pin, 3}"},
        ]
        text = [r"\boxed{2}", "unparseable", r"\boxed{small, 0}", r"\boxed{Safety Pin, 3}"]
        rows = [
            dict(score_text("burden_counting", s, t), difficulty="easy")
            for s, t in zip(samples, text)
        ]
        result = summarize(rows, "burden_counting")["overall"]
        self.assertEqual(result["accuracy"], 50)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["mae_n"], 1)
        self.assertEqual(result["mae"], 2)

    def test_separate_reasoning_channel(self):
        sample = {"sample_id": "a", "task_form": "T1a", "answer": r"\boxed{7}"}
        self.assertTrue(
            score_text("burden_counting", sample, "", "The final count is 7.")["correct"]
        )

    def test_coverage_and_duplicate_predictions(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([{"sample_id": "a"}], {}, "path_tracing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text('{"sample_id":"a","response":"x"}\n' * 2)
            with self.assertRaises(ValueError):
                read_predictions([path])

    @unittest.skipUnless(BENCHMARK.is_dir(), "NMR-Bench data is distributed on Hugging Face")
    def test_released_annotation_coverage(self):
        self.assertEqual(len(list(load_samples(BENCHMARK, "path_tracing"))), 360)
        self.assertEqual(len(list(load_samples(BENCHMARK, "burden_counting"))), 420)


if __name__ == "__main__":
    unittest.main()
