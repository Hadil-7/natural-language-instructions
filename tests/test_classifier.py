"""Deterministic scope/CLI tests plus an opt-in real-model integration test."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import ACTIONS, CommandClassifier, PROJECT_DIR, evaluate, main, rejection_rule


class ScopeTests(unittest.TestCase):
    def test_negation_is_rejected(self):
        for text in ("Do not turn left", "don't go forward", "you mustn't reverse", "Turn right without moving"):
            with self.subTest(text=text):
                self.assertEqual(rejection_rule(text), "unsupported_negation")

    def test_multi_action_is_rejected(self):
        for text in ("Turn left and go ahead", "go forward then stop", "go backwards, turn right", "left or right", "Stop\nTurn right", "go forward, stop", "Stop. Turn right", "go ahead, jump"):
            with self.subTest(text=text):
                self.assertIsNotNone(rejection_rule(text))

    def test_polite_single_command_with_comma_is_allowed(self):
        self.assertIsNone(rejection_rule("Please, turn left"))

    def test_safe_stop_exception_is_explicit(self):
        self.assertIsNone(rejection_rule("Please do not move!"))
        self.assertIsNone(rejection_rule("Don't move."))
        self.assertEqual(rejection_rule("Do not move forward"), "unsupported_negation")

    def test_empty_and_long_inputs(self):
        self.assertEqual(rejection_rule("..."), "empty_instruction")
        self.assertEqual(rejection_rule("forward " * 100), "instruction_too_long")


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = CommandClassifier("baseline")

    def test_all_reference_classes_are_retrievable(self):
        for text, expected in (("move forward", "MOVE_FORWARD"), ("move backward", "MOVE_BACKWARD"),
                               ("turn left", "TURN_LEFT"), ("turn right", "TURN_RIGHT"), ("halt", "STOP")):
            with self.subTest(text=text):
                result = self.classifier.predict(text)
                self.assertEqual(result["action"], expected)
                self.assertEqual(set(result["scores"]), set(ACTIONS))

    def test_out_of_vocabulary_and_unsafe_commands_rejected(self):
        for text in ("make a sandwich", "turn left and stop", "do not turn right", ""):
            self.assertEqual(self.classifier.predict(text)["action"], "UNKNOWN")

    def test_result_explains_rejection(self):
        result = self.classifier.predict("turn left then reverse")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "multiple_actions")
        self.assertIsNone(result["similarity"])

    def test_margin_rejects_competing_classes(self):
        self.assertEqual(self.classifier.predict("turn")["reason"], "low_margin")

    def test_bad_configuration_rejected(self):
        with self.assertRaises(ValueError):
            CommandClassifier("unrecognized")
        with self.assertRaises(ValueError):
            CommandClassifier("baseline", threshold=1.1)
        with self.assertRaises(ValueError):
            self.classifier.predict("halt", -1)

    def test_evaluation_counts_are_consistent(self):
        report = evaluate(self.classifier, PROJECT_DIR / "data" / "held_out.csv")
        self.assertEqual(report["cases"], 64)
        self.assertEqual(sum(row["support"] for row in report["per_class"].values()), report["cases"])
        self.assertEqual(sum(sum(row.values()) for row in report["confusion_matrix"].values()), report["cases"])

    def test_old_cli_still_works(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["please turn left", "--backend", "baseline"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["action"], "TURN_LEFT")


@unittest.skipUnless(os.environ.get("NLP_RUN_MODEL_TESTS") == "1", "set NLP_RUN_MODEL_TESTS=1 with a cached model")
class EmbeddingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = CommandClassifier("embeddings", local_files_only=True)

    def test_real_embeddings_are_loaded(self):
        self.assertEqual(self.classifier.matrix.shape[0], 40)
        self.assertGreater(self.classifier.matrix.shape[1], 100)
        self.assertEqual(self.classifier.predict("please turn to your left")["action"], "TURN_LEFT")

    def test_semantic_report_has_predictions(self):
        report = evaluate(self.classifier, PROJECT_DIR / "data" / "held_out.csv")
        self.assertEqual(report["backend"], "embeddings")
        self.assertEqual(len(report["predictions"]), 64)


if __name__ == "__main__":
    unittest.main()
