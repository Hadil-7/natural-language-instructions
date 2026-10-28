"""Recognize one English navigation instruction with real sentence embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ACTIONS = ("MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
LABELS = (*ACTIONS, "UNKNOWN")

# Retrieval references, kept separate from the held-out evaluation set.
EXAMPLES = {
    "MOVE_FORWARD": [
        "move forward", "go straight ahead", "advance two steps", "continue forwards",
        "walk in a straight line ahead", "proceed straight on", "head forwards",
        "take a few steps forward",
    ],
    "MOVE_BACKWARD": [
        "move backward", "go back", "reverse two steps", "retreat backwards",
        "back up a little", "walk in reverse", "take a few steps back",
        "travel backwards",
    ],
    "TURN_LEFT": [
        "turn left", "rotate to the left", "take a left turn", "steer left",
        "pivot to your left", "make a left-hand turn", "face left", "swing left",
    ],
    "TURN_RIGHT": [
        "turn right", "rotate to the right", "take a right turn", "steer right",
        "pivot to your right", "make a right-hand turn", "face right", "swing right",
    ],
    "STOP": [
        "stop now", "halt", "do not move", "stay still", "remain stationary",
        "come to a complete stop", "cease moving", "hold your position",
    ],
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(value * b.get(word, 0.0) for word, value in a.items())
    norm = math.sqrt(sum(value * value for value in a.values()) * sum(value * value for value in b.values()))
    return dot / norm if norm else 0.0


def rejection_rule(text: str) -> str | None:
    """Conservative scope checks; these are not a general English parser."""
    normalized = " ".join(text.lower().replace("’", "'").split())
    if not normalized or not tokens(normalized):
        return "empty_instruction"
    if len(normalized) > 500:
        return "instruction_too_long"
    # A small, documented exception: these unambiguously mean STOP in this task.
    safe_stop = re.fullmatch(r"(?:please )?(?:do not|don't|never) move[.!?]*", normalized)
    if not safe_stop and re.search(r"\b(?:not|never|no|cannot|without|unless|avoid)\b|\b(?:don't|can't|won't|shouldn't|mustn't)\b", normalized):
        return "unsupported_negation"
    if "\n" in text or re.search(r"\b(?:and|then|after|before|followed by|subsequently|next)\b|;", normalized):
        return "multiple_actions"
    if re.search(r"\b(?:or|either|maybe|perhaps|if)\b", normalized):
        return "ambiguous_instruction"
    clauses = re.split(r"[,.;!?]+\s*", normalized)
    action_verb = r"\b(?:turn|rotate|pivot|steer|move|go|walk|advance|reverse|retreat|stop|halt|hold|stay|remain|jump|pick|open|close)\b"
    if sum(bool(re.search(action_verb, clause)) for clause in clauses) > 1:
        return "multiple_actions"
    direction_groups = (
        r"\b(?:forward|forwards|ahead|straight)\b",
        r"\b(?:backward|backwards|reverse|retreat)\b|\bgo back\b|\bback up\b",
        r"\bleft\b", r"\bright\b",
    )
    direction_count = sum(bool(re.search(pattern, normalized)) for pattern in direction_groups)
    if direction_count > 1 or (direction_count and re.search(r"\b(?:stop|halt)\b", normalized)):
        return "multiple_actions"
    return None


class CommandClassifier:
    """Nearest-reference semantic retrieval with rejection, not a robot driver.

    Embeddings is the primary backend. Baseline is explicitly lexical and needs
    no third-party packages. A missing model never silently selects baseline.
    """

    def __init__(self, backend: str = "embeddings", model_name: str | None = None,
                 threshold: float | None = None, min_margin: float = 0.04,
                 local_files_only: bool = False, revision: str | None = None,
                 model_id: str | None = None):
        if backend not in ("embeddings", "baseline"):
            raise ValueError("backend must be 'embeddings' or 'baseline'")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 <= min_margin <= 1:
            raise ValueError("min_margin must be between 0 and 1")
        self.backend = backend
        self.model_name = model_name or os.environ.get("NLP_MODEL_NAME", DEFAULT_MODEL)
        self.local_model = Path(self.model_name).is_dir()
        self.model_id = model_id or os.environ.get("NLP_MODEL_ID") or (
            f"local:{Path(self.model_name).name}" if self.local_model else self.model_name
        )
        self.revision = revision or os.environ.get("NLP_MODEL_REVISION") or (
            DEFAULT_REVISION if self.model_id == DEFAULT_MODEL else None
        )
        self.threshold = threshold if threshold is not None else (0.48 if backend == "embeddings" else 0.35)
        self.min_margin = min_margin
        self.labels, self.sentences = zip(*[(label, text) for label, examples in EXAMPLES.items() for text in examples])
        if backend == "embeddings":
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            self.model = SentenceTransformer(self.model_name, device="cpu", local_files_only=local_files_only, revision=self.revision)
            self.matrix = self.model.encode(list(self.sentences), normalize_embeddings=True, show_progress_bar=False)
            self._cosine_similarity = cosine_similarity
        else:
            docs = [set(tokens(text)) for text in self.sentences]
            self.idf = {
                word: math.log((len(docs) + 1) / (1 + sum(word in doc for doc in docs))) + 1
                for word in set().union(*docs)
            }
            self.matrix = [self.vector(text) for text in self.sentences]

    def vector(self, text: str) -> dict[str, float]:
        return {word: count * self.idf[word] for word, count in Counter(tokens(text)).items() if word in self.idf}

    def predict(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        """Return a discrete action plus similarity evidence and a rejection reason."""
        if not isinstance(text, str):
            raise TypeError("instruction must be a string")
        cutoff = self.threshold if threshold is None else threshold
        if not 0 <= cutoff <= 1:
            raise ValueError("threshold must be between 0 and 1")
        reason = rejection_rule(text)
        result: dict[str, Any] = {
            "instruction": text, "action": "UNKNOWN", "accepted": False,
            "reason": reason, "similarity": None, "margin": None,
            "backend": self.backend, "model": self.model_id if self.backend == "embeddings" else "tf-idf-reference-baseline",
        }
        if reason:
            return result
        if self.backend == "embeddings":
            vector = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            scores = self._cosine_similarity(vector, self.matrix)[0].tolist()
        else:
            vector = self.vector(text)
            scores = [cosine(vector, row) for row in self.matrix]
        by_label = {label: max(score for score, candidate in zip(scores, self.labels) if candidate == label) for label in ACTIONS}
        ranked = sorted(by_label.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        margin = best_score - ranked[1][1]
        reason = "low_similarity" if best_score < cutoff else ("low_margin" if margin < self.min_margin else "accepted")
        best_reference = max(range(len(scores)), key=scores.__getitem__)
        result.update({
            "action": best_label if reason == "accepted" else "UNKNOWN",
            "accepted": reason == "accepted", "reason": reason,
            "similarity": round(float(best_score), 6), "margin": round(float(margin), 6),
            "nearest_action": best_label, "matched_example": self.sentences[best_reference],
            "scores": {label: round(float(score), 6) for label, score in by_label.items()},
        })
        return result


def evaluate(classifier: CommandClassifier, dataset: Path) -> dict[str, Any]:
    with dataset.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "text", "label", "category"}.issubset(rows[0]):
        raise ValueError("dataset needs id,text,label,category columns and at least one row")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("dataset IDs must be unique")
    references = {sentence.lower() for sentence in classifier.sentences}
    confusion = {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}
    predictions = []
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"unknown expected label: {row['label']}")
        if row["text"].lower() in references:
            raise ValueError(f"evaluation text also exists in reference set: {row['id']}")
        prediction = classifier.predict(row["text"])
        confusion[row["label"]][prediction["action"]] += 1
        predictions.append({**row, "prediction": prediction})
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[expected][label] for expected in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": support}
    correct = sum(confusion[label][label] for label in LABELS)
    package_versions = {}
    for package in ("sentence-transformers", "scikit-learn", "torch", "transformers", "numpy"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    try:
        dataset_description = str(dataset.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        dataset_description = str(dataset)
    return {
        "dataset": dataset_description, "backend": classifier.backend,
        "model": classifier.model_id if classifier.backend == "embeddings" else "tf-idf-reference-baseline",
        "model_loading": ("local_directory" if classifier.local_model else "huggingface_id") if classifier.backend == "embeddings" else None,
        "model_revision": classifier.revision if classifier.backend == "embeddings" else None,
        "environment": {"python": platform.python_version(), "packages": package_versions},
        "threshold": classifier.threshold, "min_margin": classifier.min_margin,
        "reference_count": len(classifier.sentences), "cases": len(rows), "correct": correct,
        "accuracy": round(correct / len(rows), 6),
        "macro_f1": round(sum(row["f1"] for row in per_class.values()) / len(LABELS), 6),
        "per_class": per_class, "confusion_matrix": confusion, "predictions": predictions,
        "note": "Small hand-written held-out suite, not a generalization or robot-safety benchmark. Thresholds are fixed defaults, not optimized on these cases.",
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Backend: {report['backend']} | Model: {report['model']}")
    print(f"Accuracy: {report['correct']}/{report['cases']} ({report['accuracy']:.1%}) | Macro F1: {report['macro_f1']:.3f}")
    print(f"Threshold: {report['threshold']} | Minimum margin: {report['min_margin']}")
    print("\nClass              Precision  Recall     F1  Support")
    for label, metrics in report["per_class"].items():
        print(f"{label:18} {metrics['precision']:9.3f} {metrics['recall']:7.3f} {metrics['f1']:6.3f} {metrics['support']:8}")
    print("\nConfusion matrix: rows = expected, columns = predicted")
    print(" " * 18 + " ".join(f"{label:>13}" for label in LABELS))
    for expected, counts in report["confusion_matrix"].items():
        print(f"{expected:18}" + " ".join(f"{counts[label]:13}" for label in LABELS))
    failures = [row for row in report["predictions"] if row["label"] != row["prediction"]["action"]]
    print(f"\nMisclassified cases ({len(failures)}):")
    for row in failures:
        prediction = row["prediction"]
        print(f"  {row['id']}: {row['text']!r} -> {prediction['action']}; expected {row['label']} ({prediction['reason']})")
    print(f"\n{report['note']}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve `python main.py "please turn left" --backend baseline`.
    if not arguments or arguments[0] not in ("predict", "evaluate", "-h", "--help"):
        arguments.insert(0, "predict")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=("embeddings", "baseline"), default="embeddings")
    common.add_argument("--model", default=None, help="Hugging Face ID or local model directory; also NLP_MODEL_NAME")
    common.add_argument("--revision", default=None, help="Optional immutable model commit; also NLP_MODEL_REVISION")
    common.add_argument("--model-id", default=None, help="Public model identity to report when loading a local directory; also NLP_MODEL_ID")
    common.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    common.add_argument("--threshold", type=float, default=None)
    common.add_argument("--min-margin", type=float, default=0.04)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict", parents=[common], help="Classify one command")
    predict_parser.add_argument("instruction", nargs="?", default="turn to the left")
    evaluate_parser = subparsers.add_parser("evaluate", parents=[common], help="Evaluate the fixed held-out suite")
    evaluate_parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data" / "held_out.csv")
    evaluate_parser.add_argument("--output", type=Path, help="Save the complete report as JSON")
    args = parser.parse_args(arguments)
    try:
        classifier = CommandClassifier(args.backend, args.model, args.threshold, args.min_margin, args.local_files_only, args.revision, args.model_id)
        if args.command == "predict":
            print(json.dumps(classifier.predict(args.instruction), indent=2))
        else:
            report = evaluate(classifier, args.dataset)
            print_report(report)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"\nSaved detailed report to {args.output}")
    except (ImportError, OSError, ValueError) as error:
        print(f"Error: {error}\nFor the semantic backend install requirements.txt and cache the model. Use --backend baseline explicitly for the offline lexical comparison.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
