# Natural Language Instructions

Semantic matching of a single English navigation instruction to a discrete action using **SentenceTransformers text embeddings** and **scikit-learn cosine similarity**. The primary implementation loads a real pretrained language model; it does not approximate semantic embeddings with keyword rules.

Supported outputs: `MOVE_FORWARD`, `MOVE_BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`, or `UNKNOWN`.

This repository is a reproducible implementation of a CV project. It runs on a CPU and returns a proposed action as data. It does not connect to or move a physical robot.

## Quick start

Use Python 3.10 or newer. Python 3.11 or 3.12 is a practical choice for the machine-learning dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py predict "Please rotate towards your left"
```

On Windows, activate with `.venv\Scripts\activate` instead. The first semantic prediction downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, pinned by default to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Internet access and disk space for the model are needed once. Downloaded model weights are not part of the repository.

Example output structure (similarity values depend on the model and input):

```json
{
  "instruction": "Please rotate towards your left",
  "action": "TURN_LEFT",
  "accepted": true,
  "reason": "accepted",
  "backend": "embeddings",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

The actual output additionally includes the best similarity, the margin between the two highest-scoring actions, a nearest reference, and all five action scores. Similarity is **not** a calibrated probability.

## Run predictions

```bash
python main.py predict "Take a step backwards"
python main.py predict "Make me a sandwich"
python main.py predict "Turn left and then stop"
python main.py predict "Do not turn right"
```

The last two are rejected as `UNKNOWN` with an explicit reason. These commands also work after the model has been cached:

```bash
python main.py predict "Please halt" --local-files-only
python main.py predict "Please turn left" --model /path/to/local/model
```

Set `HF_HOME` to choose Hugging Face's cache directory, or `NLP_MODEL_NAME` to select a model ID/local model directory. `--model` takes precedence over `NLP_MODEL_NAME`. Use `--revision COMMIT_HASH` (or `NLP_MODEL_REVISION`) to override the model revision. Using a local directory is useful for reproducing a saved model snapshot without a network connection. When loading a known local model, `--model-id sentence-transformers/all-MiniLM-L6-v2` (or `NLP_MODEL_ID`) records its public identity in reports without exposing machine-specific paths.

The older CLI syntax is still accepted:

```bash
python main.py "please turn left" --backend embeddings
```

## How it works

1. Scope checks reject empty/very long inputs, common negations, conditional/ambiguous requests and multi-action instructions.
2. `all-MiniLM-L6-v2` encodes 40 short reference instructions and the input as dense sentence vectors. Reference vectors are computed once when the classifier is loaded.
3. Cosine similarity compares the input against all reference vectors. Each action receives its best reference score.
4. The top action must meet a minimum similarity (default `0.48`) and exceed the runner-up by a margin (default `0.04`). Otherwise the output is `UNKNOWN`.

The five outputs describe a coarse action category. Distances, angles and speeds in a sentence are **not** extracted or executed. For example, a command containing “ninety degrees” may map to a turn category, but the program does not produce a 90-degree control signal.

Rejection checks are deliberately simple rules, not a full English grammar. The exact stop expressions “do not move”, “don't move” and “never move” (optionally prefixed with “please”) are allowed as stop requests; other detected negations are rejected. Some valid English commands will be rejected, and some unrelated inputs can still be accepted. Do not use this classifier as a safety mechanism.

## Evaluate the real semantic model

```bash
python main.py evaluate --output results/embeddings.json
# Equivalent compatibility entry point:
python evaluate.py --output results/embeddings.json
```

For a model already downloaded:

```bash
python main.py evaluate --local-files-only --output results/embeddings.json
```

The evaluation prints accuracy, macro F1, precision/recall/F1 for every class, a full confusion matrix, and every misclassified case. The JSON report also stores every input, expected action, prediction, similarity, rejection reason and installed package versions. The evaluator checks for exact overlap between reference and evaluation sentences and rejects duplicate case IDs.

`data/held_out.csv` contains 64 manually written cases: eight paraphrases for each of the five actions and 24 unknown/out-of-scope examples, including unrelated requests, negations, multi-action plans, and misleading direction words. The reference sentences are in `EXAMPLES` in `main.py`. The default thresholds were selected before running this expanded suite; this is not an automatically tuned benchmark. Similar paraphrases remain closely related, and this small hand-written set does not measure deployment performance. Add an independently collected test set before making wider accuracy claims.

Measured outputs are saved in `results/`. Consult the actual report rather than treating every sample as a guaranteed correct prediction. No supervised fine-tuning or task-specific training is claimed: the semantic encoder is pretrained and the project implements retrieval and rejection on top of it.

Measured results for the pinned model and fixed defaults:

| Backend | Correct / total | Accuracy | Macro F1 | Unknown examples rejected |
| --- | --- | --- | --- | --- |
| SentenceTransformer embeddings | 51 / 64 | 79.7% | 0.798 | 23 / 24 |
| Lexical TF-IDF baseline | 57 / 64 | 89.1% | 0.897 | 21 / 24 |

The embedding backend does **not** outperform the baseline on overall accuracy in this small set. Its conservative margin rejects several valid forward/backward and left/right paraphrases, while it rejects more of the unknown examples. Similar sentence vectors can place opposite navigation directions close together. It also incorrectly accepts “Right is the correct answer” as `TURN_RIGHT`. Both reports retain all failures; thresholds were not changed to improve the test score. A useful next experiment is a separate development set for threshold selection and a fresh, independently collected test set.

The checked-in reports were generated on Python 3.12.14. `requirements-tested.txt` records the exact direct package versions used. To recreate the CPU environment on Linux:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
```

`requirements.txt` allows compatible versions for a normal installation; the reports record the versions actually tested.

## Explicit offline comparison

The optional baseline uses reference-set TF-IDF weights and cosine similarity, entirely in the Python standard library. It works without packages or model downloads, but measures lexical overlap, **not semantic language understanding**. The program never silently switches to it when a model is unavailable.

```bash
python main.py predict "please turn left" --backend baseline
python main.py evaluate --backend baseline --output results/baseline.json
```

Baseline results on the included set: **57/64 correct (89.1%), macro F1 0.897**. Its false positives on “Pick up the red cup”, “Right is the correct answer” and “I left my keys at home” illustrate its limitations. These are measured results, not expected performance on unseen real-world commands.

## Tests

The fast tests cover rejection behavior, action retrieval, ambiguous margins, invalid settings, report accounting and the compatibility CLI:

```bash
python -m unittest discover -s tests -v
```

Run the real-model integration tests after downloading the model:

```bash
NLP_RUN_MODEL_TESTS=1 python -m unittest discover -s tests -v
```

On PowerShell, set `$env:NLP_RUN_MODEL_TESTS="1"` first. Integration tests use local model files only and do not initiate a download. They check that dense model embeddings are loaded and that a complete semantic evaluation can be produced. Evaluation quality is reported separately rather than asserted to be perfect.

GitHub Actions runs the lightweight scope/baseline tests on pushes and pull requests and uploads the full baseline evaluation as a workflow artifact. It does not download the semantic model on every push; run the integration command above when changing model behavior.

## Files

```text
main.py                    Classifier, prediction CLI and evaluation CLI
evaluate.py                Compatibility evaluation entry point
data/held_out.csv           Labelled held-out cases
tests/test_classifier.py   Scope, baseline and optional model integration tests
results/                   Measured per-case reports
requirements.txt           Dependencies for real semantic embeddings
```

## Model and dependency sources

- [SentenceTransformers documentation](https://www.sbert.net/)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — pretrained model, Apache-2.0 license
- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

The pretrained model remains subject to its own model license. Its weights are downloaded separately rather than copied into this repository.
