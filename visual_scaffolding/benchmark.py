"""Dataset loading and model-independent evaluation."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Protocol

from PIL import Image

from .scoring import TASKS, score_text, summarize


class ModelAdapter(Protocol):
    def predict(self, image: Image.Image, question: str) -> str | dict[str, str]:
        """Return response text, optionally with a separate reasoning channel."""
        ...


def load_samples(root, task):
    root = Path(root).resolve()
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["tasks"][task]
    annotations = (root / entry["annotations"]).resolve()
    if not annotations.is_relative_to(root):
        raise ValueError("Annotations must be inside the benchmark directory")
    blob = annotations.read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError("Benchmark annotation checksum mismatch")
    samples = [json.loads(line) for line in blob.splitlines() if line.strip()]
    ids = [s["sample_id"] for s in samples]
    if len(samples) != entry["samples"] or len(set(ids)) != len(ids):
        raise ValueError("Benchmark coverage or sample IDs are invalid")
    for sample in samples:
        if sample["difficulty"] not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown difficulty in {sample['sample_id']}")
        path = (annotations.parent / sample["image"]).resolve()
        if not path.is_relative_to(annotations.parent):
            raise ValueError("Image path leaves the task directory")
        sample = sample.copy()
        sample["image_path"] = path
        yield sample


def open_image(sample):
    path = sample["image_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample["image_sha256"]:
        raise ValueError(f"Image checksum mismatch: {sample['sample_id']}")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_adapter(spec, config):
    if ":" not in spec:
        raise ValueError("Adapter must be module:factory")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    model = factory(config)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Model adapters must implement predict(image, question)")
    return model


def read_predictions(paths):
    records = {}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row["sample_id"]
                if sid in records:
                    raise ValueError(f"Duplicate prediction: {sid}")
                if not isinstance(row.get("response"), str) or not isinstance(
                    row.get("reasoning", ""), str
                ):
                    raise TypeError(f"Prediction must contain string response/reasoning: {sid}")
                records[sid] = row
    return records


def evaluate_predictions(samples, predictions, task):
    expected = {s["sample_id"] for s in samples}
    if set(predictions) != expected:
        missing, extra = expected - predictions.keys(), predictions.keys() - expected
        raise ValueError(
            f"Prediction coverage mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    rows = []
    for sample in samples:
        prediction = predictions[sample["sample_id"]]
        row = score_text(task, sample, prediction["response"], prediction.get("reasoning", ""))
        row.update(sample_id=sample["sample_id"], difficulty=sample["difficulty"])
        rows.append(row)
    return summarize(rows, task), rows


def run_model(model, samples, path):
    """Only images and questions are passed to the model, never target answers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for index, sample in enumerate(samples, 1):
            output = model.predict(open_image(sample), sample["question"])
            if isinstance(output, str):
                text, reasoning = output, ""
            elif isinstance(output, dict):
                text, reasoning = output["text"], output.get("reasoning", "")
            else:
                raise TypeError("predict must return a string or a text/reasoning dictionary")
            if not isinstance(text, str) or not isinstance(reasoning, str):
                raise TypeError("Model text/reasoning must be strings")
            row = {"sample_id": sample["sample_id"], "response": text}
            if reasoning:
                row["reasoning"] = reasoning
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(samples)} {sample['sample_id']}", flush=True)
