"""Load the shipped LSTM and run one forward pass on a landmark window."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import keras
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from fsl.checks import assert_never
from fsl.labels import load_labels
from fsl.landmarks import KEYPOINT_DIM
from fsl.paths import models_dir
from fsl.sequence import SEQUENCE_LENGTH, scale_sequence, shape_sequence

ModelName = Literal["fsl15", "fsl11"]


@dataclass(frozen=True)
class ModelPaths:
    name: ModelName
    model: Path
    fallback_model: Path | None
    scaler: Path
    labels: Path


@dataclass
class FSLBundle:
    name: ModelName
    model: Any
    scaler: Any
    labels: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class SmokeReport:
    model_name: str
    classes: int
    parameters: int
    input_shape: tuple[int, int, int]
    prob_sum: float
    top_label: str
    top_confidence: float
    labels: tuple[str, ...]


def model_paths(name: ModelName = "fsl15") -> ModelPaths:
    root = models_dir()
    match name:
        case "fsl15":
            return ModelPaths(
                name="fsl15",
                model=root / "fsl15" / "model.h5",
                fallback_model=root / "fsl15" / "best.h5",
                scaler=root / "fsl15" / "scaler.pkl",
                labels=root / "fsl15" / "labels.npy",
            )
        case "fsl11":
            return ModelPaths(
                name="fsl11",
                model=root / "fsl11" / "model.h5",
                fallback_model=None,
                scaler=root / "fsl11" / "scaler.pkl",
                labels=root / "fsl11" / "labels.npy",
            )
        case unreachable:
            assert_never(unreachable)


def build_lstm(num_classes: int) -> keras.Model:
    """Architecture stored in the shipped ``.h5`` files (Keras 3.13)."""
    regularizer = keras.regularizers.L2(0.001)
    return keras.Sequential(
        [
            keras.layers.Input(shape=(SEQUENCE_LENGTH, KEYPOINT_DIM)),
            keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.LSTM(128, return_sequences=False, kernel_regularizer=regularizer),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def load_keras_model(path: Path, num_classes: int) -> keras.Model:
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as first_error:
        model = build_lstm(num_classes)
        try:
            model.load_weights(path)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load {path} as a Keras model ({first_error}) "
                f"or as weights ({second_error})."
            ) from second_error
        return model


def load_scaler(path: Path) -> Any:
    """Load a ``StandardScaler`` saved with scikit-learn 1.6.1."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        scaler = joblib.load(path)
    features = getattr(scaler, "n_features_in_", None)
    if features is not None and int(features) != KEYPOINT_DIM:
        raise ValueError(
            f"Scaler at {path} expects {features} features, landmark vectors are {KEYPOINT_DIM}"
        )
    return scaler


def resolve_weight_file(paths: ModelPaths) -> Path:
    if paths.model.is_file():
        return paths.model
    if paths.fallback_model is not None and paths.fallback_model.is_file():
        return paths.fallback_model
    expected = [str(paths.model)]
    if paths.fallback_model is not None:
        expected.append(str(paths.fallback_model))
    raise FileNotFoundError(
        "Missing model weights: "
        + ", ".join(expected)
        + ". The files are committed in this clone. After a history purge, run "
        "python scripts/download_weights.py once the GitHub Release assets exist."
    )


def load_bundle(name: ModelName = "fsl15") -> FSLBundle:
    paths = model_paths(name)
    if not paths.labels.is_file():
        raise FileNotFoundError(f"Missing label file: {paths.labels}")
    if not paths.scaler.is_file():
        raise FileNotFoundError(f"Missing scaler file: {paths.scaler}")
    labels = load_labels(paths.labels)
    weight_path = resolve_weight_file(paths)
    model = load_keras_model(weight_path, len(labels))
    scaler = load_scaler(paths.scaler)
    return FSLBundle(name=name, model=model, scaler=scaler, labels=labels, path=weight_path)


def predict_proba(bundle: FSLBundle, frames: Any) -> np.ndarray:
    """Return the softmax vector for one window."""
    batch = scale_sequence(shape_sequence(frames, pad=False), bundle.scaler)
    output = bundle.model(batch, training=False)
    values = output.numpy() if hasattr(output, "numpy") else np.asarray(output)
    return np.asarray(values, dtype=np.float32)[0]


def predict_label(bundle: FSLBundle, frames: Any) -> tuple[str, float]:
    probabilities = predict_proba(bundle, frames)
    index = int(np.argmax(probabilities))
    return bundle.labels[index], float(probabilities[index])


def smoke_inference(name: ModelName = "fsl15") -> SmokeReport:
    """Run the network on a 30×258 zero sequence. This is a load check, not an accuracy number."""
    bundle = load_bundle(name)
    zeros = np.zeros((SEQUENCE_LENGTH, KEYPOINT_DIM), dtype=np.float32)
    probabilities = predict_proba(bundle, zeros)
    index = int(np.argmax(probabilities))
    return SmokeReport(
        model_name=name,
        classes=len(bundle.labels),
        parameters=int(bundle.model.count_params()),
        input_shape=(1, SEQUENCE_LENGTH, KEYPOINT_DIM),
        prob_sum=float(np.sum(probabilities)),
        top_label=bundle.labels[index],
        top_confidence=float(probabilities[index]),
        labels=bundle.labels,
    )


def format_smoke_report(report: SmokeReport) -> str:
    return "\n".join(
        [
            "smoke: ok",
            f"model: {report.model_name}",
            f"classes: {report.classes}",
            f"parameters: {report.parameters}",
            f"input: zeros {report.input_shape}",
            f"prob_sum: {report.prob_sum:.6f}",
            f"top_label: {report.top_label}",
            f"top_confidence: {report.top_confidence:.6f}",
            f"labels: {', '.join(report.labels)}",
            "note: top_label is the argmax on a synthetic zero sequence, not a measured accuracy.",
        ]
    )
