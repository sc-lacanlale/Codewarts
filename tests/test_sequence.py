"""Window shaping, scaling, and the stable-sign gate."""

from __future__ import annotations

from collections import deque

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from fsl.landmarks import KEYPOINT_DIM
from fsl.paths import models_dir
from fsl.sequence import (
    SEQUENCE_LENGTH,
    StableSignDetector,
    append_sign,
    scale_sequence,
    shape_sequence,
)


def test_exact_window_is_unchanged() -> None:
    frames = np.arange(SEQUENCE_LENGTH * KEYPOINT_DIM, dtype=np.float32).reshape(
        SEQUENCE_LENGTH, KEYPOINT_DIM
    )
    shaped = shape_sequence(frames, pad=False)
    assert shaped.shape == (SEQUENCE_LENGTH, KEYPOINT_DIM)
    assert np.array_equal(shaped, frames)


def test_long_window_keeps_the_newest_frames() -> None:
    frames = np.arange(40 * KEYPOINT_DIM, dtype=np.float32).reshape(40, KEYPOINT_DIM)
    shaped = shape_sequence(frames)
    assert np.array_equal(shaped, frames[-SEQUENCE_LENGTH:])


def test_short_window_is_left_padded() -> None:
    frames = np.ones((4, KEYPOINT_DIM), dtype=np.float32)
    shaped = shape_sequence(frames)
    assert shaped.shape == (SEQUENCE_LENGTH, KEYPOINT_DIM)
    assert np.all(shaped[: SEQUENCE_LENGTH - 4] == 0)
    assert np.all(shaped[SEQUENCE_LENGTH - 4 :] == 1)


def test_short_window_without_padding_raises() -> None:
    with pytest.raises(ValueError, match="Need 30 frames"):
        shape_sequence(np.ones((4, KEYPOINT_DIM), dtype=np.float32), pad=False)


def test_wrong_feature_width_raises() -> None:
    with pytest.raises(ValueError, match="258"):
        shape_sequence(np.ones((SEQUENCE_LENGTH, 10), dtype=np.float32))


def test_deque_keeps_the_tail() -> None:
    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    for index in range(35):
        window.append(np.full(KEYPOINT_DIM, index, dtype=np.float32))
    shaped = shape_sequence(window, pad=False)
    assert shaped.shape == (SEQUENCE_LENGTH, KEYPOINT_DIM)
    assert shaped[0, 0] == 5
    assert shaped[-1, 0] == 34


def test_scale_sequence_matches_sklearn_and_adds_a_batch_axis() -> None:
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(SEQUENCE_LENGTH, KEYPOINT_DIM)).astype(np.float32)
    scaler = StandardScaler().fit(frames)
    batch = scale_sequence(frames, scaler)
    expected = scaler.transform(frames).astype(np.float32)
    assert batch.shape == (1, SEQUENCE_LENGTH, KEYPOINT_DIM)
    assert np.allclose(batch[0], expected)


def test_shipped_scalers_are_258_features_and_not_duplicates() -> None:
    shipped = joblib.load(models_dir() / "fsl15" / "scaler.pkl")
    legacy = joblib.load(models_dir() / "fsl11" / "scaler.pkl")
    assert int(shipped.n_features_in_) == KEYPOINT_DIM
    assert int(legacy.n_features_in_) == KEYPOINT_DIM
    assert not np.allclose(shipped.mean_, legacy.mean_)

    batch = scale_sequence(np.zeros((SEQUENCE_LENGTH, KEYPOINT_DIM), dtype=np.float32), shipped)
    assert batch.shape == (1, SEQUENCE_LENGTH, KEYPOINT_DIM)
    assert np.isfinite(batch).all()


def test_append_sign_drops_repeats_and_old_glosses() -> None:
    signs: list[str] = []
    assert append_sign(signs, "YES", max_len=2) is True
    assert append_sign(signs, "YES", max_len=2) is False
    assert append_sign(signs, "NO", max_len=2) is True
    assert append_sign(signs, "ONE", max_len=2) is True
    assert signs == ["NO", "ONE"]


def test_stable_detector_waits_then_cools_down() -> None:
    detector = StableSignDetector(stable_frames=3, cooldown_frames=2)
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) == "YES"
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) is None
    assert detector.update("YES", 0.95, 0.7) == "YES"


def test_low_confidence_and_label_changes_reset_the_streak() -> None:
    detector = StableSignDetector(stable_frames=2, cooldown_frames=0)
    assert detector.update("YES", 0.99, 0.7) is None
    assert detector.update("YES", 0.1, 0.7) is None
    assert detector.update("NO", 0.99, 0.7) is None
    assert detector.update("NO", 0.99, 0.7) == "NO"
