"""Landmark vector assembly and the Holistic task-file fetch."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from fsl.landmarks import (
    HAND_LANDMARKS,
    HOLISTIC_MODEL_URL,
    KEYPOINT_DIM,
    POSE_DIM,
    POSE_LANDMARKS,
    assemble_keypoints,
    ensure_holistic_model,
)


def test_full_body_vector_is_258() -> None:
    pose = np.arange(POSE_LANDMARKS * 4, dtype=np.float32).reshape(POSE_LANDMARKS, 4)
    left = np.arange(HAND_LANDMARKS * 3, dtype=np.float32).reshape(HAND_LANDMARKS, 3)
    right = np.full((HAND_LANDMARKS, 3), 0.5, dtype=np.float32)

    vector = assemble_keypoints(pose, left, right)

    assert vector.shape == (KEYPOINT_DIM,)
    assert vector.dtype == np.float32
    assert np.array_equal(vector[:POSE_DIM], pose.reshape(-1))
    assert np.array_equal(vector[POSE_DIM : POSE_DIM + 63], left.reshape(-1))
    assert np.allclose(vector[-63:], 0.5)


def test_missing_body_parts_are_zeros() -> None:
    vector = assemble_keypoints(None, None, None)
    assert vector.shape == (KEYPOINT_DIM,)
    assert np.all(vector == 0)


def test_pose_without_visibility_is_marked_present() -> None:
    pose = np.full((POSE_LANDMARKS, 3), 0.25, dtype=np.float32)
    vector = assemble_keypoints(pose, None, None)
    assert np.allclose(vector[:99], 0.25)
    assert np.allclose(vector[99:POSE_DIM], 1.0)
    assert np.all(vector[POSE_DIM:] == 0)


def test_short_hand_is_copied_into_the_prefix() -> None:
    left = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vector = assemble_keypoints(None, left, None)
    assert np.allclose(vector[POSE_DIM : POSE_DIM + 3], left)
    assert np.all(vector[POSE_DIM + 3 :] == 0)


def test_official_holistic_url() -> None:
    assert HOLISTIC_MODEL_URL.startswith(
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    )
    assert HOLISTIC_MODEL_URL.endswith("holistic_landmarker.task")


def test_existing_task_file_is_not_downloaded(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "holistic_landmarker.task"
    dest.write_bytes(b"cached")

    def fail(*_args, **_kwargs):
        raise AssertionError("urlretrieve should not run when the file exists")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)
    assert ensure_holistic_model(dest) == dest


def test_missing_task_file_is_downloaded(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "models" / "holistic_landmarker.task"

    def fake_retrieve(url: str, target: str) -> None:
        assert url == HOLISTIC_MODEL_URL
        Path(target).write_bytes(b"x" * 1_000_000)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    saved = ensure_holistic_model(dest)
    assert saved == dest
    assert saved.stat().st_size == 1_000_000
