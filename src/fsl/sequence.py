"""Fixed-length landmark windows and the stable-sign gate used by the live app."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from fsl.landmarks import KEYPOINT_DIM

SEQUENCE_LENGTH = 30
PREDICTION_CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES_TO_RECORD = 10
SIGN_COOLDOWN_FRAMES = 12
MAX_LIVE_SIGNS = 20


def shape_sequence(
    frames: np.ndarray | Sequence[np.ndarray] | deque[np.ndarray],
    length: int = SEQUENCE_LENGTH,
    dim: int = KEYPOINT_DIM,
    *,
    pad: bool = True,
) -> np.ndarray:
    """Return a ``(length, dim)`` float32 window.

    Longer recordings keep the most recent ``length`` frames. Shorter recordings
    are left-padded with zeros when ``pad`` is true, so the newest frame stays
    at the end of the window. The live app waits for a full window and calls
    this with ``pad=False``.
    """
    if isinstance(frames, deque):
        array = (
            np.stack([np.asarray(frame, dtype=np.float32) for frame in frames])
            if len(frames)
            else np.zeros((0, dim), dtype=np.float32)
        )
    else:
        array = np.asarray(frames, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(f"Expected frames with shape (N, {dim}), got {array.shape}")
    if array.shape[1] != dim:
        raise ValueError(f"Expected {dim} features per frame, got {array.shape[1]}")
    if array.shape[0] > length:
        return np.ascontiguousarray(array[-length:], dtype=np.float32)
    if array.shape[0] == length:
        return np.ascontiguousarray(array, dtype=np.float32)
    if not pad:
        raise ValueError(f"Need {length} frames, got {array.shape[0]}")
    pad_rows = np.zeros((length - array.shape[0], dim), dtype=np.float32)
    return np.concatenate((pad_rows, array), axis=0)


def scale_sequence(
    frames: np.ndarray | Sequence[np.ndarray] | deque[np.ndarray], scaler: Any
) -> np.ndarray:
    """Standard-scale each frame and add a batch axis: ``(1, 30, 258)``."""
    window = (
        shape_sequence(frames, pad=False)
        if _frame_count(frames) >= SEQUENCE_LENGTH
        else shape_sequence(frames)
    )
    flat = window.reshape(-1, window.shape[-1])
    scaled = np.asarray(scaler.transform(flat), dtype=np.float32)
    return scaled.reshape(1, window.shape[0], window.shape[1])


def _frame_count(frames: np.ndarray | Sequence[np.ndarray] | deque[np.ndarray]) -> int:
    if isinstance(frames, deque):
        return len(frames)
    array = np.asarray(frames)
    if array.ndim == 0:
        return 0
    return int(array.shape[0])


def append_sign(signs: list[str], sign: str, max_len: int = MAX_LIVE_SIGNS) -> bool:
    """Append ``sign`` unless it repeats the previous gloss. Drop the oldest past ``max_len``."""
    if signs and sign == signs[-1]:
        return False
    signs.append(sign)
    if len(signs) > max_len:
        del signs[:-max_len]
    return True


class StableSignDetector:
    """Record a gloss only after the same label holds for several frames."""

    def __init__(
        self,
        stable_frames: int = STABLE_FRAMES_TO_RECORD,
        cooldown_frames: int = SIGN_COOLDOWN_FRAMES,
    ) -> None:
        self.stable_frames = stable_frames
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.candidate: str | None = None
        self.streak = 0

    def reset(self) -> None:
        self.cooldown = 0
        self.candidate = None
        self.streak = 0

    def update(self, sign: str | None, confidence: float, threshold: float) -> str | None:
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if not sign or confidence < threshold:
            self.candidate = None
            self.streak = 0
            return None
        if sign == self.candidate:
            self.streak += 1
        else:
            self.candidate = sign
            self.streak = 1
        if self.streak >= self.stable_frames:
            recorded = self.candidate
            self.candidate = None
            self.streak = 0
            self.cooldown = self.cooldown_frames
            return recorded
        return None
