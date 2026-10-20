"""MediaPipe Holistic landmarks as a fixed 258-float vector, plus skeleton drawing."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional app dependency
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
        RunningMode,
    )
except ImportError:  # pragma: no cover - optional app dependency
    mp = None
    base_options_module = None
    HolisticLandmarker = None
    HolisticLandmarkerOptions = None
    RunningMode = None

from fsl.paths import repo_root

KEYPOINT_DIM = 258
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM = POSE_LANDMARKS * 4
LH_DIM = HAND_LANDMARKS * 3
RH_DIM = HAND_LANDMARKS * 3

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

POSE_JOINT_COLOR = (80, 255, 100)
POSE_LINE_COLOR = (0, 200, 60)
LEFT_HAND_JOINT_COLOR = (80, 255, 100)
LEFT_HAND_LINE_COLOR = (0, 200, 60)
RIGHT_HAND_JOINT_COLOR = (80, 255, 100)
RIGHT_HAND_LINE_COLOR = (0, 200, 60)


@dataclass
class HolisticOutput:
    pose_landmarks: np.ndarray | None
    left_hand_landmarks: np.ndarray | None
    right_hand_landmarks: np.ndarray | None
    keypoints: np.ndarray | None = None


def default_holistic_task_path() -> Path:
    return repo_root() / "models" / "holistic_landmarker.task"


def ensure_holistic_model(dest: Path | None = None) -> Path:
    """Return the Holistic ``.task`` file, downloading it from MediaPipe when missing."""
    target = dest or default_holistic_task_path()
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe holistic landmarker to {target}")
    urllib.request.urlretrieve(HOLISTIC_MODEL_URL, target)
    if not target.is_file() or target.stat().st_size < 1_000_000:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download from {HOLISTIC_MODEL_URL} did not produce a model file at {target}. "
            "Check network access to storage.googleapis.com."
        )
    return target


def _fit_features(source: np.ndarray | None, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=np.float32)
    if source is None:
        return out
    flat = np.asarray(source, dtype=np.float32).reshape(-1)
    count = min(int(flat.size), width)
    if count:
        out[:count] = flat[:count]
    return out


def assemble_keypoints(
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """Build the 258-float frame vector: pose (132) + left hand (63) + right hand (63).

    Missing body parts are zeros. A pose array with xyz only (99 floats) is copied
    into the first 99 slots and the visibility slots are set to 1.
    """
    pose_out = np.zeros(POSE_DIM, dtype=np.float32)
    if pose is not None:
        flat = np.asarray(pose, dtype=np.float32).reshape(-1)
        if flat.size >= POSE_DIM:
            pose_out[:] = flat[:POSE_DIM]
        elif flat.size >= POSE_LANDMARKS * 3:
            pose_out[: flat.size] = flat
            pose_out[flat.size : flat.size + flat.size // 3] = 1.0

    left = _fit_features(left_hand, LH_DIM)
    right = _fit_features(right_hand, RH_DIM)
    return np.concatenate((pose_out, left, right))


def _landmarks_to_array(landmarks, include_visibility: bool = False) -> np.ndarray | None:
    if not landmarks:
        return None
    width = 4 if include_visibility else 3
    array = np.empty((len(landmarks), width), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        array[index, 0] = landmark.x
        array[index, 1] = landmark.y
        array[index, 2] = landmark.z
        if include_visibility:
            array[index, 3] = getattr(landmark, "visibility", 0.0)
    return array


def _draw_landmark_set(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    connections: tuple[tuple[int, int], ...],
    joint_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    line_thickness: int = 2,
    joint_radius: int = 4,
) -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV is required to draw landmarks. Install with: pip install -e '.[app]'"
        )
    if landmarks is None or landmarks.size == 0:
        return
    height, width = image.shape[:2]
    xs = np.clip(landmarks[:, 0], 0.0, 1.0) * width
    ys = np.clip(landmarks[:, 1], 0.0, 1.0) * height
    points = np.stack((xs, ys), axis=1).astype(np.int32)
    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(
                image,
                tuple(points[start]),
                tuple(points[end]),
                line_color,
                line_thickness,
                cv2.LINE_8,
            )
    for x_coord, y_coord in points:
        cv2.circle(image, (int(x_coord), int(y_coord)), joint_radius, joint_color, -1, cv2.LINE_8)


class HolisticTracker:
    """Holistic landmark tracking through the MediaPipe Tasks API."""

    def __init__(self, model_path: Path | None = None, process_width: int = 640) -> None:
        if (
            mp is None
            or HolisticLandmarker is None
            or RunningMode is None
            or base_options_module is None
        ):
            raise ImportError(
                "MediaPipe is required for the webcam tracker. "
                "Install with: pip install -e '.[app]'"
            )
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for the webcam tracker. Install with: pip install -e '.[app]'"
            )

        self._mp = mp
        self._frame_ms = 0
        self._process_width = process_width
        self._rgb_buffer: np.ndarray | None = None
        resolved = ensure_holistic_model(model_path)
        options = HolisticLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(resolved)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)

    def _prepare_rgb(self, bgr_image: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for the webcam tracker. Install with: pip install -e '.[app]'"
            )
        height, width = bgr_image.shape[:2]
        if self._process_width and width > self._process_width:
            scale = self._process_width / width
            new_width = self._process_width
            new_height = max(1, int(height * scale))
            bgr_image = cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if self._rgb_buffer is None or self._rgb_buffer.shape[:2] != bgr_image.shape[:2]:
            self._rgb_buffer = np.empty_like(bgr_image)
        cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        return self._rgb_buffer

    def process(self, bgr_image: np.ndarray) -> HolisticOutput:
        rgb = self._prepare_rgb(bgr_image)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._frame_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ms)

        pose = _landmarks_to_array(result.pose_landmarks, True)
        left_hand = _landmarks_to_array(result.left_hand_landmarks)
        right_hand = _landmarks_to_array(result.right_hand_landmarks)
        keypoints = assemble_keypoints(pose, left_hand, right_hand)
        return HolisticOutput(
            pose_landmarks=pose,
            left_hand_landmarks=left_hand,
            right_hand_landmarks=right_hand,
            keypoints=keypoints,
        )

    def draw_landmarks(
        self,
        rgb_image: np.ndarray,
        output: HolisticOutput,
        inplace: bool = True,
    ) -> np.ndarray:
        """Draw pose and hand bones on an RGB frame."""
        present = (
            output.pose_landmarks is not None
            or output.left_hand_landmarks is not None
            or output.right_hand_landmarks is not None
        )
        if not present:
            return rgb_image
        annotated = rgb_image if inplace else rgb_image.copy()
        _draw_landmark_set(
            annotated, output.pose_landmarks, POSE_CONNECTIONS, POSE_JOINT_COLOR, POSE_LINE_COLOR
        )
        _draw_landmark_set(
            annotated,
            output.left_hand_landmarks,
            HAND_CONNECTIONS,
            LEFT_HAND_JOINT_COLOR,
            LEFT_HAND_LINE_COLOR,
        )
        _draw_landmark_set(
            annotated,
            output.right_hand_landmarks,
            HAND_CONNECTIONS,
            RIGHT_HAND_JOINT_COLOR,
            RIGHT_HAND_LINE_COLOR,
        )
        return annotated

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
