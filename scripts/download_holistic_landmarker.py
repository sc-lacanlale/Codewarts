#!/usr/bin/env python3
"""Download MediaPipe's official Holistic landmarker into models/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsl.landmarks import HOLISTIC_MODEL_URL, ensure_holistic_model  # noqa: E402


def main() -> int:
    path = ensure_holistic_model()
    print(f"Holistic landmarker: {path} ({path.stat().st_size} bytes)")
    print(f"Source: {HOLISTIC_MODEL_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
