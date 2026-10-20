#!/usr/bin/env python3
"""Download LSTM weights from GitHub Release ``model-weights-v1`` when they are missing.

The weights are still in this clone, under ``models/``. This script is the path
a fresh tree uses after those files are removed from git history. The owner has
to upload the release assets first. See ``fsl.weights`` for the exact ``gh`` command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fsl.weights import RELEASE_TAG, ensure_weights  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Fetch model files from the {RELEASE_TAG} GitHub Release."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the local files are already present.",
    )
    args = parser.parse_args(argv)
    return ensure_weights(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
