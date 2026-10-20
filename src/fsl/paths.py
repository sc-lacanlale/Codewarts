"""Locate the repository and the on-disk model directory."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the clone root, which contains ``pyproject.toml`` and ``models/``."""
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "models").is_dir():
            return candidate
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file() and (cwd / "models").is_dir():
        return cwd
    raise FileNotFoundError(
        "Could not find the repository root (pyproject.toml and models/). "
        "Run these commands from a clone of the project."
    )


def models_dir() -> Path:
    override = os.environ.get("FSL_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "models"
