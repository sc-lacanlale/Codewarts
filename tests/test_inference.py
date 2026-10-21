"""End-to-end load and forward pass. No camera, no accuracy claim."""

from __future__ import annotations

import subprocess
import sys

from fsl.app import main
from fsl.labels import ENGLISH_KEYS, LEGACY_11_KEYS
from fsl.model import smoke_inference
from fsl.weights import ensure_weights


def test_weights_script_reports_the_committed_files() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/download_weights.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "already on disk" in completed.stdout
    assert ensure_weights() == 0


def test_smoke_cli_loads_the_15_class_model(capsys) -> None:
    assert main(["--smoke"]) == 0
    output = capsys.readouterr().out
    assert "smoke: ok" in output
    assert "classes: 15" in output
    assert "not a measured accuracy" in output
    top_line = next(line for line in output.splitlines() if line.startswith("top_label: "))
    assert top_line.split(": ", 1)[1] in ENGLISH_KEYS


def test_legacy_11_class_checkpoint_infers() -> None:
    report = smoke_inference("fsl11")
    assert report.classes == 11
    assert report.labels == LEGACY_11_KEYS
    assert report.input_shape == (1, 30, 258)
    assert report.parameters > 0
    assert abs(report.prob_sum - 1.0) < 1e-3
    assert 0.0 < report.top_confidence <= 1.0
    assert report.top_label in LEGACY_11_KEYS
