"""Fetch trained weights from GitHub Release ``model-weights-v1`` when they are missing.

The release is published at
https://github.com/sc-lacanlale/Codewarts/releases/tag/model-weights-v1
The clone still includes the same ``.h5``, scaler, and label files under ``models/``.
This module downloads only what is absent. After a history purge removes those
files, a fresh clone gets them back from the release.

``sc-lacanlale/fsl-sign-language-translator`` is tried first so the URL keeps
working after the planned rename. Until that rename, the download uses
``sc-lacanlale/Codewarts``.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from fsl.paths import repo_root

RELEASE_TAG = "model-weights-v1"
RELEASE_REPOS: tuple[str, ...] = (
    "sc-lacanlale/fsl-sign-language-translator",
    "sc-lacanlale/Codewarts",
)

# Local path relative to the repo root -> asset name on the release.
WEIGHT_ASSETS: dict[str, str] = {
    "models/fsl15/model.h5": "fsl15-model.h5",
    "models/fsl15/best.h5": "fsl15-best.h5",
    "models/fsl15/scaler.pkl": "fsl15-scaler.pkl",
    "models/fsl15/labels.npy": "fsl15-labels.npy",
    "models/fsl11/model.h5": "fsl11-model.h5",
    "models/fsl11/scaler.pkl": "fsl11-scaler.pkl",
    "models/fsl11/labels.npy": "fsl11-labels.npy",
}


def release_asset_url(repo: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{asset_name}"


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "fsl-sign-language-translator"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if payload.lstrip().startswith(b"<!DOCTYPE") or payload.lstrip().startswith(b"<html"):
        raise RuntimeError(f"{url} returned HTML instead of a weight file")
    if len(payload) < 64:
        raise RuntimeError(f"{url} returned a {len(payload)}-byte body")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(dest)


def ensure_weights(*, force: bool = False) -> int:
    """Download any missing weight files. Return 0 when every file is present."""
    root = repo_root()
    missing: list[tuple[Path, str]] = []
    for relative, asset_name in WEIGHT_ASSETS.items():
        path = root / relative
        if path.is_file() and path.stat().st_size > 0 and not force:
            print(f"present  {relative}  ({path.stat().st_size} bytes)")
            continue
        missing.append((path, asset_name))

    if not missing:
        print("All model files are already on disk.")
        return 0

    failed: list[str] = []
    for path, asset_name in missing:
        saved = False
        errors: list[str] = []
        for repo in RELEASE_REPOS:
            url = release_asset_url(repo, asset_name)
            try:
                print(f"download {url}")
                _download(url, path)
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                errors.append(f"{repo}: {exc}")
                continue
            print(f"saved    {path.relative_to(root)}  ({path.stat().st_size} bytes)")
            saved = True
            break
        if not saved:
            failed.append(f"{asset_name}\n  " + "\n  ".join(errors))

    if failed:
        print("Could not download these release assets:")
        print("\n".join(failed))
        print(
            f"Create a GitHub Release tagged {RELEASE_TAG} and upload the asset names "
            "listed in fsl.weights.WEIGHT_ASSETS. Until that release exists, use the "
            "copies already committed under models/."
        )
        return 1
    return 0
