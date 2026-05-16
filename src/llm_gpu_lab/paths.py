"""Canonical project paths, anchored to the repository root."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory.

    Resolves by walking up from this file until a `pyproject.toml` is found.
    Falls back to the current working directory so the package keeps working
    when installed in `site-packages`.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(os.getcwd())


ROOT: Path = repo_root()
ARTIFACTS: Path = ROOT / "artifacts"
RESULTS: Path = ROOT / "results"
RESULTS_RTX4080: Path = RESULTS / "rtx4080"
EXAMPLES: Path = ROOT / "examples"
CONFIGS: Path = ROOT / "configs"
EXTERNAL: Path = ROOT / "external"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
