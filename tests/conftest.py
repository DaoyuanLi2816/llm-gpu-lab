"""pytest config: auto-skip GPU tests when CUDA isn't available."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 - pytest hook signature
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
