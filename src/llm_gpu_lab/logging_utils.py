"""Rich-backed logger that respects LLM_GPU_LAB_LOG_LEVEL."""

from __future__ import annotations

import logging
import os
from typing import Optional

from rich.logging import RichHandler

_INITIALIZED = False


def get_logger(name: str = "llm_gpu_lab", level: Optional[str] = None) -> logging.Logger:
    global _INITIALIZED
    if not _INITIALIZED:
        log_level = level or os.environ.get("LLM_GPU_LAB_LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=log_level.upper(),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=True)],
        )
        _INITIALIZED = True
    return logging.getLogger(name)
