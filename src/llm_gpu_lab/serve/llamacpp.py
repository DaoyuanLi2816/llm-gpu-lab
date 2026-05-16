"""Thin wrapper around `llama-server` for local GGUF inference."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _find_server(llamacpp_dir: Optional[Path]) -> Optional[Path]:
    for binary in ("llama-server", "llama-server.exe"):
        if llamacpp_dir is not None:
            for sub in ("build/bin", "build", "."):
                cand = llamacpp_dir / sub / binary
                if cand.is_file():
                    return cand
        on_path = shutil.which(binary)
        if on_path:
            return Path(on_path)
    return None


def serve_llamacpp(
    model: str,
    port: int = 8080,
    n_ctx: int = 4096,
    n_gpu_layers: int = 999,
    llamacpp_dir: Optional[str] = "external/llama.cpp",
    extra_args: Optional[list[str]] = None,
) -> int:
    """Exec `llama-server` and return its exit code.

    Raises:
        FileNotFoundError: if no llama-server binary is reachable.
    """
    server = _find_server(Path(llamacpp_dir) if llamacpp_dir else None)
    if server is None:
        raise FileNotFoundError(
            "llama-server binary not found. Run `bash scripts/setup_llamacpp.sh` "
            "or install llama.cpp manually and put `llama-server` on PATH."
        )
    if not Path(model).is_file():
        raise FileNotFoundError(f"Model file does not exist: {model}")

    cmd: list[str] = [
        str(server),
        "-m",
        str(model),
        "-c",
        str(n_ctx),
        "--port",
        str(port),
        "-ngl",
        str(n_gpu_layers),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[llamacpp] launching: {' '.join(cmd)}")
    return subprocess.call(cmd, env=os.environ.copy())
