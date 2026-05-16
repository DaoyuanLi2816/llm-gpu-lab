"""Environment / GPU diagnostics."""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.metadata as _im
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_gpu_lab.paths import ROOT, ensure_dir

PROBE_PACKAGES: List[str] = [
    "torch",
    "numpy",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "safetensors",
    "pydantic",
    "typer",
    "rich",
    "pandas",
    "matplotlib",
    "jinja2",
    "psutil",
    "pynvml",
    "huggingface_hub",
    "lm_eval",
    "tqdm",
    "pyyaml",
]


def _safe_run(cmd: List[str], timeout: int = 8) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git_commit() -> Optional[str]:
    git = shutil.which("git")
    if git is None:
        return None
    return _safe_run([git, "rev-parse", "HEAD"])


_PACKAGE_DIST_ALIASES = {
    "pynvml": "nvidia-ml-py",
    "yaml": "pyyaml",
    "pyyaml": "pyyaml",
}


def _package_version(name: str) -> Optional[str]:
    """Look up a distribution version. Try a few common aliases."""
    candidates = [name, name.replace("_", "-"), _PACKAGE_DIST_ALIASES.get(name, name)]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _im.version(candidate)
        except _im.PackageNotFoundError:
            continue
    return None


def _can_import(name: str) -> bool:
    """Return True if `name` is importable.

    We use ``importlib.util.find_spec`` rather than actually importing the
    module. Some optional dependencies (``datasets`` pulls in ``pyarrow``)
    have C extensions that segfault on certain Windows + OneDrive paths
    when their submodules are imported eagerly. ``find_spec`` only walks the
    finder/loader chain without running the module body, so it stays safe.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None


def _torch_block() -> Dict[str, Any]:
    info: Dict[str, Any] = {"installed": False}
    try:
        import torch
    except Exception as exc:
        info["import_error"] = repr(exc)
        return info
    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_runtime_version"] = torch.version.cuda
    info["cudnn_version"] = (
        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    )
    info["bf16_supported"] = bool(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    info["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(props.total_memory),
                    "total_memory_gib": round(props.total_memory / (1024**3), 3),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        info["devices"] = devices
    return info


def _nvml_block() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import pynvml
    except Exception as exc:
        info["error"] = repr(exc)
        return info
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        info["error"] = f"nvmlInit failed: {exc!r}"
        return info
    info["available"] = True
    try:
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode() if isinstance(
            pynvml.nvmlSystemGetDriverVersion(), bytes
        ) else pynvml.nvmlSystemGetDriverVersion()
    except Exception as exc:
        info["driver_version_error"] = repr(exc)
    try:
        count = pynvml.nvmlDeviceGetCount()
        info["device_count"] = int(count)
        devices = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            devices.append(
                {
                    "index": i,
                    "name": name,
                    "total_bytes": int(mem.total),
                    "free_bytes": int(mem.free),
                    "used_bytes": int(mem.used),
                    "total_gib": round(mem.total / (1024**3), 3),
                    "free_gib": round(mem.free / (1024**3), 3),
                    "used_gib": round(mem.used / (1024**3), 3),
                }
            )
        info["devices"] = devices
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return info


def collect_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "cwd": str(Path.cwd()),
        "repo_root": str(ROOT),
        "git_commit": _git_commit(),
        "packages": {pkg: _package_version(pkg.replace("_", "-")) for pkg in PROBE_PACKAGES},
        "imports": {
            "torch": _can_import("torch"),
            "transformers": _can_import("transformers"),
            "peft": _can_import("peft"),
            "trl": _can_import("trl"),
            "bitsandbytes": _can_import("bitsandbytes"),
            "datasets": _can_import("datasets"),
            "tokenizers": _can_import("tokenizers"),
            "lm_eval": _can_import("lm_eval"),
        },
        "torch": _torch_block(),
        "nvml": _nvml_block(),
        "nvidia_smi": _safe_run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used", "--format=csv"]),
        "llama_cpp": _detect_llama_cpp(),
    }
    return info


def _detect_llama_cpp() -> Dict[str, Any]:
    candidates_dirs = [
        ROOT / "external" / "llama.cpp",
        ROOT / ".cache" / "llama.cpp",
    ]
    info: Dict[str, Any] = {"found": False}
    for d in candidates_dirs:
        if d.exists():
            info["dir"] = str(d)
            info["found"] = True
            for binary in ("llama-cli", "llama-server", "llama-quantize"):
                exe = shutil.which(binary, path=str(d / "build" / "bin"))
                if exe is None:
                    # Windows binary names
                    exe = shutil.which(binary + ".exe", path=str(d / "build" / "bin"))
                if exe:
                    info[binary] = exe
            converter = d / "convert_hf_to_gguf.py"
            if converter.is_file():
                info["convert_hf_to_gguf"] = str(converter)
            break
    # also probe PATH
    for binary in ("llama-cli", "llama-server", "llama-quantize"):
        exe = shutil.which(binary)
        if exe and binary not in info:
            info[binary] = exe
    return info


def write_environment(out_path: str | Path) -> Path:
    info = collect_environment()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return out_path
