"""GGUF export pipeline that wraps llama.cpp's conversion script.

Steps:

1. Merge LoRA adapter into the base model and save as a standard HF directory.
2. Run ``external/llama.cpp/convert_hf_to_gguf.py`` against the merged dir.
3. If a ``llama-quantize`` binary is available, quantize to ``cfg.quant_type``.

Each failure mode appends to ``results/<gpu>/limitations.md`` instead of
crashing so the workflow can still produce a final report.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_gpu_lab.config import GGUFExportConfig
from llm_gpu_lab.export.merge_lora import merge_lora_into_base
from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)


def _append_limitation(limitations_path: Path, msg: str) -> None:
    ensure_dir(limitations_path.parent)
    header = "# Local limitations\n\nAuto-generated. Each entry records why a step degraded or was skipped.\n\n"
    body = msg.rstrip() + "\n"
    if limitations_path.is_file():
        existing = limitations_path.read_text(encoding="utf-8")
        limitations_path.write_text(existing + body, encoding="utf-8")
    else:
        limitations_path.write_text(header + body, encoding="utf-8")


def _find_llamacpp_python(llamacpp_dir: Path) -> Optional[Path]:
    candidate = llamacpp_dir / "convert_hf_to_gguf.py"
    if candidate.is_file():
        return candidate
    return None


def _find_binary(llamacpp_dir: Path, name: str) -> Optional[Path]:
    for sub in ("build/bin", "build", "."):
        cand = llamacpp_dir / sub / name
        if cand.is_file():
            return cand
        cand_exe = llamacpp_dir / sub / f"{name}.exe"
        if cand_exe.is_file():
            return cand_exe
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    return Path(found) if found else None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    logger.info("Running: %s", " ".join(map(str, cmd)))
    t0 = time.time()
    res = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": list(map(str, cmd)),
        "returncode": res.returncode,
        "stdout_tail": res.stdout[-2000:] if res.stdout else "",
        "stderr_tail": res.stderr[-2000:] if res.stderr else "",
        "duration_s": round(time.time() - t0, 2),
    }


def run_gguf_export(cfg: GGUFExportConfig) -> Dict[str, Any]:
    limitations_path = ROOT / cfg.limitations_path if not Path(cfg.limitations_path).is_absolute() else Path(cfg.limitations_path)
    llamacpp_dir = ROOT / cfg.llamacpp_dir if not Path(cfg.llamacpp_dir).is_absolute() else Path(cfg.llamacpp_dir)
    gguf_out_dir = ROOT / cfg.gguf_out_dir if not Path(cfg.gguf_out_dir).is_absolute() else Path(cfg.gguf_out_dir)
    merged_dir = ROOT / cfg.merged_dir if not Path(cfg.merged_dir).is_absolute() else Path(cfg.merged_dir)
    ensure_dir(gguf_out_dir)

    result: Dict[str, Any] = {
        "name": cfg.name,
        "base_model": cfg.base_model,
        "adapter_path": cfg.adapter_path,
        "merged_dir": str(merged_dir),
        "gguf_out_dir": str(gguf_out_dir),
        "quant_type": cfg.quant_type,
        "steps": [],
        "success": False,
    }

    # 1. Merge adapter if present, otherwise use base directly via HF download.
    if cfg.adapter_path:
        try:
            merge_lora_into_base(
                base_model=cfg.base_model,
                adapter_path=cfg.adapter_path,
                output_dir=merged_dir,
                dtype="float16",
            )
            result["steps"].append({"name": "merge_lora", "ok": True, "merged_dir": str(merged_dir)})
        except Exception as exc:
            result["steps"].append({"name": "merge_lora", "ok": False, "error": repr(exc)})
            _append_limitation(limitations_path, f"- GGUF merge_lora failed: `{exc!r}`")
            return result
    else:
        result["steps"].append({"name": "merge_lora", "ok": True, "note": "skipped — no adapter"})
        merged_dir = Path(cfg.base_model)  # let convert use HF repo id

    # 2. Locate convert_hf_to_gguf.py
    convert_py = _find_llamacpp_python(llamacpp_dir)
    if convert_py is None:
        msg = (
            f"`convert_hf_to_gguf.py` not found under {llamacpp_dir}. "
            "Run `bash scripts/setup_llamacpp.sh` first."
        )
        result["steps"].append({"name": "convert", "ok": False, "error": msg})
        _append_limitation(limitations_path, f"- GGUF export skipped: {msg}")
        return result

    # 3. Convert to f16 GGUF
    f16_path = gguf_out_dir / f"{Path(str(merged_dir)).name}.f16.gguf"
    convert_cmd = [
        sys.executable,
        str(convert_py),
        str(merged_dir),
        "--outfile",
        str(f16_path),
        "--outtype",
        "f16",
    ]
    convert_res = _run(convert_cmd, cwd=llamacpp_dir)
    convert_res["name"] = "convert_to_f16_gguf"
    result["steps"].append(convert_res)
    if convert_res["returncode"] != 0:
        _append_limitation(
            limitations_path,
            f"- GGUF convert failed: rc={convert_res['returncode']}\n  stderr: ```\n{convert_res['stderr_tail']}\n```",
        )
        return result

    # 4. Quantize
    quant_bin = _find_binary(llamacpp_dir, "llama-quantize")
    if quant_bin is None:
        _append_limitation(
            limitations_path,
            "- llama-quantize binary not found; left model at f16. Build llama.cpp with `make` first.",
        )
        result["success"] = True
        result["gguf_files"] = [str(f16_path)]
        return result

    quant_path = gguf_out_dir / f"{Path(str(merged_dir)).name}.{cfg.quant_type}.gguf"
    quant_cmd = [str(quant_bin), str(f16_path), str(quant_path), cfg.quant_type]
    quant_res = _run(quant_cmd)
    quant_res["name"] = "quantize"
    result["steps"].append(quant_res)
    if quant_res["returncode"] != 0:
        _append_limitation(
            limitations_path,
            f"- GGUF quantize failed: rc={quant_res['returncode']}\n  stderr: ```\n{quant_res['stderr_tail']}\n```",
        )
        result["success"] = True  # f16 succeeded
        result["gguf_files"] = [str(f16_path)]
        return result

    result["success"] = True
    result["gguf_files"] = [str(f16_path), str(quant_path)]
    return result
