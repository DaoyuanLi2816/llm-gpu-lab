"""Merge a LoRA adapter into its base model and save as a HF directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)


def merge_lora_into_base(
    base_model: str,
    adapter_path: str,
    output_dir: str | Path,
    dtype: Optional[str] = "float16",
) -> Path:
    """Load `base_model`, attach `adapter_path`, merge, and save to disk.

    Returns the path to the merged-model directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype or "float16", torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)

    adapter_full = ROOT / adapter_path if not Path(adapter_path).is_absolute() else Path(adapter_path)
    merged = PeftModel.from_pretrained(base, str(adapter_full))
    merged = merged.merge_and_unload()

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    ensure_dir(out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Merged model saved to %s", out_dir)
    return out_dir
