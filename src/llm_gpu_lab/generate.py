"""Load a TinyGPT checkpoint + tokenizer and generate text for a list of prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir
from llm_gpu_lab.tokenization import load_tokenizer

logger = get_logger(__name__)


def _load_tinygpt_from_checkpoint(checkpoint: Path):
    import torch

    from llm_gpu_lab.models import TinyGPT, TinyGPTConfig

    ckpt_dir = checkpoint.parent
    config_path = ckpt_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected {config_path} alongside the weights file. Did you run pretrain?"
        )
    side = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = TinyGPTConfig(**side["model_cfg"], vocab_size=side["vocab_size"])
    model = TinyGPT(model_cfg)

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint))
    else:
        state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)

    tokenizer_path = side.get("tokenizer_path")
    if tokenizer_path is None:
        raise RuntimeError("Checkpoint sidecar missing 'tokenizer_path'.")
    abs_tok = ROOT / tokenizer_path if not Path(tokenizer_path).is_absolute() else Path(tokenizer_path)
    tokenizer = load_tokenizer(abs_tok)
    return model, tokenizer, model_cfg


def generate_from_checkpoint(
    checkpoint: Path,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int = 1337,
    out_path: Path | None = None,
) -> Dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    model, tokenizer, model_cfg = _load_tinygpt_from_checkpoint(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>") or 0

    samples: List[Dict[str, Any]] = []
    total_tokens = 0
    total_dt = 0.0
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        if not ids:
            ids = [bos_id]
        if len(ids) > model_cfg.block_size:
            ids = ids[-model_cfg.block_size :]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_id,
            )
        dt = time.time() - t0
        new_ids = out[0].tolist()[len(ids):]
        cont = tokenizer.decode(new_ids)
        samples.append(
            {
                "prompt": prompt,
                "continuation": cont,
                "new_token_count": len(new_ids),
                "latency_s": round(dt, 4),
            }
        )
        total_tokens += len(new_ids)
        total_dt += dt
    result = {
        "checkpoint": str(checkpoint),
        "device": device,
        "tokens_per_s": round(total_tokens / total_dt, 2) if total_dt > 0 else None,
        "n_prompts": len(prompts),
        "samples": samples,
    }
    if out_path is not None:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Generation samples → %s", out_path)
    return result
