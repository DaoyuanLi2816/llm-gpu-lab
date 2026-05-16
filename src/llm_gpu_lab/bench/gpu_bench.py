"""GPU micro-benchmark.

We measure throughput on three operations that matter for LLM workloads:

1. matrix multiply (FP16 / BF16 / FP32)
2. softmax + cross-entropy (training-style backward step on dummy data)
3. token generation latency with the tiny GPT model

Everything runs locally with synthetic tensors so no dataset or model
download is required. Results are persisted as a small JSON file the HTML
report ingests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ensure_dir

logger = get_logger(__name__)


def _maybe_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bench_matmul(torch, device: str, sizes: List[int]) -> List[Dict[str, Any]]:
    """Time `iters` matmuls per dtype, scaling iters up for small matrices."""
    out: List[Dict[str, Any]] = []
    for size in sizes:
        # smaller matrices need more iters or `dt` underflows
        base_iters = 64 if size <= 512 else (32 if size <= 1024 else 16)
        for dtype_name, dtype in [
            ("fp32", torch.float32),
            ("fp16", torch.float16 if device == "cuda" else None),
            ("bf16", torch.bfloat16 if device == "cuda" else None),
        ]:
            if dtype is None:
                out.append({"size": size, "dtype": dtype_name, "skipped": True})
                continue
            try:
                a = torch.randn(size, size, dtype=dtype, device=device)
                b = torch.randn(size, size, dtype=dtype, device=device)
                # warm up
                for _ in range(4):
                    _ = a @ b
                if device == "cuda":
                    torch.cuda.synchronize()
                iters = base_iters
                # If the first timed run is still under 2 ms (very fast),
                # multiply iters until we get a measurable duration.
                while True:
                    t0 = time.time()
                    for _ in range(iters):
                        _ = a @ b
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dt = time.time() - t0
                    if dt >= 0.003 or iters >= 4096:
                        break
                    iters *= 2
                flops = 2 * size * size * size * iters
                tflops = flops / dt / 1e12
                out.append(
                    {
                        "size": size,
                        "dtype": dtype_name,
                        "iters": iters,
                        "duration_s": round(dt, 4),
                        "tflops": round(tflops, 3),
                    }
                )
            except Exception as exc:
                out.append({"size": size, "dtype": dtype_name, "error": repr(exc)})
    return out


def _bench_train_step(torch, device: str, sizes: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hidden in sizes:
        vocab = 4096
        batch = 16
        try:
            x = torch.randn(batch, hidden, device=device, requires_grad=True)
            w = torch.randn(hidden, vocab, device=device, requires_grad=True)
            y = torch.randint(0, vocab, (batch,), device=device)
            for _ in range(2):
                logits = x @ w
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                x.grad = None
                w.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            iters = 16
            for _ in range(iters):
                logits = x @ w
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                x.grad = None
                w.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            out.append(
                {
                    "hidden": hidden,
                    "vocab": vocab,
                    "batch": batch,
                    "iters": iters,
                    "duration_s": round(dt, 4),
                    "steps_per_s": round(iters / dt, 2),
                }
            )
        except Exception as exc:
            out.append({"hidden": hidden, "error": repr(exc)})
    return out


def _bench_tiny_gen(torch, device: str) -> Dict[str, Any]:
    from llm_gpu_lab.models import TinyGPT, TinyGPTConfig

    cfg = TinyGPTConfig(vocab_size=2048, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = TinyGPT(cfg).to(device)
    model.eval()
    idx = torch.zeros((1, 4), dtype=torch.long, device=device)
    # warm
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=8)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    new_tokens = 64
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=new_tokens, top_k=20)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {
        "model": "tiny_gpt",
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "new_tokens": new_tokens,
        "duration_s": round(dt, 4),
        "tokens_per_s": round(new_tokens / dt, 2),
        "final_seq_len": int(out.size(1)),
    }


def run_gpu_bench(out_path: str | Path) -> Dict[str, Any]:
    torch = _maybe_torch()
    if torch is None:
        result = {
            "torch_available": False,
            "note": "Install torch to run GPU benchmarks.",
        }
        out = Path(out_path)
        ensure_dir(out.parent)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    device = _device(torch)
    logger.info("Running GPU benchmark on device=%s", device)
    matmul_sizes = [512, 1024, 2048] if device == "cuda" else [256, 512]
    train_sizes = [256, 512, 1024] if device == "cuda" else [128, 256]

    result: Dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "device": device,
        "matmul": _bench_matmul(torch, device, matmul_sizes),
        "train_step": _bench_train_step(torch, device, train_sizes),
        "tiny_gen": _bench_tiny_gen(torch, device),
    }
    if device == "cuda":
        result["device_name"] = torch.cuda.get_device_name(0)
        result["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

    out = Path(out_path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("GPU benchmark JSON → %s", out)
    return result
