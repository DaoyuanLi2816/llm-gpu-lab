"""Pretrain the tiny GPT on a synthetic corpus (or TinyStories)."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from llm_gpu_lab.config import PretrainYAML
from llm_gpu_lab.data.packing import pack_token_ids, split_train_eval
from llm_gpu_lab.data.tiny_corpus import build_tiny_corpus
from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir
from llm_gpu_lab.telemetry import GPUTelemetry
from llm_gpu_lab.tokenization import load_tokenizer, train_tokenizer

logger = get_logger(__name__)
console = Console()


def _select_precision(precision: str) -> Tuple[str, any]:
    """Return (label, torch.dtype) for the autocast dtype."""
    import torch

    if precision == "fp32":
        return "fp32", torch.float32
    if precision == "bf16":
        return "bf16", torch.bfloat16
    if precision == "fp16":
        return "fp16", torch.float16
    # auto
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    return "fp32", torch.float32


def _load_or_train_tokenizer(cfg: PretrainYAML, corpus: List[str]):
    tok_path = Path(cfg.tokenizer.output_path)
    if tok_path.is_file():
        logger.info("Loading existing tokenizer from %s", tok_path)
        return load_tokenizer(tok_path)
    logger.info("Training tokenizer (vocab=%d) → %s", cfg.tokenizer.vocab_size, tok_path)
    return train_tokenizer(
        corpus=corpus,
        vocab_size=cfg.tokenizer.vocab_size,
        output_path=tok_path,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=list(cfg.tokenizer.special_tokens),
    )


def _build_corpus(cfg: PretrainYAML) -> List[str]:
    source = cfg.data.source.lower()
    if source == "synthetic":
        logger.info("Generating synthetic corpus (%d examples, seed=%d)",
                    cfg.data.n_examples, cfg.data.seed)
        return build_tiny_corpus(n_examples=cfg.data.n_examples, seed=cfg.data.seed)
    if source == "tinystories":
        from llm_gpu_lab.data.tinystories import load_tinystories_text

        logger.info("Loading TinyStories (%d examples)", cfg.data.n_examples)
        return list(load_tinystories_text(max_examples=cfg.data.n_examples,
                                          revision=cfg.data.tinystories_revision))
    if source == "local_text":
        if not cfg.data.local_text_path:
            raise ValueError("data.source=local_text but data.local_text_path is empty")
        path = Path(cfg.data.local_text_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]
    raise ValueError(f"Unknown data.source: {cfg.data.source!r}")


def _encode_corpus(tokenizer, corpus: List[str]) -> List[List[int]]:
    encs = tokenizer.encode_batch(corpus)
    return [e.ids for e in encs]


def _save_checkpoint(out_dir: Path, model, cfg: PretrainYAML, step: int, eval_loss: Optional[float]) -> Dict[str, Any]:
    """Persist a checkpoint.

    Always emits a ``final.safetensors`` for the weights plus a JSON sidecar
    describing the model config so generation knows how to rebuild the network.
    """
    import torch
    out_dir = ensure_dir(out_dir)

    sidecar = {
        "model_cfg": cfg.model.model_dump(),
        "vocab_size": cfg.tokenizer.vocab_size,
        "step": step,
        "eval_loss": eval_loss,
        "tokenizer_path": str(cfg.tokenizer.output_path),
    }
    (out_dir / "config.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    if cfg.train.save_safetensors:
        try:
            from safetensors.torch import save_file

            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_file(state, str(out_dir / "final.safetensors"))
            sidecar["weights"] = "final.safetensors"
        except ImportError:
            torch.save(model.state_dict(), out_dir / "final.pt")
            sidecar["weights"] = "final.pt"
    else:
        torch.save(model.state_dict(), out_dir / "final.pt")
        sidecar["weights"] = "final.pt"
    return sidecar


def _lr_schedule(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    # cosine decay to 10% of base after warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _make_batch(data: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator):
    import torch

    if data.shape[0] == 0:
        raise RuntimeError("Empty data tensor; corpus too small to pack a single block.")
    idx = rng.integers(0, data.shape[0], size=batch_size)
    block = data[idx]  # (B, block+1)
    x = torch.from_numpy(block[:, :block_size]).long()
    y = torch.from_numpy(block[:, 1 : block_size + 1]).long()
    return x, y


def run_pretrain(cfg: PretrainYAML) -> Dict[str, Any]:
    import torch
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    out_dir = ROOT / cfg.train.out_dir
    metrics_dir = ROOT / cfg.train.metrics_dir
    ensure_dir(out_dir)
    ensure_dir(metrics_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision_label, autocast_dtype = _select_precision(cfg.train.precision)
    logger.info("Device=%s precision=%s", device, precision_label)

    corpus = _build_corpus(cfg)
    logger.info("Corpus size: %d documents", len(corpus))
    tokenizer = _load_or_train_tokenizer(cfg, corpus)

    encoded = _encode_corpus(tokenizer, corpus)
    bos_id = tokenizer.token_to_id("<bos>") or 0
    packed = pack_token_ids(encoded, block_size=cfg.model.block_size, sep_token_id=bos_id)
    if packed.shape[0] < 4:
        raise RuntimeError(
            f"Only {packed.shape[0]} packed blocks — make data.n_examples or block_size smaller."
        )
    train_pack, eval_pack = split_train_eval(packed, eval_fraction=0.1)
    logger.info("Packed: train=%d eval=%d block_size=%d",
                train_pack.shape[0], eval_pack.shape[0], cfg.model.block_size)

    from llm_gpu_lab.models import TinyGPT, TinyGPTConfig

    model_cfg = TinyGPTConfig(
        vocab_size=cfg.tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
        tie_weights=cfg.model.tie_weights,
    )
    model = TinyGPT(model_cfg).to(device)
    n_params = model.num_params()
    logger.info("Model: layers=%d heads=%d embd=%d params=%.2fM",
                cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd, n_params / 1e6)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
    )

    rng = np.random.default_rng(cfg.train.seed)

    use_scaler = device == "cuda" and precision_label == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=autocast_dtype)
        if device == "cuda" and precision_label in ("bf16", "fp16")
        else nullcontext()
    )

    metrics_path = metrics_dir / "pretrain_metrics.jsonl"
    summary_path = metrics_dir / "pretrain_metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    telemetry = GPUTelemetry()
    telemetry.reset_peak()

    train_losses: List[float] = []
    eval_losses: List[Dict[str, Any]] = []
    wall_start = time.time()

    @torch.no_grad()
    def _eval_loss() -> float:
        model.eval()
        losses: List[float] = []
        for _ in range(cfg.train.eval_iters):
            xe, ye = _make_batch(eval_pack, cfg.train.batch_size, cfg.model.block_size, rng)
            xe, ye = xe.to(device), ye.to(device)
            with amp_ctx:
                _, loss_t = model(xe, ye)
            losses.append(float(loss_t.detach().cpu().item()))
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    model.train()
    step = 0
    final_eval = None
    while step < cfg.train.max_steps:
        x, y = _make_batch(train_pack, cfg.train.batch_size, cfg.model.block_size, rng)
        x, y = x.to(device), y.to(device)
        lr = _lr_schedule(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        with amp_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        dt = time.time() - t0
        loss_val = float(loss.detach().cpu().item())
        train_losses.append(loss_val)

        if (step + 1) % cfg.train.log_every == 0 or step == 0:
            snap = telemetry.snapshot()
            logger.info(
                "step %d/%d loss=%.4f lr=%.2e dt_ms=%.1f",
                step + 1, cfg.train.max_steps, loss_val, lr, dt * 1000,
            )
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "loss": loss_val,
                            "lr": lr,
                            "dt_s": dt,
                            "tokens_per_step": cfg.train.batch_size * cfg.model.block_size,
                            "telemetry": snap.to_dict(),
                        }
                    )
                    + "\n"
                )

        if (step + 1) % cfg.train.eval_every == 0 or step == cfg.train.max_steps - 1:
            ev = _eval_loss()
            eval_losses.append({"step": step + 1, "eval_loss": ev})
            logger.info("eval step=%d loss=%.4f", step + 1, ev)
            final_eval = ev

        step += 1

    wall_s = time.time() - wall_start
    sidecar = _save_checkpoint(out_dir, model, cfg, step=step, eval_loss=final_eval)

    summary: Dict[str, Any] = {
        "name": cfg.name,
        "device": device,
        "precision": precision_label,
        "model": {
            "n_layer": cfg.model.n_layer,
            "n_head": cfg.model.n_head,
            "n_embd": cfg.model.n_embd,
            "block_size": cfg.model.block_size,
            "vocab_size": cfg.tokenizer.vocab_size,
            "num_params": n_params,
        },
        "data": {
            "source": cfg.data.source,
            "n_examples": cfg.data.n_examples,
            "packed_train_blocks": int(train_pack.shape[0]),
            "packed_eval_blocks": int(eval_pack.shape[0]),
        },
        "train": {
            "batch_size": cfg.train.batch_size,
            "max_steps": cfg.train.max_steps,
            "lr": cfg.train.lr,
            "grad_clip": cfg.train.grad_clip,
            "warmup_steps": cfg.train.warmup_steps,
        },
        "results": {
            "first_loss": train_losses[0] if train_losses else None,
            "last_loss": train_losses[-1] if train_losses else None,
            "min_loss": min(train_losses) if train_losses else None,
            "final_eval_loss": final_eval,
            "wall_clock_s": round(wall_s, 3),
            "tokens_per_s": (
                round((cfg.train.batch_size * cfg.model.block_size * step) / wall_s, 1)
                if wall_s > 0 else None
            ),
            "steps": step,
        },
        "eval_curve": eval_losses,
        "checkpoint": {
            "dir": str(out_dir.relative_to(ROOT)) if str(out_dir).startswith(str(ROOT)) else str(out_dir),
            "weights_file": sidecar.get("weights"),
            "config_file": "config.json",
        },
        "telemetry_summary": {
            "max_torch_allocated_bytes": telemetry.snapshot().torch_allocated_bytes,
            "max_torch_reserved_bytes": telemetry.snapshot().torch_reserved_bytes,
        },
        "metrics_jsonl": str(metrics_path.relative_to(ROOT)) if str(metrics_path).startswith(str(ROOT)) else str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Pretrain done — summary written to %s", summary_path)
    return summary
