"""LoRA / QLoRA supervised fine-tuning on a small open instruction model.

We implement the training loop directly with ``torch.optim.AdamW`` instead of
``transformers.Trainer``. The Trainer pulls in HuggingFace ``datasets`` at
import time, which in turn imports ``pyarrow.dataset`` — and that segfaults on
some Windows / OneDrive paths. A hand-rolled loop is also easier to teach: ~80
lines, no hidden state, fully reproducible.

Defensive design:

* QLoRA is attempted only if ``bitsandbytes`` imports and CUDA is available;
  otherwise we silently degrade to plain LoRA so the smoke run still succeeds.
* If the base-model download fails (offline, rate-limited, HF Hub error) we
  fall back to the configured ``fallback_base_model`` and record the
  limitation under ``results/<gpu>/sft_limitations.md``.
* OOM during the first training step triggers an automatic retry with
  reduced sequence length and LoRA rank.

The intent is to make this command "boringly succeed" on a 16 GB consumer GPU
even when not everything is happy.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_gpu_lab.config import SFTConfig
from llm_gpu_lab.data.synthetic_instruct import (
    InstructExample,
    build_synthetic_instruct_dataset,
)
from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir
from llm_gpu_lab.telemetry import GPUTelemetry

logger = get_logger(__name__)


@dataclass
class _Outcome:
    used_qlora: bool
    base_model: str
    fallback_used: bool
    final_loss: Optional[float]
    n_train_examples: int
    n_eval_examples: int
    final_seq_len: int
    final_lora_r: int


def _have_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        logger.info("bitsandbytes unavailable: %r", exc)
        return False
    return True


def _select_dtype(precision: str):
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _format_for_sft(tokenizer, examples: List[InstructExample]) -> List[str]:
    texts: List[str] = []
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        for ex in examples:
            rendered = tokenizer.apply_chat_template(
                ex.to_chat(),
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(rendered)
        return texts
    for ex in examples:
        prompt = ex.instruction if not ex.input else f"{ex.instruction}\n\n{ex.input}"
        texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n{ex.output}\n")
    return texts


def _load_model_and_tokenizer(
    base_model: str,
    use_qlora: bool,
    dtype,
):
    """Returns ``(model, tokenizer, used_qlora)``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    used_qlora = False
    if use_qlora and _have_bitsandbytes() and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=False,
            )
            used_qlora = True
        except Exception as exc:
            logger.warning("QLoRA load failed (%r); falling back to plain LoRA.", exc)

    if not used_qlora:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=dtype,
                trust_remote_code=False,
            )
        except TypeError:
            # transformers < 5: the keyword was named torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
        if torch.cuda.is_available():
            model = model.to("cuda")
    return model, tokenizer, used_qlora


def _auto_target_modules(model) -> List[str]:
    """Best-effort module list that works for Llama / Qwen / SmolLM."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    hit = [c for c in candidates if c in names]
    if hit:
        return hit
    return ["q_proj", "v_proj"] if "q_proj" in names else ["c_attn"]


def _attach_lora(model, cfg: SFTConfig, used_qlora: bool, lora_r: int):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    target_modules = cfg.target_modules or _auto_target_modules(model)
    if used_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg), target_modules


def _record_limitation(metrics_dir: Path, body: str) -> None:
    p = metrics_dir / "sft_limitations.md"
    header = (
        "# SFT limitations\n\n"
        "This file is auto-generated when the SFT step had to degrade.\n\n"
    )
    if p.is_file():
        existing = p.read_text(encoding="utf-8")
        p.write_text(existing + "\n" + body, encoding="utf-8")
    else:
        p.write_text(header + body, encoding="utf-8")


def _to_input_ids_and_mask(encoded, pad_token_id: int):
    """Normalize the output of `apply_chat_template(return_tensors="pt")`.

    Older transformers versions returned a bare tensor; newer ones return a
    ``BatchEncoding`` dict.  We need ``input_ids`` and ``attention_mask`` from
    whichever shape we get.
    """
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        attn = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attn = None
    if attn is None:
        attn = (input_ids != pad_token_id).long()
    return input_ids, attn


def _generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    if tokenizer.chat_template:
        msgs = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids, attn = _to_input_ids_and_mask(encoded, tokenizer.pad_token_id)
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].size(1):], skip_special_tokens=True
        )
    return text.strip()


def _tokenize_examples(
    tokenizer, examples: List[InstructExample], max_seq_length: int
) -> List[Dict[str, List[int]]]:
    texts = _format_for_sft(tokenizer, examples)
    rows: List[Dict[str, List[int]]] = []
    pad_id = tokenizer.pad_token_id
    for txt in texts:
        ids = tokenizer(
            txt,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        labels = [tok if tok != pad_id else -100 for tok in ids["input_ids"]]
        rows.append(
            {
                "input_ids": list(ids["input_ids"]),
                "attention_mask": list(ids["attention_mask"]),
                "labels": labels,
            }
        )
    return rows


def _iter_batches(rows: List[Dict[str, List[int]]], batch_size: int, shuffle: bool, seed: int):
    import torch

    order = list(range(len(rows)))
    if shuffle:
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(rows), generator=rng).tolist()
        order = perm
    for start in range(0, len(order), batch_size):
        slab = [rows[i] for i in order[start : start + batch_size]]
        if not slab:
            continue
        yield {
            k: torch.tensor([r[k] for r in slab], dtype=torch.long)
            for k in slab[0]
        }


def _eval_loop(model, eval_rows, batch_size: int) -> Optional[float]:
    import torch

    if not eval_rows:
        return None
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in _iter_batches(eval_rows, batch_size=batch_size, shuffle=False, seed=0):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += float(out.loss.detach().cpu())
            n += 1
    model.train()
    return total_loss / max(1, n)


def _train_loop(
    model, tokenizer, train_examples, eval_examples, cfg: SFTConfig, max_seq_length: int
) -> Dict[str, Any]:
    """Hand-rolled training loop. AdamW, linear warmup, optional grad checkpointing."""
    import torch

    train_rows = _tokenize_examples(tokenizer, train_examples, max_seq_length=max_seq_length)
    eval_rows = _tokenize_examples(tokenizer, eval_examples, max_seq_length=max_seq_length)

    if cfg.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception as exc:
            logger.info("gradient_checkpointing_enable failed: %r — continuing without it", exc)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)

    total_steps = cfg.max_steps
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return cfg.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    use_amp_dtype = None
    if torch.cuda.is_available():
        if cfg.precision == "fp16" or (
            cfg.precision == "auto" and not torch.cuda.is_bf16_supported()
        ):
            use_amp_dtype = torch.float16
        elif cfg.precision in ("bf16", "auto") and torch.cuda.is_bf16_supported():
            use_amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if use_amp_dtype is torch.float16 else None

    step = 0
    losses: List[float] = []
    t_start = time.time()
    accum = cfg.grad_accum_steps
    log_every = max(1, total_steps // 10)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step < total_steps:
        for batch in _iter_batches(
            train_rows, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed + step
        ):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if use_amp_dtype is not None and use_amp_dtype is not torch.float32:
                ctx = torch.amp.autocast("cuda", dtype=use_amp_dtype)
            else:
                from contextlib import nullcontext

                ctx = nullcontext()
            with ctx:
                out = model(**batch)
                loss = out.loss / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(float(out.loss.detach().cpu()))

            if (step + 1) % accum == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = _lr(step // accum)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0 or step == 1:
                logger.info(
                    "sft step %d/%d loss=%.4f lr=%.2e",
                    step, total_steps, losses[-1], _lr(max(0, step // accum)),
                )
            if step >= total_steps:
                break
    train_runtime = time.time() - t_start
    eval_loss = _eval_loop(model, eval_rows, batch_size=max(1, cfg.batch_size))
    return {
        "train_runtime_s": round(train_runtime, 3),
        "train_loss": round(float(losses[-1]) if losses else float("nan"), 4),
        "first_loss": round(float(losses[0]) if losses else float("nan"), 4),
        "eval_loss": round(float(eval_loss), 4) if eval_loss is not None else None,
        "loss_curve_first_last_min": [
            round(float(losses[0]), 4) if losses else None,
            round(float(losses[-1]), 4) if losses else None,
            round(float(min(losses)), 4) if losses else None,
        ],
        "n_steps": step,
    }


def run_sft(cfg: SFTConfig) -> Dict[str, Any]:
    import torch

    metrics_dir = ensure_dir(ROOT / cfg.metrics_dir)
    output_dir = ensure_dir(ROOT / cfg.output_dir)

    dtype = _select_dtype(cfg.precision)
    logger.info("SFT base_model=%s qlora=%s dtype=%s", cfg.base_model, cfg.qlora, dtype)

    examples = build_synthetic_instruct_dataset(
        n_examples=cfg.n_train_examples + cfg.n_eval_examples, seed=cfg.seed
    )
    train_examples = examples[: cfg.n_train_examples]
    eval_examples = examples[cfg.n_train_examples :]

    before_after_prompts: List[str] = []
    for ex in eval_examples[:5]:
        prompt = ex.instruction if not ex.input else f"{ex.instruction}\n\n{ex.input}"
        before_after_prompts.append(prompt)

    attempted_models = [cfg.base_model]
    if cfg.fallback_base_model and cfg.fallback_base_model not in attempted_models:
        attempted_models.append(cfg.fallback_base_model)

    last_exc: Optional[Exception] = None
    samples_before: List[Dict[str, str]] = []
    samples_after: List[Dict[str, str]] = []
    chosen_model: Optional[str] = None
    used_qlora = False
    train_stats: Dict[str, Any] = {}
    fallback_used = False
    final_seq_len = cfg.max_seq_length
    final_lora_r = cfg.lora_r

    for attempt_idx, base_model in enumerate(attempted_models):
        try:
            tele = GPUTelemetry()
            tele.reset_peak()
            model, tokenizer, used_qlora = _load_model_and_tokenizer(
                base_model, use_qlora=cfg.qlora, dtype=dtype
            )

            samples_before = [
                {
                    "prompt": p,
                    "response": _generate_sample(model, tokenizer, p),
                }
                for p in before_after_prompts
            ]

            model, _target_modules = _attach_lora(
                model, cfg, used_qlora=used_qlora, lora_r=cfg.lora_r
            )
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "trainable params: %d / %d (%.4f%%)",
                trainable_params, total_params, 100 * trainable_params / total_params,
            )

            try:
                train_stats = _train_loop(
                    model, tokenizer, train_examples, eval_examples, cfg,
                    max_seq_length=cfg.max_seq_length,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                reduced_seq = max(64, cfg.max_seq_length // 2)
                reduced_r = max(4, cfg.lora_r // 2)
                _record_limitation(
                    metrics_dir,
                    f"- OOM on {base_model} at seq_len={cfg.max_seq_length}, "
                    f"lora_r={cfg.lora_r}.\n"
                    f"  Retrying with seq_len={reduced_seq}, lora_r={reduced_r}.\n",
                )
                final_seq_len = reduced_seq
                final_lora_r = reduced_r
                # Reload base + reattach LoRA with smaller r
                model, tokenizer, used_qlora = _load_model_and_tokenizer(
                    base_model, use_qlora=cfg.qlora, dtype=dtype
                )
                model, _ = _attach_lora(model, cfg, used_qlora=used_qlora, lora_r=reduced_r)
                train_stats = _train_loop(
                    model, tokenizer, train_examples, eval_examples, cfg,
                    max_seq_length=reduced_seq,
                )

            samples_after = [
                {
                    "prompt": p,
                    "response": _generate_sample(model, tokenizer, p),
                }
                for p in before_after_prompts
            ]

            adapter_dir = output_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            chosen_model = base_model
            fallback_used = attempt_idx > 0
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("SFT attempt with %s failed: %r", base_model, exc)
            _record_limitation(
                metrics_dir,
                f"- Attempt {attempt_idx + 1} with `{base_model}` failed: `{exc!r}`.\n",
            )
            continue

    if chosen_model is None:
        raise RuntimeError(
            f"All SFT attempts failed; last error: {last_exc!r}. "
            f"Models tried: {attempted_models}"
        )

    outcome = _Outcome(
        used_qlora=used_qlora,
        base_model=chosen_model,
        fallback_used=fallback_used,
        final_loss=train_stats.get("train_loss"),
        n_train_examples=len(train_examples),
        n_eval_examples=len(eval_examples),
        final_seq_len=final_seq_len,
        final_lora_r=final_lora_r,
    )

    sft_metrics: Dict[str, Any] = {
        "name": cfg.name,
        "base_model": outcome.base_model,
        "fallback_used": outcome.fallback_used,
        "used_qlora": outcome.used_qlora,
        "train_runtime_s": train_stats.get("train_runtime_s"),
        "train_loss": outcome.final_loss,
        "first_loss": train_stats.get("first_loss"),
        "eval_loss": train_stats.get("eval_loss"),
        "loss_curve_first_last_min": train_stats.get("loss_curve_first_last_min"),
        "n_train_examples": outcome.n_train_examples,
        "n_eval_examples": outcome.n_eval_examples,
        "final_seq_len": outcome.final_seq_len,
        "final_lora_r": outcome.final_lora_r,
        "lora_alpha": cfg.lora_alpha,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "adapter_dir": str(Path(cfg.output_dir) / "adapter"),
    }

    (metrics_dir / "sft_metrics.json").write_text(
        json.dumps(sft_metrics, indent=2), encoding="utf-8"
    )
    (metrics_dir / "sft_samples_before_after.json").write_text(
        json.dumps(
            {
                "base_model": outcome.base_model,
                "before": samples_before,
                "after": samples_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("SFT done — adapter at %s", Path(cfg.output_dir) / "adapter")
    return sft_metrics
