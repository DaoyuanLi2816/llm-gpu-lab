"""Light-weight local evaluation runner.

Reads JSONL prompts and scores them with simple, transparent rules:

* ``json_valid``     — does the response parse as JSON?
* ``contains_all``   — does the response contain every required substring?
* ``exact``          — does the response equal the expected answer?
* ``numeric_equal``  — does the response parse to the expected number?
* ``regex_match``    — does the response match a regex?

We deliberately do not bring in BLEU/ROUGE: this is a smoke evaluator whose
results are stable, fast, and easy to inspect by hand. lm-eval-harness
integration is provided separately for richer benchmarks.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console

from llm_gpu_lab.config import EvalConfig
from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)
console = Console()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _score_one(check: Dict[str, Any], response: str) -> Tuple[bool, str]:
    kind = check.get("type")
    if kind == "json_valid":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return False, "response is not valid JSON"
        keys = check.get("must_have_keys") or []
        for k in keys:
            if isinstance(parsed, dict) and k not in parsed:
                return False, f"missing key {k!r}"
        return True, "ok"
    if kind == "contains_all":
        needles = check.get("substrings") or []
        missing = [s for s in needles if s.lower() not in response.lower()]
        if missing:
            return False, f"missing: {missing}"
        return True, "ok"
    if kind == "exact":
        expected = str(check.get("expected", ""))
        return response.strip() == expected.strip(), "ok"
    if kind == "numeric_equal":
        expected = check.get("expected")
        m = re.search(r"-?\d+(?:\.\d+)?", response)
        if not m:
            return False, "no number in response"
        try:
            got = float(m.group(0))
        except ValueError:
            return False, "could not parse number"
        return abs(got - float(expected)) < 1e-6, f"got={got}"
    if kind == "regex_match":
        pat = check.get("pattern", "")
        return bool(re.search(pat, response)), "ok"
    return False, f"unknown check type {kind!r}"


def _make_runner(cfg: EvalConfig):
    """Return a (callable(prompt) -> response, info) tuple."""
    import torch

    if cfg.adapter_path or cfg.base_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = cfg.base_model
        if base is None and cfg.adapter_path:
            # Derive base from the adapter's config if present
            adapter_cfg = Path(ROOT / cfg.adapter_path) / "adapter_config.json"
            if adapter_cfg.is_file():
                base = json.loads(adapter_cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if base is None:
            raise ValueError("base_model is required when adapter_path is not set")
        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        if cfg.adapter_path:
            from peft import PeftModel

            adapter_full = (
                ROOT / cfg.adapter_path
                if not Path(cfg.adapter_path).is_absolute()
                else Path(cfg.adapter_path)
            )
            if adapter_full.is_dir():
                model = PeftModel.from_pretrained(model, str(adapter_full))
            else:
                logger.warning(
                    "adapter_path %s not found; running base model only.", adapter_full
                )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        def _run(prompt: str) -> str:
            if cfg.use_chat_template and getattr(tokenizer, "chat_template", None):
                msgs = [{"role": "user", "content": prompt}]
                encoded = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                # newer transformers return a BatchEncoding; older — a tensor
                if hasattr(encoded, "keys") and "input_ids" in encoded:
                    input_ids = encoded["input_ids"].to(model.device)
                    attn = encoded.get("attention_mask")
                    attn = attn.to(model.device) if attn is not None else (
                        input_ids != tokenizer.pad_token_id
                    ).long()
                else:
                    input_ids = encoded.to(model.device)
                    attn = (input_ids != tokenizer.pad_token_id).long()
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                text = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
            else:
                enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                text = tokenizer.decode(
                    out[0][enc["input_ids"].size(1):], skip_special_tokens=True
                )
            return text.strip()

        info = {"runner": "transformers", "base_model": base, "adapter_path": cfg.adapter_path}
        return _run, info

    # Fallback: deterministic echo so the eval still produces shape-correct
    # JSON when no model is configured (e.g. CI). Marked clearly in info.
    def _echo(prompt: str) -> str:
        return f"NO_MODEL_CONFIGURED prompt-len={len(prompt)}"

    info = {"runner": "echo", "note": "No base_model/adapter set; running deterministic echo."}
    return _echo, info


def run_basic_eval(cfg: EvalConfig) -> Dict[str, Any]:
    prompts_path = ROOT / cfg.prompts_path if not Path(cfg.prompts_path).is_absolute() else Path(cfg.prompts_path)
    if not prompts_path.is_file():
        raise FileNotFoundError(prompts_path)
    prompts = _load_jsonl(prompts_path)
    if len(prompts) < 10:
        raise ValueError(f"Need at least 10 prompts, got {len(prompts)}: {prompts_path}")

    runner, runner_info = _make_runner(cfg)
    results: List[Dict[str, Any]] = []
    n_pass = 0
    total_latency_ms = 0.0
    total_chars = 0
    for prompt_row in prompts:
        prompt = prompt_row["prompt"]
        t0 = time.time()
        response = runner(prompt)
        dt_ms = (time.time() - t0) * 1000.0
        total_latency_ms += dt_ms
        total_chars += len(response)
        per_check_results: List[Dict[str, Any]] = []
        overall_pass = True
        for check in prompt_row.get("checks", []):
            ok, why = _score_one(check, response)
            per_check_results.append({"type": check.get("type"), "passed": ok, "note": why})
            if not ok:
                overall_pass = False
        if overall_pass and prompt_row.get("checks"):
            n_pass += 1
        results.append(
            {
                "id": prompt_row.get("id"),
                "task": prompt_row.get("task"),
                "prompt": prompt,
                "response": response,
                "passed": overall_pass,
                "checks": per_check_results,
                "latency_ms": round(dt_ms, 2),
                "response_chars": len(response),
            }
        )
    n_with_checks = sum(1 for p in prompts if p.get("checks"))
    summary = {
        "name": cfg.name,
        "runner": runner_info,
        "n_prompts": len(prompts),
        "n_with_checks": n_with_checks,
        "n_passed": n_pass,
        "pass_rate": (n_pass / n_with_checks) if n_with_checks else None,
        "avg_latency_ms": round(total_latency_ms / max(1, len(prompts)), 2),
        "avg_response_chars": round(total_chars / max(1, len(prompts)), 2),
        "results": results,
    }
    out_path = ROOT / cfg.out if not Path(cfg.out).is_absolute() else Path(cfg.out)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Eval done — passed %d / %d (rate=%s) in %.1fs (avg %.1fms/prompt)",
        n_pass, n_with_checks, summary["pass_rate"], total_latency_ms / 1000.0,
        summary["avg_latency_ms"],
    )
    return summary
