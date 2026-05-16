"""Round-trip the YAML configs through pydantic models."""

from __future__ import annotations

from pathlib import Path

from llm_gpu_lab.config import (
    load_eval_config,
    load_gguf_config,
    load_pretrain_config,
    load_sft_config,
)
from llm_gpu_lab.paths import ROOT


def test_pretrain_smoke_config_loads() -> None:
    cfg = load_pretrain_config(ROOT / "configs" / "pretrain" / "tiny_10m_smoke.yaml")
    assert cfg.name == "tiny_10m_smoke"
    assert cfg.model.n_layer == 4
    assert cfg.train.max_steps > 0


def test_pretrain_extended_configs_load() -> None:
    for fname in ("tiny_30m_4080.yaml", "tiny_100m_4080.yaml"):
        cfg = load_pretrain_config(ROOT / "configs" / "pretrain" / fname)
        assert cfg.model.n_layer >= 4


def test_sft_configs_load() -> None:
    for fname in (
        "qwen2_5_0_5b_lora_smoke.yaml",
        "qwen3_0_6b_qlora_4080.yaml",
        "smollm2_135m_lora_fallback.yaml",
    ):
        cfg = load_sft_config(ROOT / "configs" / "sft" / fname)
        assert cfg.base_model
        assert cfg.fallback_base_model


def test_eval_config_loads() -> None:
    cfg = load_eval_config(ROOT / "configs" / "eval" / "smoke_eval.yaml")
    assert Path(cfg.prompts_path).name.endswith(".jsonl")


def test_gguf_config_loads() -> None:
    cfg = load_gguf_config(ROOT / "configs" / "export" / "gguf_q4_k_m.yaml")
    assert cfg.quant_type == "Q4_K_M"
