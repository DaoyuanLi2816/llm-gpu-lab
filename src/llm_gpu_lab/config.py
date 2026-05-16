"""Pydantic config models loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    vocab_size: int = 8192
    output_path: str = "artifacts/tokenizers/tiny_tokenizer.json"
    min_frequency: int = 2
    special_tokens: List[str] = Field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )


class DataConfig(BaseModel):
    source: str = "synthetic"  # "synthetic" | "tinystories" | "local_text"
    n_examples: int = 4000
    seed: int = 1337
    max_length: int = 512
    local_text_path: Optional[str] = None
    tinystories_revision: Optional[str] = None


class ModelConfig(BaseModel):
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True


class TrainConfig(BaseModel):
    out_dir: str = "artifacts/checkpoints/tiny_10m_smoke"
    metrics_dir: str = "results/rtx4080"
    max_steps: int = 200
    eval_every: int = 50
    log_every: int = 10
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    warmup_steps: int = 20
    seed: int = 1337
    precision: str = "auto"  # "auto" | "fp32" | "fp16" | "bf16"
    eval_iters: int = 20
    save_safetensors: bool = True


class PretrainYAML(BaseModel):
    name: str = "tiny_10m_smoke"
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


class SFTConfig(BaseModel):
    name: str = "smollm2_135m_lora_fallback"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    fallback_base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    max_seq_length: int = 256
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_steps: int = 30
    lr: float = 2.0e-4
    warmup_ratio: float = 0.1
    seed: int = 1337
    precision: str = "auto"
    n_train_examples: int = 64
    n_eval_examples: int = 16
    output_dir: str = "artifacts/adapters/smollm2_135m_lora_fallback"
    metrics_dir: str = "results/rtx4080"
    use_gradient_checkpointing: bool = True


class EvalConfig(BaseModel):
    name: str = "smoke_eval"
    base_model: Optional[str] = None
    adapter_path: Optional[str] = None
    prompts_path: str = "examples/prompts/eval_prompts.jsonl"
    max_new_tokens: int = 96
    out: str = "results/rtx4080/eval_results.json"
    precision: str = "auto"
    use_chat_template: bool = True


class GGUFExportConfig(BaseModel):
    name: str = "gguf_q4_k_m"
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    adapter_path: Optional[str] = "artifacts/adapters/smollm2_135m_lora_fallback"
    merged_dir: str = "artifacts/merged/smollm2_135m_lora"
    gguf_out_dir: str = "artifacts/gguf"
    quant_type: str = "Q4_K_M"
    llamacpp_dir: str = "external/llama.cpp"
    limitations_path: str = "results/rtx4080/limitations.md"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_pretrain_config(path: str | Path) -> PretrainYAML:
    return PretrainYAML(**load_yaml(path))


def load_sft_config(path: str | Path) -> SFTConfig:
    return SFTConfig(**load_yaml(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig(**load_yaml(path))


def load_gguf_config(path: str | Path) -> GGUFExportConfig:
    return GGUFExportConfig(**load_yaml(path))
