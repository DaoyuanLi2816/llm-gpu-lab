"""Tests for the TinyGPT model: shape, loss, optimizer step, generate."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from llm_gpu_lab.models import TinyGPT, TinyGPTConfig


@pytest.fixture
def tiny_cfg() -> TinyGPTConfig:
    return TinyGPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_forward_shape(tiny_cfg: TinyGPTConfig) -> None:
    model = TinyGPT(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (3, 8))
    logits, loss = model(x)
    assert logits.shape == (3, 8, tiny_cfg.vocab_size)
    assert loss is None


def test_loss_finite(tiny_cfg: TinyGPTConfig) -> None:
    model = TinyGPT(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    assert loss is not None
    val = float(loss.detach())
    assert math.isfinite(val) and val > 0


def test_optimizer_step_changes_params(tiny_cfg: TinyGPTConfig) -> None:
    torch.manual_seed(0)
    model = TinyGPT(tiny_cfg)
    p_before = next(model.parameters()).detach().clone()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    p_after = next(model.parameters()).detach().clone()
    assert not torch.allclose(p_before, p_after)


def test_generate_returns_more_tokens(tiny_cfg: TinyGPTConfig) -> None:
    model = TinyGPT(tiny_cfg).eval()
    idx = torch.zeros((1, 2), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=4)
    assert out.shape[1] == 2 + 5
    assert out.dtype == torch.long


def test_block_size_overflow_raises(tiny_cfg: TinyGPTConfig) -> None:
    model = TinyGPT(tiny_cfg)
    x = torch.zeros((1, tiny_cfg.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model(x)


def test_num_params_positive(tiny_cfg: TinyGPTConfig) -> None:
    assert TinyGPT(tiny_cfg).num_params() > 0
