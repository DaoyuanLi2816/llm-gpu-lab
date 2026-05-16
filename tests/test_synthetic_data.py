"""Tests for the deterministic synthetic data generators."""

from __future__ import annotations

import json

from llm_gpu_lab.data.synthetic_instruct import build_synthetic_instruct_dataset
from llm_gpu_lab.data.tiny_corpus import build_tiny_corpus


def test_tiny_corpus_deterministic() -> None:
    a = build_tiny_corpus(n_examples=120, seed=1234)
    b = build_tiny_corpus(n_examples=120, seed=1234)
    assert a == b
    assert len(a) == 120


def test_tiny_corpus_different_seeds() -> None:
    a = build_tiny_corpus(n_examples=50, seed=1)
    b = build_tiny_corpus(n_examples=50, seed=2)
    assert a != b


def test_synthetic_instruct_balanced() -> None:
    ds = build_synthetic_instruct_dataset(n_examples=25, seed=7)
    assert len(ds) == 25
    tasks = {ex.task for ex in ds}
    assert tasks == {
        "arithmetic",
        "json_format",
        "rewrite_past_tense",
        "summarize",
        "sentiment",
    }


def test_synthetic_instruct_outputs_match_check() -> None:
    ds = build_synthetic_instruct_dataset(n_examples=80, seed=3)
    # arithmetic outputs are integers as strings
    for ex in ds:
        if ex.task == "arithmetic":
            assert ex.output.lstrip("-").isdigit()
        elif ex.task == "sentiment":
            assert ex.output in {"positive", "negative"}
        elif ex.task == "json_format":
            parsed = json.loads(ex.output)
            assert isinstance(parsed, dict)
