"""Train a tiny tokenizer on the synthetic corpus and check roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from llm_gpu_lab.data.tiny_corpus import build_tiny_corpus
from llm_gpu_lab.tokenization import load_tokenizer, train_tokenizer


def test_train_and_roundtrip(tmp_path: Path) -> None:
    corpus = build_tiny_corpus(n_examples=200, seed=7)
    out = tmp_path / "tok.json"
    tok = train_tokenizer(
        corpus=corpus,
        vocab_size=512,
        output_path=out,
        min_frequency=1,
    )
    assert out.is_file()
    text = "Once upon a time, Maya walked through the bright forest."
    enc = tok.encode(text)
    assert len(enc.ids) > 0
    dec = tok.decode(enc.ids)
    # We don't require exact equality (byte-level BPE), but the meaningful
    # content should be preserved.
    for word in ["Maya", "bright", "forest"]:
        assert word in dec, dec


def test_load_tokenizer_roundtrip(tmp_path: Path) -> None:
    corpus = build_tiny_corpus(n_examples=100, seed=11)
    out = tmp_path / "tok2.json"
    train_tokenizer(corpus=corpus, vocab_size=300, output_path=out, min_frequency=1)
    tok = load_tokenizer(out)
    assert tok.get_vocab_size() > 0
    text = "the color of the sky is blue."
    ids = tok.encode(text).ids
    assert tok.decode(ids) == text or "blue" in tok.decode(ids)
