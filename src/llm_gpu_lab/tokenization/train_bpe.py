"""Train a tiny BPE tokenizer using HuggingFace tokenizers.

This is a thin wrapper: the heavy lifting is done by `tokenizers.Tokenizer`.
We expose a small subset that's convenient for the rest of the package and
matches what the tiny GPT expects: special tokens BOS / EOS / PAD / UNK and a
configurable vocab size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int = 8192,
    output_path: str | Path = "artifacts/tokenizers/tiny_tokenizer.json",
    min_frequency: int = 2,
    special_tokens: List[str] | None = None,
):
    """Train a byte-level BPE tokenizer and save it to ``output_path``.

    Returns the tokenizer object so callers can use it immediately without
    a separate load.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(list(corpus), trainer=trainer)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def token_id(tokenizer, token: str, default: int = 0) -> int:
    tid = tokenizer.token_to_id(token)
    return default if tid is None else int(tid)
