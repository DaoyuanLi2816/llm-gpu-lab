"""Pack a stream of token IDs into fixed-size blocks for next-token prediction.

We concatenate documents with a configurable separator token (default: BOS),
then chop the resulting 1-D array into ``block_size``-length windows. The
last partial window is dropped — typical of nanoGPT-style pipelines.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def pack_token_ids(
    docs_token_ids: Iterable[Iterable[int]],
    block_size: int,
    sep_token_id: int,
) -> np.ndarray:
    """Concatenate token IDs across documents and slice into (N, block_size+1)."""
    parts: List[np.ndarray] = []
    for doc in docs_token_ids:
        ids = np.asarray(list(doc), dtype=np.int64)
        if ids.size == 0:
            continue
        parts.append(ids)
        parts.append(np.asarray([sep_token_id], dtype=np.int64))
    if not parts:
        return np.zeros((0, block_size + 1), dtype=np.int64)
    flat = np.concatenate(parts, axis=0)
    # We want windows of length block_size + 1 so each window yields a (x, y)
    # pair where y is x shifted by one.
    n_full = (flat.size - 1) // block_size
    if n_full <= 0:
        return np.zeros((0, block_size + 1), dtype=np.int64)
    usable = n_full * block_size + 1  # one extra for the final target
    flat = flat[:usable]
    blocks = np.lib.stride_tricks.sliding_window_view(flat, window_shape=block_size + 1)[
        ::block_size
    ]
    return np.ascontiguousarray(blocks, dtype=np.int64)


def split_train_eval(packed: np.ndarray, eval_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if packed.shape[0] == 0:
        return packed, packed
    n_eval = max(1, int(packed.shape[0] * eval_fraction))
    return packed[n_eval:], packed[:n_eval]
