"""Optional TinyStories loader via Hugging Face Datasets.

This module is only imported when the user explicitly chooses ``source:
tinystories`` in a config. It is a thin wrapper that:

1. records the dataset name and revision in the returned metadata so the
   pretraining metrics JSON can attribute its data;
2. yields raw text lines that the tokenizer / packer can consume directly.

We intentionally avoid hard-coding licence assertions here because dataset
licences can change over time. If you use this path, read the dataset card
on the Hugging Face Hub and update ``docs/licenses.md`` accordingly.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional


def load_tinystories_text(
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: int = 5000,
) -> Iterator[str]:
    """Yield text lines from the public TinyStories dataset.

    Raises:
        ImportError: when ``datasets`` is not installed.
        RuntimeError: when the load fails for any other reason (e.g. offline).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading TinyStories requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    kwargs: Dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    try:
        ds = load_dataset("roneneldan/TinyStories", **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load TinyStories: {exc!r}") from exc

    n = 0
    for row in ds:
        text = row.get("text") if isinstance(row, dict) else None
        if not text:
            continue
        yield text
        n += 1
        if n >= max_examples:
            break


def tinystories_metadata(revision: Optional[str] = None) -> Dict[str, Any]:
    """Static metadata recorded into the pretrain metrics JSON."""
    return {
        "source": "tinystories",
        "hf_dataset": "roneneldan/TinyStories",
        "revision": revision,
        "license_note": "See https://huggingface.co/datasets/roneneldan/TinyStories",
    }


def collect_first_n(it: Iterable[str], n: int) -> list[str]:
    out: list[str] = []
    for line in it:
        out.append(line)
        if len(out) >= n:
            break
    return out
