"""Tests for the basic eval scoring rules."""

from __future__ import annotations

import json
from pathlib import Path

from llm_gpu_lab.config import EvalConfig
from llm_gpu_lab.eval.basic_eval import _score_one, run_basic_eval  # noqa: PLC2701


def test_numeric_equal_passes() -> None:
    ok, _ = _score_one({"type": "numeric_equal", "expected": 7}, "The answer is 7")
    assert ok


def test_numeric_equal_fails_when_wrong() -> None:
    ok, _ = _score_one({"type": "numeric_equal", "expected": 7}, "The answer is 8")
    assert not ok


def test_json_valid_with_keys() -> None:
    ok, _ = _score_one(
        {"type": "json_valid", "must_have_keys": ["name"]},
        '{"name": "Alice"}',
    )
    assert ok


def test_json_valid_fails_when_missing_key() -> None:
    ok, _ = _score_one(
        {"type": "json_valid", "must_have_keys": ["age"]},
        '{"name": "Alice"}',
    )
    assert not ok


def test_contains_all() -> None:
    ok, _ = _score_one(
        {"type": "contains_all", "substrings": ["walked", "park"]},
        "She walked through the park.",
    )
    assert ok


def test_regex_match() -> None:
    ok, _ = _score_one({"type": "regex_match", "pattern": r"(?i)blue"}, "BLUE")
    assert ok


def test_end_to_end_echo_runner(tmp_path: Path) -> None:
    # Build a 10-prompt JSONL on disk and run with the echo fallback.
    prompts = []
    for i in range(10):
        prompts.append(
            {
                "id": f"p{i}",
                "task": "echo",
                "prompt": f"prompt-{i}",
                "checks": [],
            }
        )
    p = tmp_path / "prompts.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in prompts), encoding="utf-8")
    out = tmp_path / "eval.json"

    cfg = EvalConfig(
        name="t",
        base_model=None,
        adapter_path=None,
        prompts_path=str(p),
        out=str(out),
        use_chat_template=False,
    )
    summary = run_basic_eval(cfg)
    assert summary["n_prompts"] == 10
    assert out.is_file()
    assert summary["pass_rate"] is None  # no checks → undefined
