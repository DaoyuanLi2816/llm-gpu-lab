"""Smoke test the HTML report against a fake results directory."""

from __future__ import annotations

import json
from pathlib import Path

from llm_gpu_lab.report import build_html_report


def _write_min_artifacts(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "environment.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-05-16T00:00:00+00:00",
                "platform": {
                    "system": "Windows",
                    "release": "11",
                    "machine": "AMD64",
                    "python_version": "3.11.15",
                    "python_implementation": "CPython",
                },
                "git_commit": "deadbeef",
                "torch": {
                    "installed": True,
                    "version": "2.4.0",
                    "cuda_available": True,
                    "cuda_runtime_version": "12.4",
                    "bf16_supported": True,
                    "devices": [
                        {
                            "index": 0,
                            "name": "NVIDIA GeForce RTX 4080",
                            "total_memory_gib": 16.0,
                            "capability": "8.9",
                        }
                    ],
                },
                "nvml": {"driver_version": "555.55"},
            }
        ),
        encoding="utf-8",
    )
    (d / "pretrain_metrics.json").write_text(
        json.dumps(
            {
                "name": "tiny_10m_smoke",
                "device": "cuda",
                "precision": "bf16",
                "model": {
                    "n_layer": 4, "n_head": 4, "n_embd": 128,
                    "block_size": 128, "vocab_size": 4096, "num_params": 1_500_000,
                },
                "results": {
                    "first_loss": 8.4, "last_loss": 5.0, "min_loss": 4.9,
                    "final_eval_loss": 5.1, "wall_clock_s": 12.0,
                    "tokens_per_s": 5000.0, "steps": 200,
                },
                "telemetry_summary": {
                    "max_torch_allocated_bytes": 100_000_000,
                    "max_torch_reserved_bytes": 200_000_000,
                },
            }
        ),
        encoding="utf-8",
    )
    (d / "pretrain_metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"step": s, "loss": 8.0 - s * 0.01}) for s in range(1, 21)
        ),
        encoding="utf-8",
    )
    (d / "eval_results.json").write_text(
        json.dumps(
            {
                "name": "smoke_eval",
                "runner": {"runner": "echo"},
                "n_prompts": 10,
                "n_with_checks": 10,
                "n_passed": 7,
                "pass_rate": 0.7,
                "avg_latency_ms": 10.0,
                "avg_response_chars": 5.0,
                "results": [
                    {
                        "id": "p", "task": "x", "prompt": "p", "response": "r",
                        "passed": True, "checks": [{"type": "exact", "passed": True}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (d / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "torch_available": True, "torch_version": "2.4.0", "device": "cuda",
                "device_name": "RTX 4080", "max_reserved_bytes": 100_000_000,
                "matmul": [
                    {"size": 512, "dtype": "fp16", "iters": 8, "duration_s": 0.1, "tflops": 12.3}
                ],
                "tiny_gen": {"new_tokens": 64, "duration_s": 0.5, "tokens_per_s": 128.0,
                             "n_layer": 4, "n_head": 4, "n_embd": 128},
            }
        ),
        encoding="utf-8",
    )
    (d / "generation_samples.json").write_text(
        json.dumps(
            {
                "checkpoint": "x.safetensors",
                "device": "cuda",
                "tokens_per_s": 1000.0,
                "n_prompts": 1,
                "samples": [{"prompt": "hi", "continuation": " world", "latency_s": 0.1, "new_token_count": 2}],
            }
        ),
        encoding="utf-8",
    )


def test_report_builds(tmp_path: Path) -> None:
    res = tmp_path / "rtx4080"
    _write_min_artifacts(res)
    out = tmp_path / "report.html"
    p = build_html_report(results_dir=res, out_path=out)
    assert p.is_file()
    html = p.read_text(encoding="utf-8")
    assert "llm-gpu-lab" in html
    assert "RTX 4080" in html
    assert "Pretrain" in html
