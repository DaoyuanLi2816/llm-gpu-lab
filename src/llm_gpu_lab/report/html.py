"""Render a single-page HTML report from local JSON artifacts.

The report is deliberately self-contained — no remote stylesheets, no
JavaScript frameworks. Plots are rendered with matplotlib and embedded as
base64 PNGs so the file works offline and can be opened from a USB stick.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %r", path, exc)
        return None


def _read_text(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _plot_to_png_b64(plot_fn) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    try:
        plot_fn(ax)
    except Exception as exc:
        logger.warning("plot failed: %r", exc)
        plt.close(fig)
        return None
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_pretrain_loss_plot(pretrain_metrics_jsonl: Path) -> Optional[str]:
    if not pretrain_metrics_jsonl.is_file():
        return None
    steps, losses = [], []
    for line in pretrain_metrics_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row and "loss" in row:
            steps.append(row["step"])
            losses.append(row["loss"])
    if not steps:
        return None

    def _plot(ax):
        ax.plot(steps, losses, marker="o", linewidth=1.5, markersize=3)
        ax.set_xlabel("step")
        ax.set_ylabel("train loss")
        ax.set_title("Pretrain loss")
        ax.grid(True, alpha=0.3)

    return _plot_to_png_b64(_plot)


def _build_matmul_bar(bench: Optional[Dict[str, Any]]) -> Optional[str]:
    if not bench:
        return None
    rows = [r for r in bench.get("matmul", []) if "tflops" in r]
    if not rows:
        return None

    def _plot(ax):
        labels = [f"{r['size']}/{r['dtype']}" for r in rows]
        tflops = [r["tflops"] for r in rows]
        bars = ax.bar(labels, tflops, color="#4f81bd")
        for bar, v in zip(bars, tflops, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("TFLOPS (matmul)")
        ax.set_title("Matmul throughput")
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    return _plot_to_png_b64(_plot)


def _build_eval_passrate_plot(eval_json: Optional[Dict[str, Any]]) -> Optional[str]:
    if not eval_json or not eval_json.get("results"):
        return None
    tasks: Dict[str, Dict[str, int]] = {}
    for r in eval_json["results"]:
        t = r.get("task") or "(unknown)"
        bucket = tasks.setdefault(t, {"pass": 0, "total": 0})
        if r.get("checks"):
            bucket["total"] += 1
            if r.get("passed"):
                bucket["pass"] += 1
    if not tasks:
        return None

    def _plot(ax):
        labels = list(tasks.keys())
        rates = [tasks[k]["pass"] / max(1, tasks[k]["total"]) for k in labels]
        bars = ax.bar(labels, rates, color="#9bbb59")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("pass rate")
        ax.set_title("Eval pass rate by task")
        for bar, v in zip(bars, rates, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    return _plot_to_png_b64(_plot)


def _format_bytes(n: Optional[int]) -> str:
    if not n:
        return "—"
    gib = n / (1024**3)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    mib = n / (1024**2)
    return f"{mib:.1f} MiB"


def build_html_report(results_dir: str | Path, out_path: str | Path) -> Path:
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    ensure_dir(out_path.parent)

    env_json = _read_json(results_dir / "environment.json")
    pretrain_json = _read_json(results_dir / "pretrain_metrics.json")
    sft_json = _read_json(results_dir / "sft_metrics.json")
    sft_limitations = _read_text(results_dir / "sft_limitations.md")
    eval_json = _read_json(results_dir / "eval_results.json")
    bench_json = _read_json(results_dir / "benchmark_summary.json")
    gen_json = _read_json(results_dir / "generation_samples.json")
    limitations_md = _read_text(results_dir / "limitations.md")

    ctx: Dict[str, Any] = {
        "env": env_json,
        "pretrain": pretrain_json,
        "sft": sft_json,
        "sft_limitations": sft_limitations,
        "eval": eval_json,
        "bench": bench_json,
        "generation": gen_json,
        "limitations": limitations_md,
        "plots": {
            "pretrain_loss": _build_pretrain_loss_plot(results_dir / "pretrain_metrics.jsonl"),
            "matmul": _build_matmul_bar(bench_json),
            "eval_passrate": _build_eval_passrate_plot(eval_json),
        },
        "fmt_bytes": _format_bytes,
    }

    j2_env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    j2_env.globals["fmt_bytes"] = _format_bytes
    template = j2_env.get_template("report.html.j2")
    html = template.render(**ctx)
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path
