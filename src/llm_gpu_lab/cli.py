"""Typer entry point — wires all subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from llm_gpu_lab.logging_utils import get_logger
from llm_gpu_lab.paths import ROOT, ensure_dir

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    add_completion=False,
    help="One GPU. Full LLM workflow. Real benchmarks. No cloud required.",
    pretty_exceptions_enable=False,
)


@app.command("doctor")
def cmd_doctor(
    out: Path = typer.Option(
        ROOT / "results" / "rtx4080" / "environment.json",
        "--out",
        help="Where to write the environment JSON.",
    ),
) -> None:
    """Collect environment diagnostics into a JSON file."""
    from llm_gpu_lab.env import write_environment

    p = write_environment(out)
    console.print(f"[green][OK][/] environment written to [bold]{p}[/]")


@app.command("train-tokenizer")
def cmd_train_tokenizer(
    config: Path = typer.Option(..., "--config", help="Path to a pretrain YAML config."),
) -> None:
    """Train a tiny BPE tokenizer on the configured corpus."""
    from llm_gpu_lab.config import load_pretrain_config
    from llm_gpu_lab.data.tiny_corpus import build_tiny_corpus
    from llm_gpu_lab.tokenization import train_tokenizer

    cfg = load_pretrain_config(config)
    if cfg.data.source != "synthetic":
        raise typer.BadParameter(
            "train-tokenizer currently expects data.source=synthetic for reproducibility."
        )
    corpus = build_tiny_corpus(n_examples=cfg.data.n_examples, seed=cfg.data.seed)
    tok = train_tokenizer(
        corpus=corpus,
        vocab_size=cfg.tokenizer.vocab_size,
        output_path=cfg.tokenizer.output_path,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=list(cfg.tokenizer.special_tokens),
    )
    console.print(f"[green][OK][/] tokenizer (vocab={tok.get_vocab_size()}) → {cfg.tokenizer.output_path}")


@app.command("pretrain")
def cmd_pretrain(
    config: Path = typer.Option(..., "--config", help="Path to a pretrain YAML config."),
) -> None:
    """Pretrain the tiny GPT on the configured corpus."""
    from llm_gpu_lab.config import load_pretrain_config
    from llm_gpu_lab.train import run_pretrain

    cfg = load_pretrain_config(config)
    summary = run_pretrain(cfg)
    console.print_json(json.dumps(summary["results"]))


@app.command("generate")
def cmd_generate(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to .safetensors or .pt weights."),
    prompts: Optional[Path] = typer.Option(
        None,
        "--prompts",
        help="Optional text file with one prompt per line. Falls back to default prompts.",
    ),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="A single prompt (overrides --prompts)."),
    out: Path = typer.Option(
        ROOT / "results" / "rtx4080" / "generation_samples.json",
        "--out",
        help="Where to write generation samples JSON.",
    ),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
    temperature: float = typer.Option(0.9, "--temperature"),
    top_k: int = typer.Option(40, "--top-k"),
    top_p: float = typer.Option(0.95, "--top-p"),
    seed: int = typer.Option(1337, "--seed"),
) -> None:
    """Load a TinyGPT checkpoint and generate text."""
    from llm_gpu_lab.generate import generate_from_checkpoint

    if prompt and prompts:
        raise typer.BadParameter("Pass either --prompt or --prompts, not both.")
    if prompt:
        prompt_list = [prompt]
    elif prompts:
        text = prompts.read_text(encoding="utf-8")
        prompt_list = [line for line in text.splitlines() if line.strip()]
    else:
        prompt_list = [
            "Once upon a time",
            "Maya walked through the",
            "two plus three equals",
            "the color of the",
            "It was a quiet",
        ]
    result = generate_from_checkpoint(
        checkpoint=checkpoint,
        prompts=prompt_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        out_path=out,
    )
    console.print(f"[green][OK][/] generation samples → {out}")
    console.print_json(json.dumps({"n_samples": len(result["samples"])}))


@app.command("sft")
def cmd_sft(
    config: Path = typer.Option(..., "--config", help="Path to an SFT YAML config."),
) -> None:
    """Run LoRA / QLoRA supervised fine-tuning."""
    from llm_gpu_lab.config import load_sft_config
    from llm_gpu_lab.train import run_sft

    cfg = load_sft_config(config)
    summary = run_sft(cfg)
    console.print_json(json.dumps(summary))


@app.command("eval")
def cmd_eval(
    config: Path = typer.Option(..., "--config", help="Path to an eval YAML config."),
) -> None:
    """Run the basic prompt-checks evaluation."""
    from llm_gpu_lab.config import load_eval_config
    from llm_gpu_lab.eval import run_basic_eval

    cfg = load_eval_config(config)
    summary = run_basic_eval(cfg)
    console.print_json(json.dumps({
        "pass_rate": summary["pass_rate"],
        "n_passed": summary["n_passed"],
        "n_with_checks": summary["n_with_checks"],
    }))


@app.command("report")
def cmd_report(
    results_dir: Path = typer.Option(
        ROOT / "results" / "rtx4080", "--results-dir", help="Folder of JSON / MD artifacts."
    ),
    out: Path = typer.Option(
        ROOT / "results" / "rtx4080" / "report.html", "--out", help="HTML output path."
    ),
) -> None:
    """Build an HTML report from existing JSON artifacts."""
    from llm_gpu_lab.report import build_html_report

    p = build_html_report(results_dir=results_dir, out_path=out)
    console.print(f"[green][OK][/] report → {p}")


@app.command("export-gguf")
def cmd_export_gguf(
    config: Path = typer.Option(..., "--config", help="GGUF export YAML config."),
) -> None:
    """Merge LoRA + convert merged model to GGUF + optional quantization."""
    from llm_gpu_lab.config import load_gguf_config
    from llm_gpu_lab.export import run_gguf_export

    cfg = load_gguf_config(config)
    res = run_gguf_export(cfg)
    out_path = ensure_dir(ROOT / "results" / "rtx4080") / "gguf_export.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    status = "[green][OK][/]" if res["success"] else "[yellow]partial[/]"
    console.print(f"{status} gguf export — details in {out_path}")


@app.command("serve-llamacpp")
def cmd_serve_llamacpp(
    model: Path = typer.Option(..., "--model", help="Path to a .gguf model."),
    port: int = typer.Option(8080, "--port"),
    n_ctx: int = typer.Option(4096, "--n-ctx"),
    n_gpu_layers: int = typer.Option(999, "--n-gpu-layers"),
) -> None:
    """Start the llama.cpp HTTP server on a local GGUF model."""
    from llm_gpu_lab.serve import serve_llamacpp

    rc = serve_llamacpp(
        model=str(model),
        port=port,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
    raise typer.Exit(code=rc)


@app.command("bench-gpu")
def cmd_bench_gpu(
    out: Path = typer.Option(
        ROOT / "results" / "rtx4080" / "benchmark_summary.json", "--out"
    ),
) -> None:
    """Run the GPU micro-benchmark and persist a JSON summary."""
    from llm_gpu_lab.bench import run_gpu_bench

    result = run_gpu_bench(out)
    console.print(f"[green][OK][/] benchmark → {out}")
    if "tiny_gen" in result:
        console.print(
            f"tiny_gpt generation: {result['tiny_gen']['tokens_per_s']} tok/s"
        )


@app.command("lm-eval")
def cmd_lm_eval(
    base_model: str = typer.Option(..., "--base-model"),
    tasks: List[str] = typer.Option(["arc_easy"], "--task", help="Repeatable."),
    limit: Optional[int] = typer.Option(20, "--limit"),
    out: Path = typer.Option(ROOT / "results" / "rtx4080" / "lm_eval_results.json", "--out"),
    adapter_path: Optional[Path] = typer.Option(None, "--adapter-path"),
    batch_size: int = typer.Option(1, "--batch-size"),
) -> None:
    """Run a small subset of EleutherAI lm-evaluation-harness tasks."""
    from llm_gpu_lab.eval.lm_eval_bridge import run_lm_eval

    run_lm_eval(
        base_model=base_model,
        tasks=list(tasks),
        limit=limit,
        out_path=out,
        adapter_path=str(adapter_path) if adapter_path else None,
        batch_size=batch_size,
    )
    console.print(f"[green][OK][/] lm-eval → {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
