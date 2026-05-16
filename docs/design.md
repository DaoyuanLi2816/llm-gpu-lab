# Design

`llm-gpu-lab` is one pipeline assembled from small, independently testable
pieces. Each piece corresponds to a CLI command, a config file, and a
results JSON artifact.

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│  synthetic   │   │   tokenizer │   │   tiny GPT  │   │   generation │
│   corpus     │──▶│  (BPE/HF)   │──▶│   pretrain  │──▶│    samples   │
└──────────────┘   └─────────────┘   └─────────────┘   └──────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐   ┌──────────────┐
                                       │   benchmark │   │   HTML report │
                                       │   matmul +  │──▶│   (Jinja2 +   │
                                       │   tiny gen  │   │   matplotlib) │
                                       └─────────────┘   └──────────────┘
                                              ▲
┌────────────┐  ┌─────────────┐  ┌─────────────┐
│ SmolLM /   │  │  LoRA /      │  │   basic     │
│ Qwen base  │─▶│  QLoRA SFT   │─▶│   eval      │
│ from HF    │  │ (PEFT/TRL)   │  │  (12 prompts)│
└────────────┘  └─────────────┘  └─────────────┘
                       │
                       ▼
                ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                │ merge_lora   │   │  GGUF        │   │  llama.cpp   │
                │ (PEFT merge) │──▶│  convert +   │──▶│  serve       │
                │              │   │  quantize    │   │  (llama-server)│
                └──────────────┘   └──────────────┘   └──────────────┘
```

Every stage writes a small JSON artifact under `results/<gpu>/` so the
final HTML report can render a faithful snapshot of the run.

## Why two model paths?

1. **TinyGPT (from scratch)** — `src/llm_gpu_lab/models/tiny_gpt.py`. About
   1 M parameters, ~250 lines of code, no external weights. Useful to:
   - teach the full pretraining loop end-to-end
   - smoke-test the workflow on a CPU when no GPU is available
   - run reproducible benchmarks of the generation latency
2. **Open Hub model + LoRA** — typically `SmolLM2-135M-Instruct` or
   `Qwen2.5-0.5B-Instruct`. Useful to:
   - exercise the realistic LoRA / QLoRA toolchain
   - produce a GGUF that has architecture support in `llama.cpp`
   - measure latency-vs-quality trade-offs against a public baseline

Both paths share the eval, telemetry, and report code.

## Why smoke configs exist

The default configs are deliberately tiny so:

- the workflow finishes in **under two minutes** on a 4080 (excluding the
  first SmolLM2 download), so contributors and reviewers can iterate;
- contributors without a GPU can still exercise everything in CPU mode;
- CI on GitHub-hosted runners (CPU only) can validate the import graph,
  configs, and the report builder.

The `_4080` configs in `configs/pretrain/` and `configs/sft/` are
extended experiments — they assume you actually have a 16 GB GPU and a few
minutes of wall-clock time.

## Why HF `Trainer` is not used for SFT

The HuggingFace `Trainer` is convenient but it imports `datasets` at
import time, which in turn imports `pyarrow.dataset`. On certain Windows
configurations (notably under OneDrive paths), `pyarrow.dataset` segfaults
on import. Rather than block contributors on a corner-case stack issue,
`src/llm_gpu_lab/train/sft_lora.py` implements an explicit ~80-line
AdamW + LoRA loop that uses only PEFT + transformers' model classes. The
loop is also easier to teach: every line is in this repo, nothing is
hidden behind framework magic.

## Telemetry

Two layers:

1. `pynvml` (a.k.a. `nvidia-ml-py`) for global GPU state — total / used
   / free memory and GPU utilization;
2. `torch.cuda.max_memory_*` for the precise peak the *current* process
   reached. This is what we record into the metrics JSON because it is
   the only thing reproducible across runs.

`telemetry_window` is a context manager that resets the torch peak and
collects a snapshot at exit.

## Reproducibility

All randomness is seeded:

* synthetic data generators take a `seed` config entry
* training loops set `torch.manual_seed(cfg.train.seed)` and seed numpy
* the generation CLI exposes `--seed`

You should be able to re-run `make smoke` and get matching numbers within
floating-point noise.
