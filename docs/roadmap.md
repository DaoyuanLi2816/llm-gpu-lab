# Roadmap

This is the only place in the repo where `TODO`, `FIXME`, and `TBD` are
allowed (the placeholder audit enforces that — see
`scripts/audit_placeholders.sh`).

## Near-term

- TODO: Real TinyStories pretraining at the 30M and 100M scale, with
  contributed `results/<gpu>/` artifacts so the README benchmark table can
  cover more of the size curve.
- TODO: Richer `lm-eval` task suite — currently the bridge supports the
  harness but only points at `arc_easy` as an example. Adding
  `truthfulqa_mc1`, `hellaswag` (with `--limit`), and `gsm8k_cot` would
  give a more interesting before / after comparison.
- TODO: Optional vLLM comparison harness for the same SFT model.
- TODO: Add a GGUF quant benchmark — measure pass-rate of `eval` on
  Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M, so users can pick the smallest quant
  that retains acceptable behaviour.
- TODO: Add `make bench-llamacpp` that runs `llama-bench` against the
  exported GGUF on the local GPU.

## Medium-term

- TODO: Multi-GPU contributed results. If someone runs the smoke on a
  3090, a 4090, a 5090, or a 7900 XTX (via ROCm), and contributes a
  `results/<gpu>/` directory, the report would compare numbers across
  cards.
- TODO: Add an optional Triton-accelerated attention kernel in the
  TinyGPT model so the matmul throughput section is comparable to
  Flash-Attention-class kernels.
- TODO: Optional `wandb` integration gated behind an env var; record
  pretrain + SFT metrics to a private project.

## Won't-fix in the short term

- We do **not** plan to ship a GitHub Actions GPU benchmark. GitHub-hosted
  runners do not have CUDA GPUs, and self-hosted GPU runners are out of
  scope for an open-source educational repo. CI runs the lint, tests, and
  audit on CPU only; GPU smoke is expected to be run locally.
- We do **not** plan to support paid hosted inference (OpenAI, Anthropic,
  Gemini) — the project's premise is local-only.

## Things to investigate (FIXME)

- FIXME: The basic eval uses a single regex (`-?\d+(?:\.\d+)?`) to pull a
  numeric answer out of the response. For prompts that include the input
  numbers in the response (which most assistant models do), that's lossy.
  A better scoring would be "last number in the response" or a
  task-specific extractor.
- FIXME: When the SFT path runs out of memory we fall back to a smaller
  LoRA rank but do not currently fall back across precision (fp16 vs
  bf16). On an RTX 3060 or 3080 with limited bf16 SM count, that might
  matter.
- TBD: Whether to ship a curated, larger synthetic instruction dataset
  built into the package, vs. always generating procedurally.
