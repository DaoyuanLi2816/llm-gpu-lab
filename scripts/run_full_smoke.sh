#!/usr/bin/env bash
# Full local smoke run. Stops at the first hard failure; logs to results/rtx4080/.
set -euo pipefail
PY="${PY:-python}"

mkdir -p results/rtx4080

echo "[1/8] doctor"
"${PY}" -m llm_gpu_lab doctor --out results/rtx4080/environment.json

echo "[2/8] train tokenizer"
"${PY}" -m llm_gpu_lab train-tokenizer --config configs/pretrain/tiny_10m_smoke.yaml

echo "[3/8] pretrain smoke"
"${PY}" -m llm_gpu_lab pretrain --config configs/pretrain/tiny_10m_smoke.yaml

echo "[4/8] generate samples"
"${PY}" -m llm_gpu_lab generate \
  --checkpoint artifacts/checkpoints/tiny_10m_smoke/final.safetensors \
  --prompts examples/prompts/generation_prompts.txt \
  --out results/rtx4080/generation_samples.json

echo "[5/8] sft smoke (LoRA on SmolLM2-135M)"
"${PY}" -m llm_gpu_lab sft --config configs/sft/smollm2_135m_lora_fallback.yaml

echo "[6/8] eval smoke"
"${PY}" -m llm_gpu_lab eval --config configs/eval/smoke_eval.yaml

echo "[7/8] gpu bench"
"${PY}" -m llm_gpu_lab bench-gpu --out results/rtx4080/benchmark_summary.json

echo "[8/8] report"
"${PY}" -m llm_gpu_lab report --results-dir results/rtx4080 --out results/rtx4080/report.html

echo "[smoke] done — open results/rtx4080/report.html"
