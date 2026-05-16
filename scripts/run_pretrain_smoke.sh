#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-python}"
"${PY}" -m llm_gpu_lab train-tokenizer --config configs/pretrain/tiny_10m_smoke.yaml
"${PY}" -m llm_gpu_lab pretrain --config configs/pretrain/tiny_10m_smoke.yaml
