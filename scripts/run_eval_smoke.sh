#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-python}"
"${PY}" -m llm_gpu_lab eval --config configs/eval/smoke_eval.yaml
