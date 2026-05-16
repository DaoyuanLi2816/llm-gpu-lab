#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-python}"
"${PY}" -m llm_gpu_lab sft --config configs/sft/smollm2_135m_lora_fallback.yaml
