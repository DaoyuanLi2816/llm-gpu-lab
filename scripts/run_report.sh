#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-python}"
"${PY}" -m llm_gpu_lab report --results-dir results/rtx4080 --out results/rtx4080/report.html
