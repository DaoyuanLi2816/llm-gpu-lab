#!/usr/bin/env bash
# Set up a Python 3.10/3.11 venv with project deps.
# Prefers `uv` if available (much faster), falls back to plain `python -m venv`.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Optional env vars:
#   PY=3.11      # which Python to use
#   VENV=.venv   # venv directory
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124  # CUDA wheel index
set -euo pipefail

PY="${PY:-3.11}"
VENV="${VENV:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if command -v uv >/dev/null 2>&1; then
  echo "[setup] using uv"
  uv venv --python "${PY}" "${VENV}"
  PYBIN="${VENV}/Scripts/python.exe"
  [ -x "${PYBIN}" ] || PYBIN="${VENV}/bin/python"
else
  echo "[setup] uv not found, using python -m venv"
  command -v "python${PY}" >/dev/null 2>&1 && PYCMD="python${PY}" || PYCMD="python"
  "${PYCMD}" -m venv "${VENV}"
  PYBIN="${VENV}/Scripts/python.exe"
  [ -x "${PYBIN}" ] || PYBIN="${VENV}/bin/python"
fi

"${PYBIN}" -m pip install --upgrade pip
echo "[setup] installing torch (CUDA wheels: ${TORCH_INDEX})"
"${PYBIN}" -m pip install --index-url "${TORCH_INDEX}" "torch>=2.1"

echo "[setup] installing package + extras"
"${PYBIN}" -m pip install -e ".[dev,nlp,hub]"
echo "[setup] optional: bitsandbytes for QLoRA"
"${PYBIN}" -m pip install -e ".[quant]" || echo "[setup] bitsandbytes failed; QLoRA disabled"

echo "[setup] done. activate with: source ${VENV}/Scripts/activate  (Windows: ${VENV}\\Scripts\\activate)"
