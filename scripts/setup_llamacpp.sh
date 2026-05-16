#!/usr/bin/env bash
# Clone and build llama.cpp into ./external/llama.cpp with CUDA support if
# available. Idempotent: re-running just rebuilds.
set -euo pipefail

LCC_DIR="${LCC_DIR:-external/llama.cpp}"
LCC_REPO="${LCC_REPO:-https://github.com/ggerganov/llama.cpp}"
LCC_REVISION="${LCC_REVISION:-master}"
CUDA="${CUDA:-auto}"  # auto|on|off

mkdir -p "$(dirname "${LCC_DIR}")"

if [ ! -d "${LCC_DIR}/.git" ]; then
  echo "[llama.cpp] cloning into ${LCC_DIR}"
  git clone --depth 1 "${LCC_REPO}" "${LCC_DIR}"
fi

pushd "${LCC_DIR}" >/dev/null
git fetch --depth 1 origin "${LCC_REVISION}" || true
git checkout "${LCC_REVISION}" || true

# Decide whether to enable CUDA
WANT_CUDA=0
if [ "${CUDA}" = "on" ]; then
  WANT_CUDA=1
elif [ "${CUDA}" = "auto" ]; then
  if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
    WANT_CUDA=1
  fi
fi

echo "[llama.cpp] building (cuda=${WANT_CUDA})"
if command -v cmake >/dev/null 2>&1; then
  CMAKE_ARGS=("-B" "build")
  if [ "${WANT_CUDA}" = "1" ]; then
    CMAKE_ARGS+=("-DGGML_CUDA=ON")
  fi
  cmake "${CMAKE_ARGS[@]}"
  cmake --build build --config Release -j
else
  echo "[llama.cpp] cmake not found, falling back to 'make'"
  if [ "${WANT_CUDA}" = "1" ]; then
    GGML_CUDA=1 make -j
  else
    make -j
  fi
fi

popd >/dev/null

echo "[llama.cpp] binaries:"
ls "${LCC_DIR}/build/bin" 2>/dev/null || ls "${LCC_DIR}" || true

# Install the converter's Python deps into the active venv if requirements file
# exists. Failure is non-fatal.
if [ -f "${LCC_DIR}/requirements.txt" ]; then
  echo "[llama.cpp] installing converter Python deps (best-effort)"
  pip install -r "${LCC_DIR}/requirements.txt" || true
fi

echo "[llama.cpp] done."
