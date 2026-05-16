#!/usr/bin/env bash
# Quick local diagnostic. Mirrors what `doctor` writes to JSON, in human form.
set -u

echo "==== OS ===="
uname -a 2>/dev/null || ver

echo
echo "==== python ===="
python --version 2>&1 || true
which python 2>&1 || true

echo
echo "==== pip freeze (head) ===="
python -m pip freeze 2>/dev/null | head -40 || true

echo
echo "==== nvidia-smi ===="
nvidia-smi 2>&1 || echo "nvidia-smi not available"

echo
echo "==== torch ===="
python - <<'PY' 2>/dev/null
import sys
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda runtime:", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
except Exception as e:
    print("torch not importable:", repr(e))
PY

echo
echo "==== llama.cpp ===="
ls external/llama.cpp/build/bin 2>/dev/null || echo "no llama.cpp build dir"
