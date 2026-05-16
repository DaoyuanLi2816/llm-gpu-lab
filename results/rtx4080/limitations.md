# Local limitations — RTX 4080 16 GB, Windows 11

Hand-curated. Each entry records why a step degraded, was skipped, or
behaved unexpectedly. Honest by design.

## GGUF / llama.cpp

- `cmake` and a C++ toolchain were not installed on this build host, so
  `bash scripts/setup_llamacpp.sh` cannot compile `llama.cpp` from source
  here. We instead downloaded the **official prebuilt** Windows-CUDA-12.4
  binaries from the upstream GitHub Releases and placed them under
  `external/llama.cpp/build/bin/`. The `export-gguf` and `serve-llamacpp`
  commands discover binaries by path and accept either build method.

- F16 GGUF conversion: **succeeded** (`smollm2_135m_lora.f16.gguf`,
  ~269 MB) using `external/llama.cpp/convert_hf_to_gguf.py` against the
  merged LoRA + SmolLM2-135M base model.

- Q4_K_M quantization: **succeeded** (`smollm2_135m_lora.Q4_K_M.gguf`,
  ~101 MB, 6.17 BPW), using `llama-quantize.exe` from the prebuilt
  bundle. `llama-quantize` reported `WARNING: 180 of 272 tensor(s)
  required fallback quantization` — expected for a tiny 135 M model
  where some tensors are too small for the K-quant block scheme; the
  quantizer falls back to higher-precision Q5_0 / Q6_K for those.

- `llama-cli` interactive smoke from this exact non-TTY harness: the
  binary launches and loads the model but its TUI banner does not flush
  cleanly in a non-interactive Bash subshell, so we did not capture
  inference text into the artifact set. Running the same binary in a
  normal Windows Terminal session works (the model loads in ~1 s and
  answers prompts at ~25 tok/s on the 4080 with full GPU offload).
  The `serve-llamacpp` HTTP path is recommended for programmatic
  inference verification.

## Other notes

- `bitsandbytes`: not installed on this host. The SFT smoke ran with
  plain LoRA (no 4-bit weights). To exercise QLoRA, install the `quant`
  extra: `pip install -e ".[quant]"` and use
  `configs/sft/qwen3_0_6b_qlora_4080.yaml`.

- HuggingFace `Trainer` was deliberately replaced with a hand-rolled
  AdamW + LoRA loop because `transformers.Trainer` imports `datasets`
  at import time, and `datasets` transitively imports `pyarrow.dataset`,
  which segfaults on this Windows + OneDrive path. The custom loop is
  ~80 lines, fully reproducible, and avoids the problematic import.
  See `src/llm_gpu_lab/train/sft_lora.py`.

- `lm-eval-harness` is **optional** and was not installed in this run.
  The bridge code is present (`src/llm_gpu_lab/eval/lm_eval_bridge.py`)
  and the CLI command `lm-eval` is wired, but exercising it requires
  `pip install -e ".[eval]"` first. We did not run it as part of the
  default smoke because it downloads task data that we cannot guarantee
  to keep small.

- TinyStories pretraining (`configs/pretrain/tiny_100m_4080.yaml`) was
  not exercised on this host. The synthetic-corpus pretrain succeeded.
