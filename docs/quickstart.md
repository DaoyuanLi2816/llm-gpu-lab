# Quickstart

Five commands gets you from a fresh clone to an HTML report on a single
NVIDIA GPU.

## 0. Prerequisites

- Python 3.10 or 3.11 (3.12 also works for most paths)
- An NVIDIA GPU. Smoke configs are tuned for a 16 GB consumer card
  (RTX 4080 / 4070 Ti SUPER / 3090). Smaller cards still work — the smoke
  pretrain peaks at ~600 MB VRAM.
- `git`, plus `cmake` + a C++ compiler **only if** you want to build
  `llama.cpp` from source. The repo also supports the official
  pre-built `llama.cpp` Windows binaries (see step 6).

## 1. Create the venv and install the package

The easiest way is [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11 .venv
source .venv/Scripts/activate     # Windows
# source .venv/bin/activate       # Linux / macOS

uv pip install pip                # vanilla pip inside the uv venv
```

If you don't have `uv`, plain `python -m venv .venv` works the same way.

## 2. Install PyTorch with the right CUDA wheel

Pick the index URL that matches your local CUDA toolkit / driver. The
PyTorch official selector is at <https://pytorch.org/get-started/locally/>.

```bash
# CUDA 12.4 wheels — works with RTX 30/40-series + recent drivers
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.4"

# CPU-only (CI, no GPU)
# pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
```

## 3. Install the rest of the project

```bash
pip install -e ".[dev,nlp,hub]"

# Optional: 4-bit weights via bitsandbytes (Linux / Windows-x64 GPU only)
pip install -e ".[quant]" || echo "bitsandbytes unavailable — QLoRA disabled"
```

## 4. Run the doctor

```bash
python -m llm_gpu_lab doctor --out results/rtx4080/environment.json
```

You should see CUDA available with your GPU name. If not, see
`docs/troubleshooting.md`.

## 5. Run the smoke pipeline

```bash
make smoke
```

That target chains together:

1. `train-tokenizer` — trains a tiny BPE tokenizer on a synthetic local corpus
2. `pretrain`        — trains a ~1.3 M-parameter TinyGPT for 200 steps
3. `generate`        — produces text continuations from the checkpoint
4. `sft`             — LoRA-finetunes `HuggingFaceTB/SmolLM2-135M-Instruct`
   on a synthetic instruction dataset
5. `eval`            — scores 12 prompts (math, JSON formatting, rewriting,
   summarization, sentiment)
6. `bench-gpu`       — measures matmul TFLOPS and tiny-GPT generation speed
7. `report`          — writes `results/rtx4080/report.html`
8. `audit`           — placeholder audit

Open `results/rtx4080/report.html` in any browser.

> The first SFT step downloads ~270 MB from Hugging Face. After that,
> everything runs offline from the HF cache.

## 6. Optional — GGUF export and llama.cpp serving

```bash
# Either: build llama.cpp from source (needs cmake + a C++ toolchain)
bash scripts/setup_llamacpp.sh

# Or: place the official pre-built llama.cpp binaries under
# external/llama.cpp/build/bin/ — the export script discovers them
# the same way.

python -m llm_gpu_lab export-gguf --config configs/export/gguf_q4_k_m.yaml
python -m llm_gpu_lab serve-llamacpp \
    --model artifacts/gguf/smollm2_135m_lora.Q4_K_M.gguf \
    --port 8080
```

`serve-llamacpp` is a thin wrapper around `llama-server`; visit
<http://localhost:8080> when it's up.

## 7. Extended experiments (optional)

Two extended pretrain configs are provided for users who want to try
something more ambitious on a 4080:

```bash
python -m llm_gpu_lab pretrain --config configs/pretrain/tiny_30m_4080.yaml
python -m llm_gpu_lab pretrain --config configs/pretrain/tiny_100m_4080.yaml
```

The 100M config switches the data source to `tinystories`, which downloads
the public TinyStories dataset on first run.
