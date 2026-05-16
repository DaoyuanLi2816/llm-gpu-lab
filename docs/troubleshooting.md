# Troubleshooting

Quick lookup table for issues we have actually hit while building or
running this repo.

## `python -m llm_gpu_lab doctor` says `cuda_available: false`

Probable causes:

1. You installed the **CPU-only** PyTorch wheel. Re-install with the CUDA
   index URL (see `docs/quickstart.md` step 2).
2. Your driver is older than the CUDA runtime that PyTorch wants.
   `nvidia-smi` shows the driver's CUDA capability — that must be ≥ the
   wheel's CUDA version (e.g. `cu124` needs driver-CUDA 12.4 or newer).
3. You launched Python inside WSL but only have a Windows driver. WSL2
   needs `nvidia-cuda-toolkit` installed inside the distro.

`python -m llm_gpu_lab doctor --out env.json` will write a JSON that
records every relevant version field — paste it into an issue if you are
stuck.

## `pip install bitsandbytes` fails / `bitsandbytes` import error

bitsandbytes only ships wheels for `manylinux2014_x86_64` and recent
Windows x64. On macOS or 32-bit Windows you cannot use QLoRA. Workaround:
stay on **plain LoRA** — the default smoke config (`smollm2_135m_lora_fallback.yaml`)
uses LoRA without bitsandbytes, so the workflow still completes.

## CUDA OOM during SFT

The SFT loop catches `torch.cuda.OutOfMemoryError` once and retries with
half the sequence length and half the LoRA rank. If it still OOMs:

1. switch to the smaller fallback model
   (`configs/sft/smollm2_135m_lora_fallback.yaml`);
2. lower `max_seq_length` in your config to 128;
3. set `precision: fp16` if bf16 was using more memory;
4. set `use_gradient_checkpointing: true` (the default already).

The 135M LoRA smoke run peaks at ~1.2 GB VRAM on an RTX 4080.

## HF Hub download fails

```
huggingface_hub.utils._errors.HfHubHTTPError: 429 — too many requests
```

You are being rate-limited. Set a token:

```bash
huggingface-cli login        # one-time
# or
export HF_TOKEN=hf_...
```

If you are offline entirely, run the synthetic-data smoke pipeline first.
It does not require any network access after dependencies are installed.

## TinyStories download fails

```
DatasetGenerationError: An error occurred while generating the dataset
```

The TinyStories revision pin may have rotted. Two options:

1. Drop `data.tinystories_revision` from your config to take the current `main`.
2. Switch `data.source` back to `synthetic` — every other step still
   works.

## `llama.cpp` build failure

If `bash scripts/setup_llamacpp.sh` errors out:

1. Verify `cmake` ≥ 3.16 and a C++ compiler are on `PATH`.
2. On Windows, install Visual Studio "Desktop development with C++"
   workload — it ships `cmake` and `cl.exe`.
3. As a fallback, download the **official prebuilt** binaries from
   <https://github.com/ggerganov/llama.cpp/releases>, unzip them into
   `external/llama.cpp/build/bin/`, and re-run `export-gguf`. The repo
   discovers the binaries by path, not by who built them.

## GGUF conversion: "unsupported architecture"

`convert_hf_to_gguf.py` does not know about every HF architecture.
SmolLM, Qwen 2 / 2.5 / 3, Llama 2 / 3 and Mistral are all supported.
The custom `TinyGPT` in this repo is **not** llama.cpp-compatible (it is
designed for teaching the pretraining loop, not deployment). If you
specifically want GGUF for the from-scratch model, you would need to
follow llama.cpp's "Adding a Model" guide — that is out of scope here.

## Windows + UnicodeEncodeError on Rich output

```
UnicodeEncodeError: 'gbk' codec can't encode character '✓' …
```

The Windows legacy console uses your locale's codepage by default. Set
UTF-8 mode globally:

```powershell
setx PYTHONUTF8 1     # restart shell after
```

The CLI also avoids non-ASCII glyphs everywhere it writes to stdout to
prevent this in the first place.

## Disk pressure / Hugging Face cache

The HF cache lives under `~/.cache/huggingface/` (Linux) or
`%USERPROFILE%\.cache\huggingface\` (Windows). It can grow past 10 GB if
you experiment with many models. To clean:

```bash
huggingface-cli scan-cache         # see what is there
huggingface-cli delete-cache       # interactive UI
```

The `artifacts/` folder in this repo is auto-cleaned by `make clean-artifacts`.

## `nvidia-smi` works but `pynvml` is missing

The Python package is shipped as `nvidia-ml-py` on PyPI; the importable
name is `pynvml`. We list it under `nvidia-ml-py` in `pyproject.toml`,
and the doctor script knows how to find it via either name.

## Slow training

If you observe training step times of seconds rather than tens of
milliseconds:

1. Check the doctor JSON: `torch.cuda_available` must be `true` **and**
   `torch.devices[0].capability` should match your card.
2. The first ~5 steps include CUDA kernel JIT — that is expected. Look
   at the **median** step time, not the first.
3. Other GPU processes (browsers, Ollama, ChatGPT desktop) hold
   ~7 GB of VRAM on a typical Windows box; this is fine but be aware
   they share SM time. Close them for a clean benchmark.

## Insufficient disk / RAM

The default smoke needs roughly:

- 2.5 GB for the PyTorch CUDA wheel
- 300 MB for the SmolLM2-135M HF cache
- 100 MB for transformers + peft + tokenizers
- 10 MB for the LoRA adapter
- 270 MB for the F16 GGUF (if you do the optional export)
- 100 MB for the Q4_K_M GGUF (if you also quantize)

If you are below 4 GB free disk, skip the GGUF export.

The pretraining smoke uses < 1 GB of RAM. SFT on SmolLM2-135M needs
~2 GB of RAM in addition to the VRAM footprint.
