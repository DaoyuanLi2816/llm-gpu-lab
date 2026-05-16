# Run manifest — RTX 4080 16 GB

Hand-recorded summary of the run that produced the JSON / HTML artifacts
under this directory.

## Hardware

- GPU: NVIDIA GeForce RTX 4080 (16 GB GDDR6X, compute capability 8.9)
- Driver: 595.97
- CUDA runtime reported by torch: 12.4
- BF16 supported: yes
- CPU: Intel 13th-gen (Raptor Lake) x86_64
- OS: Windows 11 Pro 22H2 / 10.0.22631
- Python: CPython 3.11.15 (uv-managed)

## Software versions (key)

- `torch` 2.6.0+cu124
- `transformers` 5.8.1
- `tokenizers` 0.22.2
- `peft` 0.19.1
- `trl` 1.4.0
- `bitsandbytes` — not installed (QLoRA disabled)
- `huggingface_hub` 1.15.0

The full version list lives in `environment.json`.

## Commands executed

```bash
python -m llm_gpu_lab doctor --out results/rtx4080/environment.json
python -m llm_gpu_lab train-tokenizer --config configs/pretrain/tiny_10m_smoke.yaml
python -m llm_gpu_lab pretrain --config configs/pretrain/tiny_10m_smoke.yaml
python -m llm_gpu_lab generate \
  --checkpoint artifacts/checkpoints/tiny_10m_smoke/final.safetensors \
  --prompts examples/prompts/generation_prompts.txt \
  --out results/rtx4080/generation_samples.json \
  --max-new-tokens 48 --temperature 0.85 --top-k 40 --top-p 0.95 --seed 7
python -m llm_gpu_lab sft --config configs/sft/smollm2_135m_lora_fallback.yaml
python -m llm_gpu_lab eval --config configs/eval/smoke_eval.yaml
python -m llm_gpu_lab bench-gpu --out results/rtx4080/benchmark_summary.json
python -m llm_gpu_lab export-gguf --config configs/export/gguf_q4_k_m.yaml
./external/llama.cpp/build/bin/llama-quantize.exe \
    artifacts/gguf/smollm2_135m_lora.f16.gguf \
    artifacts/gguf/smollm2_135m_lora.Q4_K_M.gguf Q4_K_M
python -m llm_gpu_lab report --results-dir results/rtx4080 --out results/rtx4080/report.html
```

## Per-stage status

| Stage              | Status         | Headline number                              |
|--------------------|----------------|----------------------------------------------|
| doctor             | OK             | CUDA available, bf16 supported               |
| tokenizer          | OK             | vocab=715 from 3 000 synthetic docs          |
| pretrain (smoke)   | OK             | loss 8.37 → 2.11 in 200 steps, ~274 k tok/s |
| generation         | OK             | 10 prompts, ~243 tok/s                       |
| SFT (SmolLM2-135M) | OK             | train_loss 3.30 → 2.40, eval_loss 2.66, 8 s |
| eval (12 prompts)  | OK             | pass-rate 6/12 = 0.50, avg 1470 ms/prompt    |
| GPU bench          | OK             | bf16 2048 ≈ 34 TFLOPS, tiny gen ≈ 271 tok/s |
| GGUF F16 convert   | OK             | 269 MB                                       |
| GGUF Q4_K_M quant  | OK             | 101 MB, 6.17 BPW                             |
| llama-cli serve    | recorded limit | non-interactive harness; see limitations.md  |
| HTML report        | OK             | self-contained, ~110 KB                      |
| pytest             | OK             | 25 / 25 passed                               |
| ruff               | OK             | clean                                        |
| placeholder audit  | OK             | passed                                       |

## Known limitations

See `limitations.md` for the detailed list.

## Artifact locations (relative to the repo root)

- `results/rtx4080/environment.json` — doctor output
- `results/rtx4080/pretrain_metrics.json` — pretrain summary
- `results/rtx4080/pretrain_metrics.jsonl` — per-step pretrain log
- `results/rtx4080/generation_samples.json` — text generations
- `results/rtx4080/sft_metrics.json` — SFT summary
- `results/rtx4080/sft_samples_before_after.json` — qualitative before/after
- `results/rtx4080/eval_results.json` — basic eval
- `results/rtx4080/benchmark_summary.json` — GPU bench
- `results/rtx4080/gguf_export.json` — GGUF conversion + quantization log
- `results/rtx4080/report.html` — the single-page HTML report
- `results/rtx4080/limitations.md` — honest list of what didn't run perfectly
- `results/rtx4080/run_manifest.md` — this file
