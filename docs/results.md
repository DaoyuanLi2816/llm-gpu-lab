# Local results

This page summarizes what is in `results/rtx4080/` — the artifacts
produced by an actual run on the maintainer's RTX 4080 16 GB.

## How to read the report

Open `results/rtx4080/report.html` in any browser. The page is fully
self-contained: plots are inline base64 PNGs, no external stylesheets
or scripts.

## Artifact list

| File                              | Source command                                       |
|-----------------------------------|------------------------------------------------------|
| `environment.json`                | `python -m llm_gpu_lab doctor`                        |
| `pretrain_metrics.json`           | `python -m llm_gpu_lab pretrain`                      |
| `pretrain_metrics.jsonl`          | streaming log of every `log_every` step               |
| `generation_samples.json`         | `python -m llm_gpu_lab generate`                      |
| `sft_metrics.json`                | `python -m llm_gpu_lab sft`                           |
| `sft_samples_before_after.json`   | qualitative inspection: same prompt before / after SFT |
| `sft_limitations.md` (optional)   | only present if the SFT path had to degrade           |
| `eval_results.json`               | `python -m llm_gpu_lab eval`                          |
| `benchmark_summary.json`          | `python -m llm_gpu_lab bench-gpu`                     |
| `gguf_export.json`                | `python -m llm_gpu_lab export-gguf`                   |
| `limitations.md`                  | global limitations across the pipeline                |
| `run_manifest.md`                 | timestamps, hardware, and the commit each run was on  |
| `report.html`                     | the final report                                      |

## What "real" means

Every metric, table, and chart in `report.html` — pretrain loss, SFT
loss, eval pass rate, GPU TFLOPS — is generated straight from the JSON
files above; none of it is mocked or interpolated.

The one exception is the report's **Limitations** section: it embeds
`limitations.md` / `sft_limitations.md` verbatim, and those two files
are hand-curated by whoever ran the pipeline (their own headers say so).
That includes the GGUF file-size / bits-per-weight figures — they are
transcribed from the quantizer's own console output, not parsed into a
JSON field yet. Everything else in the report has no hand-written
numbers in it.

If a number disagrees with what you expect, the right next step is:

1. Open the corresponding JSON.
2. Read the section in `docs/troubleshooting.md`.
3. Re-run the matching CLI command with `LLM_GPU_LAB_LOG_LEVEL=DEBUG`.

## Re-running on a different GPU

The pipeline writes into `results/<dir>/` — change the path with
`--out` flags or by editing the configs. Convention: name the
directory after the GPU model (`results/rtx_3080/`, `results/a40/`,
`results/7900_xtx/`). If you contribute a new directory back, we will
include it in the README benchmark table.
