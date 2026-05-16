# SFT limitations

This file is auto-generated when the SFT step has to degrade. It is
re-created on every run, so missing entries means nothing degraded this
time.

In the recorded run:

- The base model (`HuggingFaceTB/SmolLM2-135M-Instruct`) succeeded with
  plain LoRA (r=8, alpha=16, seq_len=256, 30 steps). No fallback to a
  smaller model was needed.
- `bitsandbytes` is not installed on this host, so QLoRA was not
  attempted. To exercise the QLoRA path, install the `quant` extra
  (`pip install -e ".[quant]"`) and use the
  `configs/sft/qwen3_0_6b_qlora_4080.yaml` config.
- No CUDA OOM events were observed, so no retry with reduced sequence
  length / LoRA rank was triggered.

See `limitations.md` for the global picture and `sft_metrics.json` for
the numbers.
