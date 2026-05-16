# Contributing to llm-gpu-lab

Thanks for your interest! This project is an open-source educational toolkit
for running a complete LLM workflow on a single consumer NVIDIA GPU.

## Ground rules

- All code must run locally. We do not accept changes that require paid hosted
  inference (OpenAI / Anthropic / Gemini / etc.) for the core pipeline.
- Public open datasets and open model weights are welcome. Document their
  license in `docs/licenses.md` and never commit downloaded weights.
- No private data, no company-internal code, no recommendation-system content.
  See `docs/ip_safety.md`.
- Benchmark numbers in docs **must** come from real runs and committed
  `results/rtx4080/*.json` artifacts. Do not paste fake numbers.

## Dev setup

```bash
# 1. Create / activate a Python 3.10 or 3.11 venv (uv is the easiest)
uv venv --python 3.11 .venv

# 2. Install package with dev deps
.venv/Scripts/python -m pip install -e ".[dev]"

# 3. Install torch with the correct CUDA wheel — see docs/quickstart.md
#    Then install the rest: nlp, quant, hub
.venv/Scripts/python -m pip install -e ".[nlp,hub]"
```

## Before you submit a PR

```bash
make lint          # ruff check .
make test          # pytest -q
make audit         # placeholder audit
```

## Adding a new command

1. Implement the logic under `src/llm_gpu_lab/<area>/`.
2. Wire a Typer command in `src/llm_gpu_lab/cli.py`.
3. Add a config under `configs/<area>/` if the command takes one.
4. Add a Makefile target if it is part of the smoke workflow.
5. Add at least one unit test under `tests/`.
6. Update `README.md`'s command table and `docs/quickstart.md`.

## What goes in `docs/roadmap.md`

`docs/roadmap.md` is the **only** place where `TODO` / `FIXME` are allowed.
The placeholder audit script (`scripts/audit_placeholders.sh`) enforces this.

## Honest limitations

If something does not work on your hardware, document it under
`results/<gpu>/limitations.md` and link from `docs/troubleshooting.md`. Do not
delete code or silently disable steps — record the failure honestly so the
report tells the truth.
