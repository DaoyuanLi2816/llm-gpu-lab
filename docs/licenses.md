# Licenses

## This repository

The source code, configs, scripts, docs, and the synthetic data
generators are licensed under **Apache-2.0**. See `LICENSE`.

Apache-2.0 was chosen because the project links against several other
Apache-2.0 libraries (`transformers`, `peft`, `trl`, `tokenizers`,
`safetensors`). Using the same licence avoids any subtle conflict at the
combination level and makes contribution rules easy to understand. MIT
would also have been acceptable.

## Major third-party Python packages

| Package        | Typical licence                                        |
|----------------|--------------------------------------------------------|
| `torch`        | BSD-style ("PyTorch License")                          |
| `transformers` | Apache-2.0                                             |
| `peft`         | Apache-2.0                                             |
| `trl`          | Apache-2.0                                             |
| `tokenizers`   | Apache-2.0                                             |
| `datasets`     | Apache-2.0                                             |
| `accelerate`   | Apache-2.0                                             |
| `safetensors`  | Apache-2.0                                             |
| `bitsandbytes` | MIT                                                    |
| `pynvml` (`nvidia-ml-py`) | BSD-3-Clause                                |
| `pydantic`     | MIT                                                    |
| `typer`        | MIT                                                    |
| `rich`         | MIT                                                    |
| `numpy`        | BSD-3-Clause                                           |
| `pandas`       | BSD-3-Clause                                           |
| `matplotlib`   | PSF-based (matplotlib licence)                         |
| `jinja2`       | BSD-3-Clause                                           |
| `lm-eval`      | MIT                                                    |
| `huggingface_hub`| Apache-2.0                                           |

These are the licences at the time of writing; always check the
upstream `LICENSE` file of the version you actually install.

## Downloaded model weights

We do not redistribute any model weights. They are downloaded by the
Hugging Face Hub client on demand and stay in the user's HF cache. Each
model's licence is published on its model card:

- `HuggingFaceTB/SmolLM2-135M-Instruct` — model card states Apache-2.0.
- `Qwen/Qwen2.5-0.5B-Instruct` — model card states Apache-2.0.
- `Qwen/Qwen3-0.6B` — model card states Apache-2.0.

If you fine-tune one of these models with `llm-gpu-lab`, the resulting
LoRA adapter is **your** derivative work — you choose the licence you
publish it under (subject to the base model's licence terms).

## Datasets

For the default `synthetic` data source, the corpus is generated locally
by `src/llm_gpu_lab/data/tiny_corpus.py` and is original to this repo.
It is released under the project's Apache-2.0 licence.

For the optional `tinystories` data source we use the public Hugging
Face dataset `roneneldan/TinyStories`. The license is documented on
the dataset card; please check it before redistributing any derivative
work. We do **not** vendor any TinyStories text in this repo.

## llama.cpp

`scripts/setup_llamacpp.sh` clones `ggerganov/llama.cpp` into
`external/llama.cpp/`. That directory is .gitignore'd — we don't
redistribute llama.cpp source. Its MIT licence is preserved in its
own repository.

## What is and is not committed

Committed:

- All source files under `src/`, `tests/`, `configs/`, `scripts/`,
  `docs/`, and `examples/`.
- Small JSON / HTML / MD artifacts under `results/<gpu>/` that prove
  the pipeline ran on a real machine.

NOT committed (.gitignore'd):

- `artifacts/checkpoints/`, `artifacts/adapters/`, `artifacts/tokenizers/`,
  `artifacts/merged/`, `artifacts/gguf/` — these can grow past 1 GB.
- `external/` — third-party clones (llama.cpp, etc.).
- `.cache/`, `~/.cache/huggingface/`, virtualenvs, `*.gguf`, `*.safetensors`,
  `*.pt`, `*.bin`.
