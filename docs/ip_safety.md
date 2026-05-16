# IP and safety

This is a **clean-room** open-source project. It contains no proprietary
code, no employer-internal data, no private model weights, and no
recommendation-system content. It is intentionally generic — any developer
with one consumer GPU should be able to take it as a personal learning
project.

## What this repo does **not** contain

- No company-internal source code
- No internal architecture diagrams, metrics, configs, logs, dashboard
  links, or terminology that would be considered an employer's IP
- No proprietary or licensed-internal-only datasets
- No private fine-tuned model checkpoints
- No recommendation-system pipelines, ranking models, candidate stores,
  feature stores, or industry-specific business logic
- No code or content paraphrased from confidential sources

## What this repo *does* contain

- Generic LLM building blocks (decoder-only transformer, BPE tokenizer,
  training loop) implemented from public knowledge.
- Synthetic data generated procedurally inside this repo — no scraping,
  no copying.
- References to public open models and public open datasets, each cited
  with their Hugging Face repo IDs. See `docs/licenses.md`.

## Public datasets and models used

| Asset                                  | License / note                                  |
|----------------------------------------|-------------------------------------------------|
| `HuggingFaceTB/SmolLM2-135M-Instruct`  | Apache-2.0 (per the model card)                 |
| `Qwen/Qwen2.5-0.5B-Instruct`           | Apache-2.0 (per the model card)                 |
| `Qwen/Qwen3-0.6B`                      | Apache-2.0 (per the model card)                 |
| `roneneldan/TinyStories` (optional)    | Check the dataset card for the current licence  |

Downloaded weights and datasets are **never** committed. They live in
the Hugging Face cache on the user's machine and are .gitignore'd here.

## How development happened

- Personal hardware (one consumer RTX 4080 16 GB), personal time, no
  employer resources used.
- No internal repositories, internal source trees, or internal Slack /
  Confluence / wiki content was consulted while writing this code.
- Each commit is signed by an account that is not affiliated with any
  employer-managed identity.

## Reporting concerns

If you believe any file in this repository accidentally contains
copyrighted, proprietary, or otherwise sensitive content, open a
**private** GitHub Security Advisory on the repo. We will respond
quickly and remove anything that doesn't belong.
