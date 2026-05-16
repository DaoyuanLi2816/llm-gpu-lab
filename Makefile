SHELL := /usr/bin/env bash
PY ?= python
PIP ?= $(PY) -m pip
RESULTS := results/rtx4080

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup            install package with dev deps"
	@echo "  doctor           run env diagnostics"
	@echo "  tokenizer        train tiny BPE tokenizer"
	@echo "  pretrain-smoke   tiny GPT pretraining smoke run"
	@echo "  generate-smoke   generate text from pretrained checkpoint"
	@echo "  sft-smoke        LoRA/QLoRA smoke run"
	@echo "  eval-smoke       basic eval on tiny model"
	@echo "  report           build HTML report"
	@echo "  smoke            full end-to-end smoke pipeline"
	@echo "  bench            run GPU benchmarks"
	@echo "  test             pytest"
	@echo "  lint             ruff"
	@echo "  audit            placeholder audit"
	@echo "  setup-llamacpp   clone + build llama.cpp"
	@echo "  gguf-smoke       attempt GGUF export of fine-tuned model"
	@echo "  clean-artifacts  remove artifacts/ and re-create"

.PHONY: setup
setup:
	$(PIP) install -e ".[dev]"
	@echo "Install torch + nlp extras manually — see docs/quickstart.md"

.PHONY: doctor
doctor:
	mkdir -p $(RESULTS)
	$(PY) -m llm_gpu_lab doctor --out $(RESULTS)/environment.json

.PHONY: tokenizer
tokenizer:
	$(PY) -m llm_gpu_lab train-tokenizer --config configs/pretrain/tiny_10m_smoke.yaml

.PHONY: pretrain-smoke
pretrain-smoke:
	$(PY) -m llm_gpu_lab pretrain --config configs/pretrain/tiny_10m_smoke.yaml

.PHONY: generate-smoke
generate-smoke:
	$(PY) -m llm_gpu_lab generate \
	  --checkpoint artifacts/checkpoints/tiny_10m_smoke/final.safetensors \
	  --prompts examples/prompts/generation_prompts.txt \
	  --out $(RESULTS)/generation_samples.json

.PHONY: sft-smoke
sft-smoke:
	$(PY) -m llm_gpu_lab sft --config configs/sft/smollm2_135m_lora_fallback.yaml

.PHONY: eval-smoke
eval-smoke:
	$(PY) -m llm_gpu_lab eval --config configs/eval/smoke_eval.yaml

.PHONY: report
report:
	$(PY) -m llm_gpu_lab report --results-dir $(RESULTS) --out $(RESULTS)/report.html

.PHONY: bench
bench:
	$(PY) -m llm_gpu_lab bench-gpu --out $(RESULTS)/benchmark_summary.json

.PHONY: smoke
smoke: doctor tokenizer pretrain-smoke generate-smoke sft-smoke eval-smoke bench report audit

.PHONY: test
test:
	$(PY) -m pytest -q

.PHONY: lint
lint:
	$(PY) -m ruff check .

.PHONY: audit
audit:
	bash scripts/audit_placeholders.sh

.PHONY: setup-llamacpp
setup-llamacpp:
	bash scripts/setup_llamacpp.sh

.PHONY: gguf-smoke
gguf-smoke:
	$(PY) -m llm_gpu_lab export-gguf --config configs/export/gguf_q4_k_m.yaml

.PHONY: clean-artifacts
clean-artifacts:
	rm -rf artifacts/
	mkdir -p artifacts/checkpoints artifacts/adapters artifacts/tokenizers
