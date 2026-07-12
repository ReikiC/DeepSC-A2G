# DeepSC — common operations.
# Run `make help` to list available targets.
# Tip: override the channel model with CHANNEL=AWGN|Rayleigh|Rician, e.g.
#       make train CHANNEL=AWGN

PY      := uv run python
SRC     := src
CHANNEL ?= Rayleigh

.DEFAULT_GOAL := help

.PHONY: help setup lock requirements preprocess train eval clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "DeepSC — make targets\n  pass CHANNEL=AWGN|Rayleigh|Rician to train/eval (default Rayleigh)\n\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Install dependencies and create .env from the template
	uv sync
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example — edit DATA_DIR inside."; fi

lock: ## Re-resolve and refresh uv.lock
	uv lock

requirements: ## Regenerate requirements.txt from the lock (pip fallback)
	uv export --no-dev --no-emit-project --no-hashes -o requirements.txt

preprocess: ## Preprocess the Europarl corpus into vocab + pkl datasets
	$(PY) $(SRC)/preprocess_text.py

train: ## Train DeepSC (e.g. make train CHANNEL=AWGN)
	$(PY) $(SRC)/main.py --channel $(CHANNEL)

eval: ## Evaluate BLEU vs SNR (e.g. make eval CHANNEL=AWGN)
	$(PY) $(SRC)/performance.py --channel $(CHANNEL)

clean: ## Remove Python caches (keeps checkpoints and .venv)
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
