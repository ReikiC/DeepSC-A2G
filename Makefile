# DeepSC — common operations.
# Run `make help` to list available targets.
# Tip: override the channel model with CHANNEL=AWGN|Rayleigh|Rician, e.g.
#       make train CHANNEL=AWGN

PY      := uv run python
SCRIPTS := scripts
TOOLS   := tools
CHANNEL ?= Rayleigh

.DEFAULT_GOAL := help

.PHONY: help setup lock requirements preprocess train eval quick-eval plot clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "DeepSC — make targets\n  pass CHANNEL=AWGN|Rayleigh|Rician to train/eval (default Rayleigh)\n\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Install the package (editable) and create .env from the template
	uv sync
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example — edit DATA_DIR inside."; fi

lock: ## Re-resolve and refresh uv.lock
	uv lock

requirements: ## Regenerate requirements.txt from the lock (pip fallback)
	uv export --no-dev --no-emit-project --no-hashes -o requirements.txt

preprocess: ## Preprocess the Europarl corpus into vocab + pkl datasets
	$(PY) $(SCRIPTS)/preprocess.py

train: ## Train DeepSC (e.g. make train CHANNEL=AWGN)
	$(PY) $(SCRIPTS)/train.py --channel $(CHANNEL)

eval: ## Evaluate BLEU vs SNR on the full test set (e.g. make eval CHANNEL=AWGN)
	$(PY) $(SCRIPTS)/evaluate.py --channel $(CHANNEL)

quick-eval: ## Quick BLEU-vs-SNR preview on a 2000-sentence subset
	$(PY) $(TOOLS)/quick_eval.py --channel $(CHANNEL)

plot: ## Plot BLEU-vs-SNR from data/quick_eval.log into figures/
	$(PY) $(TOOLS)/plot_curve.py

clean: ## Remove Python caches (keeps checkpoints and .venv)
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
