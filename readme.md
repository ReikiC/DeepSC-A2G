# Deep Learning Enabled Semantic Communication Systems

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of  Deep learning enabled semantic communication systems.

## Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management (Python 3.11+).

```shell
uv sync                     # install the deepsc package (editable) into .venv
uv run python scripts/train.py   # entry points live in scripts/
```

Dependencies are pinned in `uv.lock` for reproducibility. `requirements.txt` is
generated from the lock as a fallback for plain `pip`
(`pip install -r requirements.txt`).

> **PyTorch backend:** `torch` is pulled from PyPI, which provides the CPU/MPS
> build on macOS (for local experimentation) and the CUDA build on Linux (for
> training on a rented GPU server) — no extra index configuration required.

### Data path

The dataset root is read from the `DATA_DIR` environment variable (see
[`src/deepsc/config.py`](src/deepsc/config.py)). Copy `.env.example` to `.env` and point it at
the folder that contains `europarl/`:

```shell
cp .env.example .env        # then edit DATA_DIR in .env
```

## Make targets

Common operations are wrapped as [`make`](Makefile) targets — run `make help` to list them.

```shell
make setup        # uv sync + create .env from the template
make preprocess   # preprocess the Europarl corpus
make train        # train (e.g. make train CHANNEL=AWGN)
make eval         # evaluate BLEU vs SNR
make clean        # remove Python caches
```

The `CHANNEL` variable selects the channel model — `AWGN`, `Rayleigh`, or `Rician` (default `Rayleigh`).

## Smoke test (quickstart)

Fastest end-to-end path to a result — train for a couple of epochs, then plot
BLEU-1 vs. SNR on a 2000-sentence subset. The numbers are rough (2 epochs is
nowhere near converged); this only proves the whole pipeline runs.

> Prerequisite: preprocess the Europarl corpus once first (`make preprocess`,
> see [Preprocess](#preprocess)).

```shell
# 1. Train for 2 epochs only (default is 80). `--epochs` isn't wired through
#    `make train`, so call the script directly.
uv run python scripts/train.py --channel Rayleigh --epochs 2

# 2. Quick BLEU-vs-SNR preview on the first 2000 test sentences; save the log
#    so plot_curve.py can read it. Loads the latest checkpoint under
#    checkpoints/deepsc-Rayleigh/.
make quick-eval > data/quick_eval.log

# 3. Plot the curve -> figures/bleu_vs_snr.png
make plot
```

[`tools/quick_eval.py`](tools/quick_eval.py) reuses the same `greedy_decode` +
BLEU-1 path as [`evaluate.py`](scripts/evaluate.py) but on a subset, so the
curve shape lands in minutes instead of ~2h. For full-fidelity numbers, run
`make eval` in the background instead of step 2.

## Bibtex
```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}
```
## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
uv run python scripts/preprocess.py
```

## Train
```shell
uv run python scripts/train.py
```
### Notes
+ Please carefully set the $\lambda$ of mutual information part since I have tested the model in different platform, 
i.e., Tensorflow and Pytorch, same $\lambda$ shows different performance.  

## Evaluation
```shell
uv run python scripts/evaluate.py
```
### Notes
+ If you want to compute the sentence similarity, please download the bert model.
