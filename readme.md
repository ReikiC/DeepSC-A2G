# Deep Learning Enabled Semantic Communication Systems

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of  Deep learning enabled semantic communication systems.

## Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management (Python 3.11+).

```shell
uv sync                     # install dependencies into .venv
uv run python src/main.py   # run scripts via uv (venv is used automatically)
```

Dependencies are pinned in `uv.lock` for reproducibility. `requirements.txt` is
generated from the lock as a fallback for plain `pip`
(`pip install -r requirements.txt`).

> **PyTorch backend:** `torch` is pulled from PyPI, which provides the CPU/MPS
> build on macOS (for local experimentation) and the CUDA build on Linux (for
> training on a rented GPU server) — no extra index configuration required.

### Data path

The dataset root is read from the `DATA_DIR` environment variable (see
[`src/config.py`](src/config.py)). Copy `.env.example` to `.env` and point it at
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
uv run python src/preprocess_text.py
```

## Train
```shell
uv run python src/main.py
```
### Notes
+ Please carefully set the $\lambda$ of mutual information part since I have tested the model in different platform, 
i.e., Tensorflow and Pytorch, same $\lambda$ shows different performance.  

## Evaluation
```shell
uv run python src/performance.py
```
### Notes
+ If you want to compute the sentence similarity, please download the bert model.
