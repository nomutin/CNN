# CNN

![python](https://img.shields.io/badge/python-3.8-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

Small project for pre-training of CNN encoder/decoder.

## API

```python
from cnn import VAE

vae = VAE.load_from_wandb(reference=<wandb reference>)
```

## Installation

### pip

```shell
pip install git+https://github.com/nomutin/CNN.git
```

### poetry

```shell
poetry add git+https://github.com/nomutin/CNN.git
```

### rye

```shell
rye add cnn --git=https://github.com/nomutin/CNN.git
```
