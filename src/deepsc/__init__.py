# -*- coding: utf-8 -*-
"""DeepSC — PyTorch implementation of Xie et al. 2021 semantic communication.

The package is installed editable (see pyproject.toml), so
``from deepsc.models import DeepSC`` works from any working directory without
needing PYTHONPATH hacks.
"""
__version__ = "0.1.0"

from deepsc.config import DATA_DIR, device  # noqa: F401  (convenience re-export)
