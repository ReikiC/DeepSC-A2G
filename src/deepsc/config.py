# -*- coding: utf-8 -*-
"""Centralized runtime configuration.

Single source of truth for the two values that used to be redefined in every
script:

  - ``DATA_DIR`` : dataset root, loaded from the local ``.env`` (see
    ``.env.example``); must contain an ``europarl/`` folder.
  - ``device``   : the torch device to run on (cuda > mps > cpu).

Every other module imports these from here instead of re-defining them.
"""
import os

import torch
from dotenv import load_dotenv

# Load .env from the project root if present.
load_dotenv()

# Root directory holding the europarl/ data. Keep the trailing slash.
DATA_DIR = os.environ.get('DATA_DIR')

# Compute device once: prefer CUDA, fall back to Apple MPS, then CPU.
device = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
