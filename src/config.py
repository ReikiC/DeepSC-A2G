# -*- coding: utf-8 -*-
"""
Centralized configuration.

Loads variables from a local .env file (see .env.example) so that the
dataset root path no longer has to be hardcoded across the codebase.
"""
import os
from dotenv import load_dotenv

# Load .env from the project root if present.
load_dotenv()

# Root directory holding the europarl/ data. Keep the trailing slash.
DATA_DIR = os.environ.get('DATA_DIR', '/import/antennas/Datasets/hx301/')
