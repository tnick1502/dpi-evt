"""Paths for public external data; no machine-specific fallbacks."""
from pathlib import Path
import os
REPO_ROOT = Path(os.environ.get('DPI_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
RAW = Path(os.environ.get('DPI_EXTERNAL_DATA', REPO_ROOT / 'data/external'))
RESULTS = REPO_ROOT / 'results/external'
