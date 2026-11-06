"""Locate locally reproduced runs without pinning a checkout or run fingerprint."""
from pathlib import Path
import json
import os

def primary_run(root):
    root = Path(root).resolve()
    supplied = os.environ.get('DPI_PRIMARY_RUN')
    if supplied:
        p = Path(supplied).expanduser()
        return p.resolve() if p.is_absolute() else (root / p).resolve()
    latest = root / 'results/experiments/latest.json'
    if latest.is_file():
        p = Path(json.loads(latest.read_text())['path'])
        return p if p.is_absolute() else root / p
    return root / 'results/experiments/not-yet-run'

def interval_run(root):
    root = Path(root)
    p = root / 'results/onset_intervals/latest.json'
    if not p.is_file():
        raise FileNotFoundError('Run scripts/calibrate_intervals.py after the main experiment.')
    return root / json.loads(p.read_text())['path']
