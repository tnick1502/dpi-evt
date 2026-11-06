"""Read and write prepared datasets.

Each directory contains meta.parquet, arrays.npz, config.json and feature_names.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from dpi_evt.config import ExperimentConfig

__all__ = ["save_population_artifact", "load_population_artifact"]

_BENCHMARK_KEYS = ("benchmark_idx", "train_rel", "val_rel", "test_rel")
_CRR_KEYS = ("hyperbolic", "power", "exponential", "logarithmic")
_NON_ARRAY_KEYS = (
    "meta",
    "benchmark",
    "crr_families",
    "static_feature_names",
    "prefix_summary_names",
    "seq_feature_names",
    "cohort_filter_counts",
)


def save_population_artifact(artifact_dir: str | Path, population: Dict[str, Any], config: ExperimentConfig) -> None:
    """Save a prepared dataset.

    Write numeric arrays to compressed arrays.npz and metadata to meta.parquet.
    Flatten nested split and CRR-family dictionaries using benchmark_* and
    crr_family_* prefixes.

    :param artifact_dir: output directory, created if missing
    :param population: dataset dictionary
    :param config: preparation configuration
    :return: None
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    population["meta"].to_parquet(artifact_dir / "meta.parquet")
    (artifact_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    arrays: Dict[str, np.ndarray] = {}
    for key, value in population.items():
        if key in _NON_ARRAY_KEYS:
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value

    bench = population["benchmark"]
    for bk in _BENCHMARK_KEYS:
        arrays[f"benchmark_{bk}"] = bench[bk]

    crr_f = population.get("crr_families")
    if crr_f is not None:
        for ck in _CRR_KEYS:
            arrays[f"crr_family_{ck}"] = crr_f[ck]

    np.savez_compressed(artifact_dir / "arrays.npz", **arrays)

    feature_names = {
        "static_feature_names": population["static_feature_names"],
        "prefix_summary_names": population["prefix_summary_names"],
        "seq_feature_names": population["seq_feature_names"],
    }
    (artifact_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    if "cohort_filter_counts" in population:
        (artifact_dir / "cohort_filter_counts.json").write_text(
            json.dumps(population["cohort_filter_counts"], indent=2), encoding="utf-8"
        )


def load_population_artifact(artifact_dir: str | Path) -> Tuple[Dict[str, Any], ExperimentConfig]:
    """Load a prepared dataset saved by save_population_artifact.

    Restore metadata, numeric arrays, nested split/CRR-family dictionaries and settings.

    :param artifact_dir: saved dataset directory
    :return: (dataset dictionary, experiment configuration)
    """
    artifact_dir = Path(artifact_dir)
    raw = np.load(artifact_dir / "arrays.npz")

    def arr(name: str) -> np.ndarray:
        return raw[name]

    config_dict = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(**config_dict)

    feature_names = json.loads((artifact_dir / "feature_names.json").read_text(encoding="utf-8"))

    benchmark = {bk: arr(f"benchmark_{bk}") for bk in _BENCHMARK_KEYS}
    skip = {f"benchmark_{bk}" for bk in _BENCHMARK_KEYS} | {f"crr_family_{ck}" for ck in _CRR_KEYS}

    population: Dict[str, Any] = {
        "meta": pd.read_parquet(artifact_dir / "meta.parquet"),
        "benchmark": benchmark,
        "static_feature_names": feature_names["static_feature_names"],
        "prefix_summary_names": feature_names["prefix_summary_names"],
        "seq_feature_names": feature_names["seq_feature_names"],
    }
    cohort_path = artifact_dir / "cohort_filter_counts.json"
    if cohort_path.exists():
        population["cohort_filter_counts"] = json.loads(cohort_path.read_text(encoding="utf-8"))

    if all(f"crr_family_{ck}" in raw.files for ck in _CRR_KEYS):
        population["crr_families"] = {ck: arr(f"crr_family_{ck}") for ck in _CRR_KEYS}

    for key in raw.files:
        if key in skip:
            continue
        population[key] = raw[key]

    return population, config
