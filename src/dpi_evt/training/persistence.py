"""Read and write fitted models.

Each directory contains weights.pt (state_dict), hyperparams.json (class,
constructor and training settings), and history.parquet (epochwise train/val history).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

__all__ = ["save_trained_model", "load_model_metadata", "load_weights_into", "read_hyperparams"]


def read_hyperparams(models_root: str | Path, name: str) -> Dict[str, Any]:
    """Read saved model parameters from hyperparams.json.

    :param models_root: model-root directory
    :param name: model subdirectory name
    :return: parameter dictionary
    """
    path = Path(models_root) / name / "hyperparams.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_trained_model(
    model: nn.Module,
    models_root: str | Path,
    name: str,
    hyperparams: Dict[str, Any],
    history: pd.DataFrame,
) -> Path:
    """Save a fitted model, its settings and training history.

    :param model: fitted torch model
    :param models_root: model-root directory
    :param name: model subdirectory name
    :param hyperparams: settings, preferably including model_type and model_kwargs
    :param history: DataFrame with epoch, train_loss and val_loss
    :return: saved model directory
    """
    out_dir = Path(models_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2, ensure_ascii=False), encoding="utf-8")
    history.to_parquet(out_dir / "history.parquet")
    return out_dir


def load_model_metadata(models_root: str | Path, name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load model parameters and training history.

    :param models_root: model-root directory
    :param name: model subdirectory name
    :return: parameter dictionary and history DataFrame
    """
    model_dir = Path(models_root) / name
    hyperparams = json.loads((model_dir / "hyperparams.json").read_text(encoding="utf-8"))
    history = pd.read_parquet(model_dir / "history.parquet")
    return hyperparams, history


def load_weights_into(model: nn.Module, models_root: str | Path, name: str, device: torch.device) -> nn.Module:
    """Load saved weights into an existing matching architecture.

    :param model: matching model instance
    :param models_root: model-root directory
    :param name: model subdirectory name
    :param device: target device
    :return: model with loaded weights in evaluation mode
    """
    weights_path = Path(models_root) / name / "weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        # A mismatch between hyperparams.json and weights.pt raises shape errors.
        # Report the first incompatible tensor and the number of mismatches.
        cur = model.state_dict()
        mismatches = [
            (k, tuple(v.shape), tuple(cur[k].shape))
            for k, v in state.items()
            if k in cur and tuple(cur[k].shape) != tuple(v.shape)
        ]
        if mismatches:
            key, ckpt_shape, model_shape = mismatches[0]
            raise RuntimeError(
                f"Weights of '{name}' do not match hyperparams.json: "
                f"weights.pt has {key}={ckpt_shape}, the constructed model has {model_shape} "
                f"({len(mismatches)} mismatched tensors). "
                f"hyperparams.json was probably changed after the weights were saved. "
                f"Refit '{name}' so that weights.pt and hyperparams.json agree."
            ) from err
        raise
    model.to(device)
    model.eval()
    return model
