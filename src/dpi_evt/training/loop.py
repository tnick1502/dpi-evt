"""Shared training loop for models implementing compute_loss(batch).

Use AdamW, gradient clipping, epochwise validation and the best validation
checkpoint. Optional validation metrics provide learning curves.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dpi_evt.config import ExperimentConfig, set_global_seed
from dpi_evt.data.splits import iterate_minibatches
from dpi_evt.training.losses import clone_state_dict

__all__ = ["train_model", "evaluate_epoch_metrics"]


def evaluate_epoch_metrics(
    model: nn.Module,
    split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Calculate validation metrics for training monitoring.

    Include event AUROC/Brier and pressure RMSE when a trajectory head is present.

    :param model: current model
    :param split: validation partition
    :param config: batch settings
    :param device: inference device
    :return: val_auroc, val_brier and optional val_traj_rmse
    """
    from dpi_evt.evaluation.metrics import collect_outputs, safe_binary_metrics

    outputs = collect_outputs(model, split, config, device)
    y_true = split["label"].cpu().numpy()
    auroc, _, brier = safe_binary_metrics(y_true, outputs["risk_prob"])
    metrics: Dict[str, float] = {"val_auroc": auroc, "val_brier": brier}

    if "traj_mean" in outputs:
        pred = outputs["traj_mean"]
        # Compare the predicted trajectory with measured PPR, r_obs.
        true_key = "r_true" if "r_true" in split else "r_obs"
        true = split[true_key].cpu().numpy()
        mask = split["mask"].cpu().numpy()
        mse = float(np.sum(((pred - true) ** 2) * mask) / np.maximum(mask.sum(), 1.0))
        metrics["val_traj_rmse"] = float(np.sqrt(mse))
    return metrics


def _val_loss(model: nn.Module, val_split: Dict[str, object], config: ExperimentConfig, device: torch.device) -> float:
    """Calculate the current model mean validation loss."""
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in iterate_minibatches(val_split, config.batch_size, device, shuffle=False):
            losses.append(float(model.compute_loss(batch)["loss"].detach().cpu()))
    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    epochs: int,
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
    track_metrics: bool = False,
    verbose: bool = True,
    scheduler: str = "none",
    ema_decay: float = 0.0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train a model and restore its best validation checkpoint.

    Each epoch runs mini-batch backpropagation with gradient-norm clipping at 1.0,
    then evaluates validation loss. Optional cosine scheduling uses warm-up. With
    EMA enabled, checkpoint selection evaluates averaged weights.

    :param model: torch model implementing compute_loss(batch) with a loss output
    :param train_split: training partition
    :param val_split: validation partition
    :param epochs: epoch limit
    :param model_name: log display name
    :param config: learning rate, weight decay, batch size and seed
    :param device: training device
    :param track_metrics: record validation AUROC/Brier/RMSE each epoch
    :param verbose: print epoch progress
    :param scheduler: none or cosine with short warm-up
    :param ema_decay: weight EMA decay; zero disables it, typical values 0.998-0.999
    :param early_stopping_patience: epochs without improvement; None uses config, zero disables it
    :param early_stopping_min_delta: minimum validation improvement; None uses config
    :return: best-state model and training-history DataFrame
    """
    # Reset the project seed before fitting each model
    # to make its random stream independent of invocation order.
    set_global_seed(config.seed)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.05)\
        if scheduler == "cosine" else None
    use_ema = ema_decay > 0.0
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if use_ema else None

    best_state = clone_state_dict(model)
    best_val = float("inf")
    best_epoch = 0
    patience = config.early_stopping_patience if early_stopping_patience is None else early_stopping_patience
    min_delta = config.early_stopping_min_delta if early_stopping_min_delta is None else early_stopping_min_delta
    patience = int(patience or 0)
    min_delta = float(min_delta or 0.0)
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    warmup_epochs = 1

    for epoch in range(1, epochs + 1):
        # Linear learning-rate warm-up in the first epoch.
        if scheduler == "cosine" and epoch <= warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * epoch / max(warmup_epochs, 1)

        model.train()
        train_losses: List[float] = []
        for batch in iterate_minibatches(train_split, config.batch_size, device, shuffle=True, seed=config.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.compute_loss(batch)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss_dict["loss"].detach().cpu()))
            if use_ema:
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if value.dtype.is_floating_point:
                            ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                        else:
                            ema_state[key].copy_(value)

        if lr_sched is not None and epoch > warmup_epochs:
            lr_sched.step()

        # Evaluate averaged weights when EMA is enabled, restoring state afterward.
        if use_ema:
            backup = clone_state_dict(model)
            model.load_state_dict(ema_state)
        train_loss = float(np.mean(train_losses))
        val_loss = _val_loss(model, val_split, config, device)
        record: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if track_metrics:
            record.update(evaluate_epoch_metrics(model, val_split, config, device))
        history.append(record)
        improved = val_loss < (best_val - min_delta)
        record["best_val_loss"] = min(best_val, val_loss)
        record["epochs_without_improvement"] = stale_epochs
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        record["best_epoch"] = best_epoch
        record["epochs_without_improvement"] = stale_epochs
        if use_ema:
            model.load_state_dict(backup)

        if verbose:
            extra = ""
            if track_metrics:
                extra = f" | val_AUROC={record.get('val_auroc', float('nan')):.3f}"
                if "val_traj_rmse" in record:
                    extra += f" | val_RMSE={record['val_traj_rmse']:.4f}"
            stop_note = f" | stale={stale_epochs}/{patience}" if patience > 0 else ""
            print(f"[{model_name}] epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}{extra}{stop_note}", flush=True)

        if patience > 0 and stale_epochs >= patience:
            if verbose:
                print(
                    f"[{model_name}] early stopping at epoch {epoch:02d}; "
                    f"best val_loss={best_val:.4f} at epoch {best_epoch:02d}",
                    flush=True,
                )
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
