"""Auxiliary training targets derived from measured PPR(N).

g_obs is a smooth event trigger based on the cumulative maximum of measured PPR.
risk_proxy is the peak measured PPR, a continuous proximity-to-threshold target.
An optional CRR(N) target, such as one fitted from a six-specimen series, can be
provided as crr_obs with a specimen-level crr_obs_mask.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from dpi_evt.config import LIQ_THRESHOLD

__all__ = ["derive_observed_targets"]


def derive_observed_targets(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    liq_threshold: float = LIQ_THRESHOLD,
    kappa: float = 12.0,
) -> Dict[str, np.ndarray]:
    """Derive auxiliary targets from measured PPR(N).

    g_obs = sigmoid(kappa * (cummax(PPR) - threshold)) is nondecreasing.
    risk_proxy is the peak on the valid record, clipped to [0, 1].

    :param r_obs: measured PPR, shape (n, seq_len)
    :param valid_mask: measurement mask, shape (n, seq_len)
    :param liq_threshold: PPR event threshold, typically 0.9-1.0
    :param kappa: smooth-trigger steepness
    :return: g_obs with shape (n, seq_len) and risk_proxy with shape (n,)
    """
    r_obs = np.asarray(r_obs, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=np.float64)

    running_max = np.maximum.accumulate(r_obs, axis=1)
    g_obs = 1.0 / (1.0 + np.exp(-kappa * (running_max - liq_threshold)))

    masked = np.where(valid_mask > 0, r_obs, -np.inf)
    peak = masked.max(axis=1)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    risk_proxy = np.clip(peak, 0.0, 1.0)

    return {"g_obs": g_obs.astype(np.float32), "risk_proxy": risk_proxy.astype(np.float32)}
