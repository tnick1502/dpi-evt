"""Trajectory-induced event distributions and train-referenced censored scoring.

These functions are used for evaluation only. A path without a crossing has T=inf, which differs
from a crossing exactly at H. The DPI posterior samples latent trajectories without adding Gaussian
emission noise to them.
"""
from __future__ import annotations

import numpy as np
import torch


def hitting_time_distribution(paths: torch.Tensor, cycles: torch.Tensor,
                              threshold: float = 0.95) -> dict[str, torch.Tensor]:
    """paths: (samples, batch, time), cycles: (batch, time). Exact empirical pushforward."""
    if paths.ndim != 3 or paths.shape[1:] != cycles.shape:
        raise ValueError("Expected paths=(M,B,T), cycles=(B,T)")
    if not torch.isfinite(paths).all() or not torch.isfinite(cycles).all():
        raise ValueError("Nonfinite path or query time")
    if (torch.diff(cycles, dim=-1) <= 0).any():
        raise ValueError("Query times must increase strictly")
    if (torch.diff(paths, dim=-1) < -1e-6).any():
        raise ValueError("Event pushforward requires monotone paths")
    if ((paths < -1e-6) | (paths > 1 + 1e-6)).any():
        raise ValueError("Event pushforward requires paths in [0,1]")
    hit = paths >= threshold
    has = hit.any(dim=-1)
    j = hit.to(torch.int64).argmax(dim=-1)
    c = cycles.unsqueeze(0).expand_as(paths)
    hi = c.gather(-1, j[..., None]).squeeze(-1)
    lo_j = (j - 1).clamp_min(0)
    lo = c.gather(-1, lo_j[..., None]).squeeze(-1)
    yhi = paths.gather(-1, j[..., None]).squeeze(-1)
    ylo = paths.gather(-1, lo_j[..., None]).squeeze(-1)
    frac = ((threshold - ylo) / (yhi - ylo).clamp_min(1e-12)).clamp(0, 1)
    times = torch.where(j == 0, hi, lo + frac * (hi - lo))
    times = torch.where(has, times, torch.full_like(times, float("inf")))
    cdf = hit.to(paths.dtype).mean(dim=0)
    ordered = times.sort(dim=0).values
    def quantile(q):
        # Inverse empirical CDF; avoid interpolating finite values with infinity.
        return ordered[max(0, int(np.ceil(q * len(paths))) - 1)]
    return {
        "event_cdf": cdf,
        "event_survival": 1 - cdf,
        "event_tail_mass": 1 - cdf[:, -1],
        "event_time_samples": times.transpose(0, 1),
        "event_time_median": quantile(.5),
        "nliq_q05": quantile(.05),
        "nliq_q95": quantile(.95),
        "event_q25": quantile(.25),
        "event_q75": quantile(.75),
    }


def native_survival_scores(cdf, query_times, time, event, train_time, train_event,
                           landmark: float = 0.0, min_censor_survival: float = .05):
    """IPCW Brier/IBS using the training censoring KM, including the initial interval.

    Integrate only the common train-supported interval [N0,tau], stopping before G vanishes.
    Ties between censoring and events use G(T-) for observed events. The model CDF is scored
    directly. Returns per-specimen integrated contributions and aggregate metrics.
    """
    from dpi_evt.evaluation.metrics import _km_censoring_survival
    cdf, q = np.asarray(cdf, float), np.asarray(query_times, float)
    if q.ndim == 2:
        if not np.allclose(q, q[:1]):
            raise ValueError("Native survival scoring requires common query times")
        q = q[0]
    if cdf.shape != (len(time), len(q)):
        raise ValueError("CDF shape does not match subjects/query times")
    if not np.isfinite(cdf).all() or np.any(np.diff(cdf, axis=1) < -1e-6):
        raise ValueError("Invalid event CDF")
    if np.any((cdf < -1e-6) | (cdf > 1 + 1e-6)):
        raise ValueError("CDF outside [0,1]")
    grid, G = _km_censoring_survival(np.asarray(train_time), np.asarray(train_event))
    def at(t, left=False):
        j = np.searchsorted(grid, t, side="left" if left else "right") - 1
        return np.where(j < 0, 1.0, G[np.maximum(j, 0)])
    # Include N0 and omit times with insufficient censoring support. Event mass that an
    # unanchored model already predicts by N0 is scored as predicted.
    keep = (q > landmark) & (at(q) >= min_censor_survival)
    tgrid = np.r_[landmark, q[keep]]
    early = np.flatnonzero(q <= landmark)
    initial = 1 - cdf[:, early[-1]] if len(early) else np.ones(len(time))
    S = np.c_[initial, 1 - cdf[:, keep]]
    if len(tgrid) < 2:
        return {"N_liq_IBS": np.nan, "IBS_tau": np.nan}, np.full(len(time), np.nan)
    time, event = np.asarray(time, float), np.asarray(event, bool)
    scores = np.zeros_like(S)
    for k, tt in enumerate(tgrid):
        failed = (time <= tt) & event
        alive = time > tt
        scores[failed, k] = S[failed, k] ** 2 / at(time[failed], left=True).clip(min_censor_survival)
        scores[alive, k] = (1 - S[alive, k]) ** 2 / max(float(at(tt)), min_censor_survival)
    dt = np.diff(tgrid)
    per_test = ((scores[:, 1:] + scores[:, :-1]) * .5 * dt).sum(1) / (tgrid[-1] - landmark)
    return {"N_liq_IBS": float(per_test.mean()), "IBS_tau": float(tgrid[-1]),
            "IBS_start": float(landmark)}, per_test


def concordance_ipcw(pred_time, time, event, train_time, train_event, min_censor_survival=.05):
    """Uno concordance with training censoring weights and strictly later comparable records."""
    from dpi_evt.evaluation.metrics import _km_censoring_survival
    grid, G = _km_censoring_survival(np.asarray(train_time), np.asarray(train_event))
    p, t, e = np.asarray(pred_time), np.asarray(time), np.asarray(event, bool)
    num = den = 0.
    for i in np.flatnonzero(e):
        j = np.searchsorted(grid, t[i], side="left") - 1
        g = 1. if j < 0 else G[j]
        later = t > t[i]
        if g < min_censor_survival or not later.any():
            continue
        w = 1 / g ** 2
        num += w * ((p[i] < p[later]).sum() + .5 * (p[i] == p[later]).sum())
        den += w * later.sum()
    return float(num / den) if den else np.nan
