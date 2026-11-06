"""Classical survival baselines for right-censored liquefaction onset."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["CoxPHBaseline"]


class CoxPHBaseline:
    """L2-regularized Cox proportional-hazards baseline with Breslow ties.

    The model consumes the same fold-standardized static and causal-prefix summaries as tabular
    baselines. Every liquefied test is an event; every eligible non-liquefied test contributes a
    right-censoring time. It therefore estimates the same censored event-time target as DPI-EVT.
    """

    def __init__(self, static_dim: int, prefix_dim: int, max_cycle_reference: float = 3000.0,
                 l2: float = 1e-3, max_iter: int = 500):
        self.static_dim = int(static_dim)
        self.prefix_dim = int(prefix_dim)
        self.max_cycle_reference = float(max_cycle_reference)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.coef_: np.ndarray | None = None
        self.event_times_: np.ndarray | None = None
        self.cum_baseline_hazard_: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def eval(self):
        return self

    def to(self, device):
        return self

    @staticmethod
    def _features(batch: Dict[str, torch.Tensor]) -> np.ndarray:
        return torch.cat([batch["static"], batch["prefix_summary"]], dim=-1).detach().cpu().numpy()

    @staticmethod
    def _arrays(split):
        X = CoxPHBaseline._features(split).astype(np.float64, copy=False)
        time = split["n_liq_true"].detach().cpu().numpy().astype(np.float64)
        event = (split["label"].detach().cpu().numpy() > 0.5).astype(np.float64)
        valid_t = split.get("nliq_censor_valid", split.get("n_liq_observed"))
        valid = (valid_t.detach().cpu().numpy() > 0.5) if valid_t is not None else np.ones(len(time), bool)
        return X[valid], time[valid], event[valid]

    def fit(self, train_split: Dict[str, torch.Tensor], val_split=None) -> "CoxPHBaseline":
        X, time, event = self._arrays(train_split)
        if len(X) == 0 or int(event.sum()) == 0:
            raise ValueError("CoxPH requires at least one valid event-time observation and one event")
        # Standardize static/prefix features within the training fold
        # to condition partial-likelihood optimization.
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-6] = 1.0
        X = (X - self._mean) / self._std
        event_times = np.unique(time[event > 0.5])
        n_events = max(float(event.sum()), 1.0)

        def objective(beta):
            eta = X @ beta
            nll = 0.0
            grad = np.zeros_like(beta)
            for t in event_times:
                at_event = (time == t) & (event > 0.5)
                risk = time >= t
                d = int(at_event.sum())
                er = eta[risk]
                logden = logsumexp(er)
                weights = np.exp(er - logden)
                nll -= eta[at_event].sum() - d * logden
                grad -= X[at_event].sum(axis=0) - d * (weights[:, None] * X[risk]).sum(axis=0)
            nll = nll / n_events + 0.5 * self.l2 * float(beta @ beta)
            grad = grad / n_events + self.l2 * beta
            return float(nll), grad

        result = minimize(objective, np.zeros(X.shape[1], dtype=np.float64), method="L-BFGS-B",
                          jac=True, options={"maxiter": self.max_iter, "ftol": 1e-8, "gtol": 1e-6})
        beta = np.asarray(result.x, dtype=np.float64)
        # Near separation can exhaust the L-BFGS-B iteration budget
        # while leaving a finite L2-regularized solution.
        # Accept finite parameters and reject divergent optimization results.
        if not (np.all(np.isfinite(beta)) and np.isfinite(result.fun)):
            raise RuntimeError(f"CoxPH optimization diverged: {result.message}")
        self.coef_ = beta

        eta = X @ self.coef_
        increments = []
        for t in event_times:
            d = int(((time == t) & (event > 0.5)).sum())
            increments.append(d / np.exp(eta[time >= t]).sum())
        self.event_times_ = event_times.astype(np.float64)
        self.cum_baseline_hazard_ = np.cumsum(np.asarray(increments, dtype=np.float64))
        return self

    def _cum_hazard_at(self, times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.event_times_, times, side="right") - 1
        out = np.zeros_like(times, dtype=np.float64)
        valid = idx >= 0
        out[valid] = self.cum_baseline_hazard_[idx[valid]]
        return out

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.coef_ is None:
            raise RuntimeError("CoxPHBaseline must be fit before prediction")
        X = self._features(batch).astype(np.float64, copy=False)
        X = (X - self._mean) / self._std  # Reuse the training fold's feature standardization.
        relative_risk = np.exp(np.clip(X @ self.coef_, -30.0, 30.0))
        h0_h = float(self._cum_hazard_at(np.array([self.max_cycle_reference]))[0])
        risk = np.clip(1.0 - np.exp(-h0_h * relative_risk), 1e-6, 1.0 - 1e-6)

        # Median survival time on the learned Breslow grid. If S(t)>0.5 throughout, onset is
        # right-censored at the horizon and not extrapolated beyond the observed support.
        target_hazard = np.log(2.0) / relative_risk
        idx = np.array([np.searchsorted(self.cum_baseline_hazard_, h, side="left")
                        for h in target_hazard], dtype=int)
        onset = np.full(len(X), self.max_cycle_reference, dtype=np.float64)
        hit = idx < len(self.event_times_)
        onset[hit] = np.minimum(self.event_times_[idx[hit]], self.max_cycle_reference)
        norm = np.log1p(onset) / np.log1p(self.max_cycle_reference)

        query = batch.get("cycles", torch.full((len(X), 1), self.max_cycle_reference))
        cycles = query.detach().cpu().numpy().astype(float)
        cdf = 1 - np.exp(-relative_risk[:, None] * self._cum_hazard_at(cycles))
        quantiles = {}
        for alpha, key in ((.05, "nliq_q05"), (.95, "nliq_q95"), (.25, "event_q25"), (.75, "event_q75")):
            target = -np.log1p(-alpha) / relative_risk
            j = np.searchsorted(self.cum_baseline_hazard_, target, side="left")
            times = np.full(len(X), np.inf)
            ok = j < len(self.event_times_)
            times[ok] = self.event_times_[j[ok]]
            times[times > self.max_cycle_reference] = np.inf
            quantiles[key] = torch.from_numpy(times.astype(np.float32))

        p = torch.from_numpy(risk.astype(np.float32))
        return {
            "risk_prob": p,
            "risk_logit": torch.logit(p),
            "nliq_pred": torch.from_numpy(norm.astype(np.float32)),
            "supports_censored_nliq": torch.ones(len(onset), dtype=torch.float32),
            "event_cdf": torch.from_numpy(cdf.astype(np.float32)),
            "event_survival": torch.from_numpy((1 - cdf).astype(np.float32)),
            "event_tail_mass": torch.from_numpy((1 - cdf[:, -1]).astype(np.float32)),
            **quantiles,
        }
