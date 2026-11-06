"""Small controls for the final study: prefix GP and a horizon-mixture GBM.

GP kernels are fixed in advance and conditioning uses only the specimen's available prefix.
The GBM combines p(event by H) with a truncated log-normal event-time model. Short censored
records are excluded from the binary negatives, and the control does not maximize a full
censored likelihood.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF

from dpi_evt.evaluation.event_distribution import hitting_time_distribution


class PrefixGP:
    def __init__(self, seed=42, samples=128):
        self.seed, self.samples = int(seed), int(samples)

    def fit(self, train, val=None):
        return self

    def to(self, device):
        return self

    def eval(self):
        return self

    def forward_batch(self, batch):
        cycles = batch["cycles"].cpu().numpy()
        pm = batch["prefix_mask"].cpu().numpy() > .5
        po = batch["prefix_obs"].cpu().numpy()
        mus, variances, paths = [], [], []
        for c, mask, y in zip(cycles, pm, po):
            # Linear plus squared-exponential covariance in log cycle count, fixed in advance.
            kernel = ConstantKernel(.04, "fixed") * DotProduct(1., "fixed")
            kernel += ConstantKernel(.04, "fixed") * RBF(1., "fixed")
            gp = GaussianProcessRegressor(kernel=kernel, alpha=.01 ** 2, optimizer=None,
                                           normalize_y=False, random_state=self.seed)
            x = np.log1p(c.astype(np.float64))[:, None]
            if mask.any():
                gp.fit(x[mask], y[mask].astype(np.float64))
            mean, cov = gp.predict(x, return_cov=True)
            # Joint draws, with only roundoff-sized negative eigenvalues removed. Float32
            # kernel subtraction can otherwise produce spurious non-PSD covariance warnings.
            eig, vectors = np.linalg.eigh((cov + cov.T) / 2)
            if eig.min() < -1e-7 * max(float(eig.max()), 1.):
                raise ValueError("GP posterior covariance is not positive semidefinite")
            factor = vectors * np.sqrt(eig.clip(0))[None, :]
            normal = np.random.default_rng(self.seed).normal(size=(self.samples, len(x)))
            samples = mean[None, :] + normal @ factor.T
            mus.append(mean); variances.append(np.diag(cov)); paths.append(samples)
        mean = np.asarray(mus, np.float32)
        var = np.asarray(variances, np.float32).clip(1e-10)
        samples = torch.as_tensor(np.stack(paths, axis=1), dtype=torch.float32)
        projected = torch.cummax(samples.clamp(0, 1), dim=-1).values
        dist = hitting_time_distribution(projected, batch["cycles"].cpu())
        med = dist["event_time_median"]
        H = batch["cycles"][:, -1].cpu()
        return {"traj_mean": torch.from_numpy(mean), "traj_logvar": torch.log(torch.from_numpy(var)),
                "traj_q05": torch.from_numpy(mean - 1.644854 * np.sqrt(var)),
                "traj_q95": torch.from_numpy(mean + 1.644854 * np.sqrt(var)),
                "traj_projected_mean": projected.mean(0),
                "nliq": torch.minimum(med, H),
                "risk_prob": 1 - dist["event_tail_mass"],
                "supports_censored_nliq": torch.ones(len(mean)), **dist}


class GBMHorizonMixture:
    def __init__(self, seed=42, iterations=500, depth=6, learning_rate=.05):
        self.seed, self.iterations, self.depth, self.learning_rate = seed, iterations, depth, learning_rate

    @staticmethod
    def features(batch):
        return torch.cat([batch["static"], batch["prefix_summary"]], -1).cpu().numpy()

    def fit(self, train, val=None):
        from catboost import CatBoostClassifier, CatBoostRegressor
        kw = dict(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                  random_seed=self.seed, verbose=False, thread_count=1, allow_writing_files=False)
        X = self.features(train)
        label = train["label"].cpu().numpy() > .5
        known = train["risk_label_observed"].cpu().numpy() > .5
        event = label & (train["nliq_censor_valid"].cpu().numpy() > .5)
        if len(np.unique(label[known])) != 2 or event.sum() < 2:
            raise ValueError("GBM mixture needs known outcomes of both classes and >=2 events")
        self.classifier = CatBoostClassifier(**kw).fit(X[known], label[known].astype(int))
        time = train["n_liq_true"].cpu().numpy()
        self.regressor = CatBoostRegressor(**kw).fit(X[event], np.log1p(time[event]))
        if val is None:
            raise ValueError("GBM uncertainty scale requires a held-out validation split")
        ve = (val["label"].cpu().numpy() > .5) & (val["nliq_censor_valid"].cpu().numpy() > .5)
        if ve.sum() < 2:
            raise ValueError("GBM uncertainty calibration needs >=2 validation events")
        err = np.log1p(val["n_liq_true"].cpu().numpy()[ve]) - self.regressor.predict(self.features(val)[ve])
        self.scale = max(float(np.sqrt(np.mean(err ** 2))), .05)
        return self

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward_batch(self, batch):
        X = self.features(batch)
        prob = self.classifier.predict_proba(X)[:, 1]
        mu = self.regressor.predict(X)
        c = batch["cycles"].cpu().numpy()
        pm = batch["prefix_mask"].cpu().numpy() > .5
        n0 = np.where(pm, c, 0).max(1)
        lo = ndtr((np.log1p(n0) - mu) / self.scale)
        hi = ndtr((np.log1p(c[:, -1]) - mu) / self.scale)
        G = ((ndtr((np.log1p(c) - mu[:, None]) / self.scale) - lo[:, None]) /
             np.maximum(hi - lo, 1e-10)[:, None]).clip(0, 1)
        F = prob[:, None] * G
        quantiles = {}
        for alpha, key in ((.05, "nliq_q05"), (.95, "nliq_q95"), (.25, "event_q25"), (.75, "event_q75"), (.5, "event_time_median")):
            target = lo + np.minimum(alpha / prob.clip(1e-12), 1.) * (hi - lo)
            value = np.expm1(mu + self.scale * ndtri(target.clip(1e-12, 1 - 1e-12)))
            quantiles[key] = np.where(alpha <= prob, np.minimum(value, c[:, -1]), np.inf)
        # Use the analytical mixture median, avoiding a query-grid rounding penalty for GBM.
        n = np.minimum(quantiles["event_time_median"], c[:, -1])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in {
            "risk_prob": prob, "nliq": n, "event_cdf": F, "event_survival": 1 - F,
            "event_tail_mass": 1 - prob, "supports_censored_nliq": np.ones(len(X)),
            **quantiles,
        }.items()}
