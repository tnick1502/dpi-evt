"""Paired comparisons and cluster intervals from out-of-fold predictions.

Match model errors on the same specimens, folds and objects. Cluster resampling
keeps repeated predictions of a specimen together.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

PAIR_METRICS = ["traj_rmse_continuation", "nliq_log_err"] # per-sample, lower=better
CLS_METRICS = ["AUROC", "AUPRC", "Brier", "ECE"]


# Classification metrics.
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    ps = p[order]; i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y) > 0.5
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    # Equal scores enter at one threshold, as in average_precision_score. Treating tied
    # records individually makes AP depend on row order (notably for binary references).
    end = np.r_[np.flatnonzero(np.diff(np.asarray(p)[order])), len(order) - 1]
    prec = tp[end] / np.maximum(tp[end] + fp[end], 1); rec = tp[end] / max(int(y.sum()), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - (np.asarray(y) > 0.5).astype(float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = (np.asarray(y) > 0.5).astype(float); e = 0.0; n = len(p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)


_CLS_FN = {"AUROC": auroc, "AUPRC": auprc, "Brier": brier, "ECE": ece}


# Statistical helpers.
def _erf(x):
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def wilcoxon_signed_rank(d: np.ndarray):
    """Two-sided signed-rank statistic, with exact sign permutation for small samples."""
    d = d[np.abs(d) > 0]; n = len(d)
    if n < 6:
        return float("nan"), float("nan"), float("nan"), n
    a = np.abs(d); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, float); ranks[order] = np.arange(1, n + 1)
    a_sorted = a[order]; i = 0; tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            t = j - i + 1; tie_term += t ** 3 - t
        i = j + 1
    w_plus = ranks[d > 0].sum()
    if n <= 20:
        observed = abs(float(np.sum(np.sign(d) * ranks)))
        ids = np.arange(1 << n, dtype=np.uint32)[:, None]
        signs = 2.0 * ((ids >> np.arange(n, dtype=np.uint32)) & 1).astype(float) - 1.0
        stats = np.abs(signs @ ranks)
        p = float((np.count_nonzero(stats >= observed - 1e-12) + 1) / (len(stats) + 1))
        return float(w_plus), float("nan"), min(p, 1.0), n
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if sigma == 0:
        return w_plus, float("nan"), float("nan"), n
    z = (w_plus - mu - 0.5 * np.sign(w_plus - mu)) / sigma
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(w_plus), float(z), float(min(max(p, 0.0), 1.0)), n


def holm_bonferroni(pvals) -> np.ndarray:
    m = len(pvals); order = np.argsort(pvals)
    adj = np.empty(m, float); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx]); adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(d: np.ndarray) -> float:
    d = d[np.abs(d) > 0]
    if len(d) == 0:
        return float("nan")
    r = pd.Series(np.abs(d)).rank().to_numpy()
    wp, wn = r[d > 0].sum(), r[d < 0].sum(); tot = wp + wn
    return float((wp - wn) / tot) if tot > 0 else float("nan")


def _bootstrap_median(d: np.ndarray, nboot: int, rng):
    if len(d) == 0:
        return float("nan"), float("nan")
    bs = np.array([np.median(rng.choice(d, size=len(d), replace=True)) for _ in range(min(nboot, 1000))])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


# Public API.
def paired_significance(df: pd.DataFrame, ref: str = "DPI-EVT",
                        metrics: Optional[List[str]] = None, nboot: int = 1000, seed: int = 0,
                        cluster: Optional[str] = "object") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank with Holm correction and effect size.

    cluster='object' first averages errors by site, then compares model pairs on those
    site means. This retains the site as the analysis unit for correlated specimens.
    cluster=None selects a diagnostic record-level test.
    """
    metrics = metrics or PAIR_METRICS
    models = list(df["model"].unique())
    if ref not in models:
        raise ValueError(f"reference model {ref} not found; available: {models}")
    others = [m for m in models if m != ref]
    # Prefer site_id over object for clustering, matching the CV split unit;
    # boreholes from one site form a single cluster.
    if cluster in ("object", "site_id") and "site_id" in df.columns:
        cluster = "site_id"
    use_cluster = bool(cluster) and cluster in df.columns
    # Use one aggregate per cluster, otherwise match (repeat, fold, sidx).
    idx_cols = [cluster] if use_cluster else [c for c in ("repeat", "fold", "sidx") if c in df.columns]
    rng = np.random.default_rng(seed); rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        if use_cluster:
            agg = df.groupby(["model", cluster])[metric].mean().reset_index()
            piv = agg.pivot_table(index=cluster, columns="model", values=metric)
        else:
            piv = df.pivot_table(index=idx_cols, columns="model", values=metric)
        if ref not in piv.columns:
            continue
        # Compare models only where the metric is available.
        present = [m for m in others if m in piv.columns]
        raw_p, recs = {}, {}
        for m in present:
            sub = piv[[ref, m]].dropna()
            d = (sub[m] - sub[ref]).to_numpy() # Positive difference: baseline error exceeds the reference error.
            if len(d) == 0:
                continue
            w, z, p, n = wilcoxon_signed_rank(d)
            raw_p[m] = p if p == p else 1.0
            lo, hi = _bootstrap_median(d, nboot, rng)
            recs[m] = dict(metric=metric, model=m, ref=ref, n=n, z=z, p_raw=p,
                           median_diff=float(np.median(d)), median_diff_lo=lo, median_diff_hi=hi,
                           rank_biserial=rank_biserial(d))
        if not recs:
            continue
        keys = list(recs.keys())
        adj = holm_bonferroni([raw_p[m] for m in keys])
        for m, pa in zip(keys, adj):
            recs[m]["p_holm"] = float(pa); recs[m]["significant_0.05"] = bool(pa < 0.05)
            rows.append(recs[m])
    return pd.DataFrame(rows)


def object_cluster_bootstrap(df: pd.DataFrame, ref: str = "DPI-EVT", nboot: int = 2000,
                             seed: int = 0, err_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Build pooled-OOF intervals by resampling sites.

    Calculate each metric within a repeat and average over repeats. Each selected
    site carries all its specimens and repeated predictions. Repeated predictions
    are not new independent clusters.

    :param df: specimen OOF table with object, liq_label, risk_prob_pred and error columns
    :param ref: reference model for paired differences
    :param nboot: bootstrap iteration count
    :param err_metrics: specimen errors to average, defaulting to continuation and log-time errors
    :return: model point estimates, 95-percent intervals and paired differences
    """
    err_metrics = err_metrics or [m for m in ("traj_rmse_continuation", "nliq_log_err", "coverage90")
                                  if m in df.columns]
    # Bootstrap site_id, the same unit used in grouped splitting;
    # multiple boreholes at one site are not independent clusters.
    ccol = "site_id" if "site_id" in df.columns else "object"
    if ccol not in df.columns:
        raise ValueError("cluster bootstrap requires a 'site_id' or 'object' column")
    models = list(df["model"].unique())
    rng = np.random.default_rng(seed)
    objects = np.array(sorted(df[ccol].unique()))
    by = {m: df[df.model == m] for m in models}

    def single_repeat_metrics(sub: pd.DataFrame) -> dict:
        # Classification metrics use only observed binary outcomes,
        # risk_label_observed>0.5, matching compute_metrics;
        # unfinished records are excluded from binary negatives.
        risk_mask_col = "risk_label_observed" if "risk_label_observed" in sub.columns else "n_liq_observed"
        if risk_mask_col in sub.columns:
            cs = sub[sub[risk_mask_col].to_numpy() > 0.5]
        else:
            cs = sub
        y = cs["liq_label"].to_numpy(); p = cs["risk_prob_pred"].to_numpy()
        if len(y) and len(np.unique(y)) > 1:
            out = {k: _CLS_FN[k](y, p) for k in CLS_METRICS}
        else:
            out = {k: float("nan") for k in CLS_METRICS}
        for em in err_metrics:
            if em == "traj_rmse_continuation" and {"traj_sse_continuation", "continuation_points"}.issubset(sub.columns):
                valid = sub["continuation_points"].to_numpy() > 0
                denom = float(sub.loc[valid, "continuation_points"].sum())
                out[em] = (float(np.sqrt(sub.loc[valid, "traj_sse_continuation"].sum() / denom))
                           if denom > 0 else float("nan"))
            elif em == "coverage90" and {"coverage90_hits", "continuation_points"}.issubset(sub.columns):
                valid = sub["continuation_points"].to_numpy() > 0
                denom = float(sub.loc[valid, "continuation_points"].sum())
                out[em] = (float(sub.loc[valid, "coverage90_hits"].sum() / denom)
                           if denom > 0 else float("nan"))
            elif em == "traj_interval_width" and {"traj_interval_width_sum", "continuation_points"}.issubset(sub.columns):
                valid = (sub["continuation_points"] > 0) & sub["traj_interval_width_sum"].notna()
                denom = float(sub.loc[valid, "continuation_points"].sum())
                out[em] = float(sub.loc[valid, "traj_interval_width_sum"].sum() / denom) if denom else float("nan")
            else:
                v = sub[em].to_numpy(); v = v[~np.isnan(v)]
                out[em] = float(np.mean(v)) if len(v) else float("nan")
        return out

    def metrics_on(sub):
        if "repeat" not in sub or sub["repeat"].nunique() < 2:
            return single_repeat_metrics(sub)
        values = pd.DataFrame([single_repeat_metrics(g) for _, g in sub.groupby("repeat")])
        return values.mean().to_dict()

    point = {m: metrics_on(by[m]) for m in models}
    keys = CLS_METRICS + err_metrics
    draws = {m: {k: np.empty(nboot) for k in keys} for m in models}
    # Pre-index rows by object for faster resampling.
    idx_by_obj = {m: {o: by[m].index[by[m][ccol].to_numpy() == o].to_numpy() for o in objects} for m in models}
    for b in range(nboot):
        sample_objs = rng.choice(objects, size=len(objects), replace=True)
        for m in models:
            rid = np.concatenate([idx_by_obj[m][o] for o in sample_objs]) if len(objects) else np.array([], int)
            sub = by[m].loc[rid]
            mm = metrics_on(sub)
            for k in keys:
                draws[m][k][b] = mm[k]
    rows = []
    for m in models:
        rec = {"model": m}
        for k in keys:
            arr = draws[m][k]
            if np.all(np.isnan(arr)): # Metric not applicable to this model.
                rec[k] = round(point[m][k], 4) if point[m][k] == point[m][k] else np.nan
                rec[f"{k}_lo"] = np.nan; rec[f"{k}_hi"] = np.nan
                if m != ref:
                    rec[f"{k}_vs_ref_sig"] = False
                continue
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            rec[k] = round(point[m][k], 4); rec[f"{k}_lo"] = round(float(lo), 4); rec[f"{k}_hi"] = round(float(hi), 4)
            if m != ref:
                diff = arr - draws[ref][k]
                if np.all(np.isnan(diff)):
                    rec[f"{k}_vs_ref_sig"] = False
                else:
                    dlo, dhi = np.nanpercentile(diff, [2.5, 97.5])
                    rec[f"{k}_vs_ref_sig"] = bool(dlo > 0 or dhi < 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_classification(df: pd.DataFrame, ref: str = "DPI-EVT",
                             nboot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap AUROC/AUPRC/Brier/ECE by fold, including paired differences from ref."""
    # Classification metrics use observed outcomes, as in compute_metrics.
    # All models share the mask; filtering therefore preserves
    # positional alignment for paired bootstrap resampling.
    risk_mask_col = "risk_label_observed" if "risk_label_observed" in df.columns else "n_liq_observed"
    if risk_mask_col in df.columns:
        df = df[df[risk_mask_col] > 0.5].copy()
    models = list(df["model"].unique())
    rng = np.random.default_rng(seed)
    sort_cols = [c for c in ("repeat", "fold", "sidx") if c in df.columns]
    by_model = {m: df[df.model == m].sort_values(sort_cols).reset_index(drop=True) for m in models}
    base = by_model[ref][["fold", "sidx"]]
    fold_idx = {f: np.where(base["fold"].to_numpy() == f)[0] for f in base["fold"].unique()}
    point, draws = {}, {}
    for m in models:
        d = by_model[m]; y = d["liq_label"].to_numpy(); p = d["risk_prob_pred"].to_numpy()
        point[m] = {k: _CLS_FN[k](y, p) for k in CLS_METRICS}
        draws[m] = {k: np.empty(nboot) for k in CLS_METRICS}
    for b in range(nboot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in fold_idx.values()])
        for m in models:
            d = by_model[m]; y = d["liq_label"].to_numpy()[idx]; p = d["risk_prob_pred"].to_numpy()[idx]
            for k in CLS_METRICS:
                draws[m][k][b] = _CLS_FN[k](y, p)
    rows = []
    for m in models:
        rec = {"model": m}
        for k in CLS_METRICS:
            lo, hi = np.nanpercentile(draws[m][k], [2.5, 97.5])
            rec[k] = round(point[m][k], 4); rec[f"{k}_lo"] = round(float(lo), 4); rec[f"{k}_hi"] = round(float(hi), 4)
            if m != ref:
                diff = draws[m][k] - draws[ref][k]
                dlo, dhi = np.nanpercentile(diff, [2.5, 97.5])
                rec[f"{k}_vs_ref_lo"] = round(float(dlo), 4); rec[f"{k}_vs_ref_hi"] = round(float(dhi), 4)
                rec[f"{k}_vs_ref_sig"] = bool(dlo > 0 or dhi < 0)
        rows.append(rec)
    return pd.DataFrame(rows)
