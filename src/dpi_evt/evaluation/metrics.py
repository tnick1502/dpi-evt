"""Forecast metrics, grouped summaries and evaluation helpers.

Collect model outputs and calculate pressure errors, event discrimination,
probability scores, interval coverage and width for the study protocol.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import erf
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dpi_evt.config import ExperimentConfig, set_global_seed
from dpi_evt.data.meta import localize_meta_frame
from dpi_evt.data.splits import iterate_minibatches

__all__ = [
    "collect_outputs", "compute_metrics", "fit_interval_scale", "resolve_nliq_prediction",
    "expected_calibration_error", "safe_binary_metrics",
]


def collect_outputs(
    model: nn.Module,
    split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Evaluate a partition and concatenate tensor outputs into numpy arrays.

    :param model: fitted model with forward_batch
    :param split: partition from prepare_benchmark_dataset
    :param config: experiment settings, including batch size
    :param device: inference device
    :return: output arrays
    """
    # Seed Monte Carlo prediction and preserve RNG state
    # so evaluation does not change the training random stream.
    # Use predictive(...) with several theta samples when supported;
    # otherwise use forward_batch.
    import random as _random
    _torch_state = torch.get_rng_state()
    _cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _mps_state = (torch.mps.get_rng_state() if hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available() else None)
    _np_state = np.random.get_state()
    _py_state = _random.getstate()
    set_global_seed(config.seed)
    model.eval()
    mc = max(int(getattr(config, "mc_samples_eval", 1) or 1), int(getattr(model, "mc_predict_samples", 0) or 0))
    use_mc = mc > 1 and hasattr(model, "predictive")
    collected: Dict[str, List[torch.Tensor]] = {}
    try:
        with torch.no_grad():
            for batch in iterate_minibatches(split, config.batch_size, device, shuffle=False):
                outputs = model.predictive(batch, mc) if use_mc else model.forward_batch(batch)
                for key, value in outputs.items():
                    if torch.is_tensor(value):
                        collected.setdefault(key, []).append(value.detach().cpu())
    finally:
        torch.set_rng_state(_torch_state)
        if _cuda_state is not None:
            torch.cuda.set_rng_state_all(_cuda_state)
        if _mps_state is not None:
            torch.mps.set_rng_state(_mps_state)
        np.random.set_state(_np_state)
        _random.setstate(_py_state)
    return {key: torch.cat(value, dim=0).numpy() for key, value in collected.items()}


def fit_interval_scale(model, val_split: Dict[str, object], config: ExperimentConfig,
                       device: torch.device, level: float = 0.90) -> float:
    """Fit a standard-deviation multiplier s to validation coverage.

    Grid search minimizes the difference between level and coverage of
    ``y_hat +/- z*s*sigma``. Store ln(s) in calib_log_scale for forward_batch.
    This fitted scale has no finite-sample coverage guarantee.

    :param model: model with calib_log_scale and a trajectory head
    :param val_split: validation partition
    :param config: batch settings
    :param device: inference device
    :param level: target coverage, default 0.90
    :return: fitted scale, or 1.0 for an unsupported model
    """
    if not hasattr(model, "calib_log_scale"):
        return 1.0
    with torch.no_grad():
        model.calib_log_scale.zero_()
    out = collect_outputs(model, val_split, config, device)
    if "traj_logvar" not in out:
        return 1.0
    pred = out["traj_mean"]
    total_var = np.exp(out["traj_logvar"])
    epistemic_var = np.asarray(out.get("traj_epistemic_var", np.zeros_like(total_var)))
    aleatoric_var = np.maximum(total_var - epistemic_var, 0.0)
    true = val_split["r_obs"].cpu().numpy(); mask = val_split["mask"].cpu().numpy()
    if "continuation_mask" in val_split:
        mask = val_split["continuation_mask"].cpu().numpy()
    elif "prefix_mask" in val_split:
        mask = mask * (1.0 - np.minimum(val_split["prefix_mask"].cpu().numpy(), 1.0))
    z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}.get(level, 1.6449)
    denom = float(mask.sum())
    if denom <= 0:
        return 1.0

    def coverage(s):
        # calib_log_scale multiplies the emission variance only. The latent-path variance
        # remains unchanged in predictive(), and the fit uses the same variance decomposition.
        std = np.sqrt(epistemic_var + s * s * aleatoric_var)
        lo = pred - z * std; hi = pred + z * std
        return float(np.sum(((true >= lo) & (true <= hi)) * mask) / denom)

    grid = np.linspace(0.3, 4.0, 75)
    best = min(grid, key=lambda s: abs(coverage(s) - level))
    # Optional configured width multiplier, default 1.0.
    # Coverage on new sites is assessed separately.
    infl = float(getattr(config, "calibration_shift_inflation", 1.0) or 1.0)
    best = best * max(infl, 1e-3)
    with torch.no_grad():
        model.calib_log_scale.fill_(float(np.log(best)))
    model._variance_scale_validation = {"target_coverage": level, "coverage": coverage(best),
        "emission_sd_scale": float(best), "latent_variance_scaled": False}
    return float(best)


def resolve_nliq_prediction(outputs: Dict[str, np.ndarray], max_cycle_reference: float) -> np.ndarray:
    """Resolve an event-time prediction from model outputs.

    Use nliq directly, or invert log-normalized nliq_pred/nliq_norm with expm1 and
    the reference horizon.

    :param outputs: model output dictionary
    :param max_cycle_reference: horizon for inverse log normalization
    :return: predicted N_liq array
    :raises KeyError: if no supported field is present
    """
    if "nliq" in outputs:
        return outputs["nliq"]
    if "nliq_pred" in outputs:
        return np.expm1(outputs["nliq_pred"] * math.log1p(max_cycle_reference))
    if "nliq_norm" in outputs:
        return np.expm1(outputs["nliq_norm"] * math.log1p(max_cycle_reference))
    raise KeyError("outputs contain no N_liq prediction")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins.

    ECE = sum_b (|B_b|/N) * |acc(B_b)-conf(B_b)|. Compare mean observed frequency
    and mean predicted probability within each bin on [0, 1].

    :param y_true: binary outcomes
    :param y_prob: predicted probabilities
    :param n_bins: probability-bin count
    :return: ECE as a float
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(y_true))
    return float(ece)


def safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """Calculate binary metrics, allowing one-class subsets.

    Return NaN for AUROC/AUPRC when only one class is present; always calculate Brier.

    :param y_true: binary outcomes
    :param y_prob: predicted probabilities
    :return: (AUROC, AUPRC, Brier)
    """
    auroc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if np.unique(y_true).size > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)
    return float(auroc), float(auprc), float(brier)


def _interpolated_crossing_numpy(
    traj: np.ndarray,
    cycles: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """First crossing of the mean curve using the same linear interpolation as the models."""
    r = np.maximum.accumulate(np.asarray(traj, dtype=float), axis=1)
    cyc = np.asarray(cycles, dtype=float)
    above = r >= float(threshold)
    has = above.any(axis=1)
    idx = above.argmax(axis=1)
    out = cyc[:, -1].copy()
    for i in np.flatnonzero(has):
        j = int(idx[i])
        if j == 0:
            out[i] = cyc[i, 0]
            continue
        dr = r[i, j] - r[i, j - 1]
        frac = np.clip((float(threshold) - r[i, j - 1]) / dr, 0.0, 1.0) if dr > 0 else 0.0
        out[i] = cyc[i, j - 1] + frac * (cyc[i, j] - cyc[i, j - 1])
    return out, has


def _km_censoring_survival(times: np.ndarray, event: np.ndarray):
    """Estimate censoring survival G(t)=P(C>t) by Kaplan-Meier.

    Treat censoring as the event for this estimator.

    :return: step arrays (grid, G); use the last node <=t
    """
    order = np.argsort(times, kind="mergesort")
    t = np.asarray(times, float)[order]
    cens = (np.asarray(event, float)[order] < 0.5).astype(float)  # Censoring indicator.
    uniq = np.unique(t)
    G = np.empty(uniq.size, dtype=float)
    surv = 1.0
    for k, ut in enumerate(uniq):
        at_risk = float(np.sum(t >= ut))
        d = float(np.sum((t == ut) & (cens > 0.5)))
        if at_risk > 0.0:
            surv *= (1.0 - d / at_risk)
        G[k] = surv
    return uniq, G


def _eval_step(grid: np.ndarray, values: np.ndarray, query, floor: float = 1e-3):
    """Right-continuous step lookup at the greatest grid node <=query, bounded below by floor."""
    idx = np.searchsorted(grid, query, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return np.maximum(values[idx], floor)


def survival_discrimination_metrics(pred_time, true_time, event, mask, horizon):
    """Calculate censored event-time discrimination and deterministic-step IBS.

    Calculate Harrell C, Uno IPCW C and Graf IPCW integrated Brier score. Earlier
    predicted times imply higher risk. IBS here uses the same deterministic step
    S_i(t)=1[t<pred_time_i] for all models, rather than a native predictive CDF.
    """
    nan3 = {"N_liq_Cindex": float("nan"), "N_liq_Cindex_ipcw": float("nan"), "N_liq_IBS": float("nan")}
    m = np.asarray(mask, bool)
    if m.sum() < 4:
        return dict(nan3)
    t = np.asarray(true_time, float)[m]
    e = (np.asarray(event, float)[m] > 0.5).astype(float)
    p = np.asarray(pred_time, float)[m]
    if e.sum() < 2 or not np.isfinite(p).all():
        return dict(nan3)
    risk = -p  # Earlier predicted onset implies higher risk.
    tau = float(min(horizon, np.quantile(t, 0.9))) if horizon else float(np.quantile(t, 0.9))
    grid, G = _km_censoring_survival(t, e)

    num_h = den_h = num_u = den_u = 0.0
    for i in np.where(e > 0.5)[0]:
        comparable = t > t[i]  # Record j remains observed after the event in i.
        if not comparable.any():
            continue
        conc = ((risk[i] > risk[comparable]).astype(float)
                + 0.5 * (risk[i] == risk[comparable]).astype(float))
        num_h += float(conc.sum()); den_h += float(comparable.sum())
        if t[i] <= tau:
            w = 1.0 / (float(_eval_step(grid, G, t[i])) ** 2)
            num_u += w * float(conc.sum()); den_u += w * float(comparable.sum())
    c_harrell = num_h / den_h if den_h > 0 else float("nan")
    c_uno = num_u / den_u if den_u > 0 else float("nan")

    eval_t = np.linspace(0.0, tau, 25)[1:]
    bs = []
    for tt in eval_t:
        S = (p > tt).astype(float)  # Predicted survival at tt.
        Gt = float(_eval_step(grid, G, tt))
        contrib = np.zeros_like(t)
        c1 = (t <= tt) & (e > 0.5)          # Event before tt: observed survival is zero.
        contrib[c1] = (0.0 - S[c1]) ** 2 / _eval_step(grid, G, t[c1])
        c2 = t > tt                         # Still at risk: observed survival is one.
        contrib[c2] = (1.0 - S[c2]) ** 2 / Gt
        bs.append(float(contrib.mean()))    # Censoring before tt contributes zero.
    # A trapezoid sum avoids numpy-version-specific integration functions.
    bs_arr = np.asarray(bs, dtype=float)
    integral = float(np.sum(np.diff(eval_t) * 0.5 * (bs_arr[:-1] + bs_arr[1:])))
    ibs = integral / tau if tau > 0 else float("nan")
    return {"N_liq_Cindex": float(c_harrell), "N_liq_Cindex_ipcw": float(c_uno), "N_liq_IBS": ibs}


def compute_metrics(
    model_name: str,
    outputs: Dict[str, np.ndarray],
    split: Dict[str, object],
    config: ExperimentConfig,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Calculate summary metrics and a specimen-level evaluation table.

    Include event-time errors, binary probability metrics, pressure errors and
    90-percent interval diagnostics when the corresponding outputs are available.

    :param model_name: display name
    :param outputs: predictions from collect_outputs
    :param split: targets, metadata and observation masks
    :param config: experiment settings and reference horizon
    :return: summary dictionary and specimen-level DataFrame
    """
    meta_df = split["meta"].copy().reset_index(drop=True)
    y_true = split["label"].cpu().numpy()
    y_prob = outputs["risk_prob"]
    nliq_pred = resolve_nliq_prediction(outputs, config.max_cycle_reference)
    nliq_true = split["n_liq_true"].cpu().numpy()
    supports_censored_nliq = not (
        "supports_censored_nliq" in outputs
        and np.asarray(outputs["supports_censored_nliq"]).size
        and float(np.nanmax(outputs["supports_censored_nliq"])) < 0.5
    )

    # Binary by-H outcomes and event times have distinct masks. A short non-event
    # record has unknown by-H status but supplies a lower bound T>C.
    risk_obs_key = "risk_label_observed" if "risk_label_observed" in split else "n_liq_observed"
    nliq_obs_key = "nliq_censor_valid" if "nliq_censor_valid" in split else "n_liq_observed"
    risk_obs = (split[risk_obs_key].cpu().numpy() > 0.5 if risk_obs_key in split
                else np.ones_like(nliq_true, dtype=bool))
    nliq_obs = (split[nliq_obs_key].cpu().numpy() > 0.5 if nliq_obs_key in split
                else np.ones_like(nliq_true, dtype=bool))

    sample_df = localize_meta_frame(meta_df.copy()).reset_index(drop=True)
    # Keep site_id as the cluster unit for CV statistics
    # and coverage diagnostics; boreholes at one site remain one cluster.
    if "site_id" in meta_df.columns and "site_id" not in sample_df.columns:
        sample_df["site_id"] = meta_df["site_id"].to_numpy()
    sample_df["risk_prob_pred"] = y_prob
    sample_df["liq_label"] = y_true
    sample_df["nliq_pred"] = nliq_pred
    # Metadata retains raw event/censoring times, including times beyond H.
    # Keep the raw time distinct from the horizon-restricted scoring target
    # so a later observation is not mistaken for an error relative to H=3000.
    if "N_liq_true" in sample_df.columns:
        sample_df["N_liq_raw"] = sample_df["N_liq_true"].to_numpy()
    elif "n_liq_raw" in split:
        sample_df["N_liq_raw"] = split["n_liq_raw"].cpu().numpy()
    sample_df["N_liq_true"] = nliq_true

    nliq_err = nliq_pred - nliq_true
    log_pred = np.log1p(np.maximum(nliq_pred, 0.0))
    log_true = np.log1p(np.maximum(nliq_true, 0.0))
    log_err = log_pred - log_true
    liq = y_true > 0.5
    # Exact events use absolute error; censored records use one-sided error.
    nliq_cens_err = np.where(liq, np.abs(nliq_err), np.maximum(nliq_true - nliq_pred, 0.0))
    log_cens_err = np.where(liq, np.abs(log_err), np.maximum(log_true - log_pred, 0.0))
    # Set excluded specimen errors to NaN so groupby means
    # use the same observation mask as the aggregate.
    aggregate_nliq_mask = nliq_obs if supports_censored_nliq else np.zeros_like(nliq_obs)
    sample_df["nliq_abs_err"] = np.where(aggregate_nliq_mask, nliq_cens_err, np.nan)
    sample_df["nliq_log_err"] = np.where(aggregate_nliq_mask, log_cens_err, np.nan)
    sample_df["n_liq_observed"] = nliq_obs.astype(float) # event-time alias used by the report tables
    sample_df["nliq_censor_valid"] = nliq_obs.astype(float)
    sample_df["risk_label_observed"] = risk_obs.astype(float)

    if aggregate_nliq_mask.any():
        nliq_mae = float(np.mean(nliq_cens_err[aggregate_nliq_mask]))
        nliq_rmse = float(np.sqrt(np.mean(nliq_cens_err[aggregate_nliq_mask] ** 2)))
        nliq_logmae = float(np.mean(log_cens_err[aggregate_nliq_mask]))
        nliq_logrmse = float(np.sqrt(np.mean(log_cens_err[aggregate_nliq_mask] ** 2)))
    else:
        nliq_mae = nliq_rmse = nliq_logmae = nliq_logrmse = float("nan")

    metrics: Dict[str, object] = {
        "model": model_name,
        "N_liq_MAE": nliq_mae,
        "N_liq_RMSE": nliq_rmse,
        "N_liq_logMAE": nliq_logmae,
        "N_liq_logRMSE": nliq_logrmse,
        "N_liq_n_observed": int(aggregate_nliq_mask.sum()),
        "Supports_Censored_Nliq": supports_censored_nliq,
    }

    # Censored-time diagnostics shared across models, including Cox PH:
    # Harrell/IPCW concordance and deterministic-step Graf IBS.
    metrics.update(survival_discrimination_metrics(
        nliq_pred, nliq_true, liq.astype(float), aggregate_nliq_mask,
        float(config.max_cycle_reference)))

    # N_liq errors on observed events with exact times.
    if int(liq.sum()) > 0:
        metrics["N_liq_logMAE_liq"] = float(np.mean(np.abs(log_err[liq])))
        metrics["N_liq_MAE_liq"] = float(np.mean(np.abs(nliq_err[liq])))
        # Event timing and lead time.
        # Fraction of observed events predicted no later than their true onset,
        # and signed timing bias in cycles; negative values indicate earlier predictions.
        pred_on = nliq_pred[liq]; true_on = nliq_true[liq]; timing_bias = pred_on - true_on
        metrics["Onset_EarlyWarning_Rate"] = float(np.mean(pred_on <= true_on + 1e-6))
        metrics["Onset_Timing_Bias_cyc"] = float(np.median(timing_bias))
        metrics["Onset_Timing_MAE_cyc"] = float(np.median(np.abs(timing_bias)))
        sb = np.full(len(nliq_true), np.nan); sb[liq] = timing_bias
        sample_df["onset_timing_bias_cyc"] = sb
        if "nliq_q05" in outputs and "nliq_q95" in outputs:
            q05 = np.asarray(outputs["nliq_q05"]); q95 = np.asarray(outputs["nliq_q95"])
            covered = (nliq_true >= q05) & (nliq_true <= q95)
            metrics["N_liq_Coverage_90_liq"] = float(np.mean(covered[liq]))
            widths = np.full_like(q95, np.inf, dtype=float)
            finite = np.isfinite(q05) & np.isfinite(q95)
            np.subtract(q95, q05, out=widths, where=finite)
            metrics["N_liq_Interval_Width_90_liq"] = float(np.mean(widths[liq]))
        else:
            metrics["N_liq_Coverage_90_liq"] = float("nan")
            metrics["N_liq_Interval_Width_90_liq"] = float("nan")
    else:
        for _k in ("N_liq_logMAE_liq", "N_liq_MAE_liq", "Onset_EarlyWarning_Rate",
                   "Onset_Timing_Bias_cyc", "Onset_Timing_MAE_cyc",
                   "N_liq_Coverage_90_liq", "N_liq_Interval_Width_90_liq"):
            metrics[_k] = float("nan")
        sample_df["onset_timing_bias_cyc"] = np.nan

    # By-H outcomes are known for events and non-event records observed through H.
    if risk_obs.any():
        yt_r, yp_r = y_true[risk_obs], y_prob[risk_obs]
        auroc, auprc, brier = safe_binary_metrics(yt_r, yp_r)
        ece = expected_calibration_error(yt_r, yp_r)
    else:
        auroc = auprc = brier = ece = float("nan")
    metrics["AUROC"] = auroc
    metrics["AUPRC"] = auprc
    metrics["Brier"] = brier
    metrics["ECE"] = ece
    metrics["N_risk_observed"] = int(risk_obs.sum())

    if "traj_mean" in outputs:
        pred = outputs["traj_mean"]
        # The trajectory target is measured pore pressure,
        # evaluated only where measurements are valid.
        true = split["r_obs"].cpu().numpy()
        mask = split["mask"].cpu().numpy()
        mse = float(np.sum(((pred - true) ** 2) * mask) / np.maximum(mask.sum(), 1.0))
        mae = float(np.sum(np.abs(pred - true) * mask) / np.maximum(mask.sum(), 1.0))
        rmse = float(np.sqrt(mse))
        metrics["Traj_MSE"] = mse
        metrics["Traj_MAE"] = mae
        metrics["Traj_RMSE"] = rmse

        sample_df["traj_pred_max"] = np.max(pred, axis=1)
        if "cycles" in split:
            curve_nliq, curve_crosses = _interpolated_crossing_numpy(
                pred, split["cycles"].cpu().numpy(), config.liq_threshold,
            )
            curve_gap = np.abs(nliq_pred - curve_nliq)
            risk_curve_coherent = (y_prob >= 0.5) == curve_crosses
            metrics["N_liq_Curve_Coherence_MAE"] = float(np.mean(curve_gap))
            metrics["Risk_Curve_Coherence_Rate"] = float(np.mean(risk_curve_coherent))
            sample_df["curve_crosses_threshold"] = curve_crosses.astype(float)
            sample_df["nliq_from_curve"] = curve_nliq
            sample_df["nliq_curve_gap"] = curve_gap
            sample_df["risk_curve_coherent"] = risk_curve_coherent.astype(float)
        else:
            metrics["N_liq_Curve_Coherence_MAE"] = float("nan")
            metrics["Risk_Curve_Coherence_Rate"] = float("nan")
            for key in ("curve_crosses_threshold", "nliq_from_curve", "nliq_curve_gap",
                        "risk_curve_coherent"):
                sample_df[key] = np.nan

        sample_mask_count = np.maximum(mask.sum(axis=1), 1.0)
        sample_df["traj_rmse"] = np.sqrt(np.sum(((pred - true) ** 2) * mask, axis=1) / sample_mask_count)

        # Models observe the early PPR prefix as context. The error above covers reconstruction
        # and continuation; the post-prefix forecast error is reported separately below.
        if "continuation_mask" in split:
            # The controlled prefix experiment supplies the same N>20 window for every N0.
            # Deriving the window from prefix_mask would compare different intervals across
            # prefix lengths.
            continuation_mask = split["continuation_mask"].cpu().numpy() * mask
        elif "prefix_mask" in split:
            prefix_mask = split["prefix_mask"].cpu().numpy()
            continuation_mask = mask * (1.0 - np.minimum(prefix_mask, 1.0))
        else:
            continuation_mask = mask
        cont_total = float(continuation_mask.sum())
        if cont_total > 0:
            cont_mse = float(np.sum(((pred - true) ** 2) * continuation_mask) / cont_total)
            cont_mae = float(np.sum(np.abs(pred - true) * continuation_mask) / cont_total)
            cont_rmse = float(np.sqrt(cont_mse))
        else:
            cont_mse = cont_mae = cont_rmse = float("nan")
        metrics["Traj_MSE_continuation"] = cont_mse
        metrics["Traj_MAE_continuation"] = cont_mae
        metrics["Traj_RMSE_continuation"] = cont_rmse
        sample_cont_count = continuation_mask.sum(axis=1)
        sample_cont_valid = sample_cont_count > 0
        sample_cont_sse = np.sum(((pred - true) ** 2) * continuation_mask, axis=1)
        sample_cont_rmse = np.full(len(true), np.nan, dtype=float)
        sample_cont_rmse[sample_cont_valid] = np.sqrt(
            sample_cont_sse[sample_cont_valid] / sample_cont_count[sample_cont_valid]
        )
        sample_df["traj_rmse_continuation"] = sample_cont_rmse
        sample_df["traj_sse_continuation"] = np.where(sample_cont_valid, sample_cont_sse, np.nan)
        sample_df["continuation_points"] = sample_cont_count
        sample_df["continuation_valid"] = sample_cont_valid.astype(float)
        # Late-window RMSE over the latter half of the valid trajectory,
        # beyond the observed prefix, describes longer-range continuation.
        # Report it separately from full-window trajectory error.
        idxg = np.arange(true.shape[1])[None, :]
        horizon = (mask * idxg).max(axis=1, keepdims=True) # Last valid index for each specimen.
        far_mask = continuation_mask * (idxg >= 0.5 * horizon)
        far_total = float(far_mask.sum())
        metrics["Traj_RMSE_late"] = (float(np.sqrt(np.sum(((pred - true) ** 2) * far_mask) / far_total))
                                     if far_total > 0 else float("nan"))
        sample_far_count = far_mask.sum(axis=1)
        sample_far = np.full(len(true), np.nan, dtype=float)
        _fv = sample_far_count > 0
        sample_far[_fv] = np.sqrt(
            np.sum(((pred - true) ** 2) * far_mask, axis=1)[_fv] / sample_far_count[_fv]
        )
        sample_df["traj_rmse_late"] = sample_far

        # Trajectory errors by event/completed/unfinished observation state.
        # States are evaluated separately from binary-risk and censoring masks.
        # Balanced error is a macro-average over
        # the states present in this subset;
        # worst-state error is the largest state-specific value.
        # Retain both alongside the pooled error.
        _se = ((pred - true) ** 2) * mask
        def _split_bool(name, fallback):
            return split[name].cpu().numpy() > 0.5 if name in split else fallback

        _states = {
            "liq": _split_bool("regime_liq", y_true > 0.5),
            "stab": _split_bool("regime_stable", (y_true < 0.5) & risk_obs),
            "nostab": _split_bool("regime_unfinished", (y_true < 0.5) & (~risk_obs)),
        }
        sample_df["regime"] = np.select(
            [_states["liq"], _states["stab"], _states["nostab"]],
            ["liquefied", "stabilized", "unfinished"], default="unknown"
        )
        _present = []
        for _nm, _m in _states.items():
            if int(_m.sum()) > 0:
                _v = float(np.sqrt(_se[_m].sum() / np.maximum(mask[_m].sum(), 1.0)))
            else:
                _v = float("nan")
            metrics[f"Traj_RMSE_{_nm}"] = _v
            if _v == _v:
                _present.append(_v)
        metrics["Traj_RMSE_balanced"] = float(np.mean(_present)) if _present else float("nan")
        metrics["Traj_RMSE_worst"] = float(np.max(_present)) if _present else float("nan")

        _se_cont = ((pred - true) ** 2) * continuation_mask
        _present_cont = []
        for _nm, _m in _states.items():
            if int(_m.sum()) > 0 and float(continuation_mask[_m].sum()) > 0:
                _v = float(np.sqrt(_se_cont[_m].sum() / np.maximum(continuation_mask[_m].sum(), 1.0)))
            else:
                _v = float("nan")
            metrics[f"Traj_RMSE_continuation_{_nm}"] = _v
            if _v == _v:
                _present_cont.append(_v)
        metrics["Traj_RMSE_continuation_balanced"] = (
            float(np.mean(_present_cont)) if _present_cont else float("nan")
        )
        metrics["Traj_RMSE_continuation_worst"] = (
            float(np.max(_present_cont)) if _present_cont else float("nan")
        )

        # Check nondecreasing PPR in [0, 1] across the entire forecast horizon,
        # including the tail after the observed record ends.
        # Do not apply the measurement mask to this construction check.
        # Epsilon accommodates floating-point rounding only.
        _pvr_tol = 1e-5
        diffs = pred[:, 1:] - pred[:, :-1]
        finite = np.isfinite(pred)
        finite_steps = finite[:, 1:] & finite[:, :-1]
        decreasing = np.any((diffs < -_pvr_tol) & finite_steps, axis=1)
        out_of_bounds = np.any(((pred > 1.0 + _pvr_tol) | (pred < -_pvr_tol)) & finite, axis=1)
        nonfinite = np.any(~finite, axis=1)
        violation = decreasing | out_of_bounds | nonfinite
        metrics["Physics_Violation_Rate"] = float(violation.mean())
        sample_df["physics_violation"] = violation.astype(float)
        sample_df["physics_decrease_violation"] = decreasing.astype(float)
        sample_df["physics_bounds_violation"] = out_of_bounds.astype(float)
        sample_df["physics_nonfinite_violation"] = nonfinite.astype(float)

        if "traj_logvar" in outputs:
            std = np.maximum(np.sqrt(np.exp(outputs["traj_logvar"])), 1e-6)
            # Compute coverage, NLL and CRPS on the continuation mask.
            # The observed prefix is conditioning information;
            # including it would change the coverage denominator and mix
            # reconstruction with forecasting. pm_cnt is the continuation count.
            pm = continuation_mask
            pm_total = float(pm.sum())
            pm_row = pm.sum(axis=1)
            pm_valid = pm_row > 0
            cov_gaps = []
            for level, z in [(80, 1.2816), (90, 1.6449), (95, 1.9600)]:
                lower = pred - z * std
                upper = pred + z * std
                cov = (float(np.sum(((true >= lower) & (true <= upper)) * pm) / pm_total)
                       if pm_total > 0 else float("nan"))
                width = (float(np.sum((upper - lower) * pm) / pm_total)
                         if pm_total > 0 else float("nan"))
                metrics[f"Coverage_{level}"] = cov
                metrics[f"Interval_Width_{level}"] = width
                if np.isfinite(cov):
                    cov_gaps.append(abs(cov - level / 100.0))
                if level == 90:
                    # Specimen continuation coverage at 90 percent for site-cluster intervals.
                    inb = ((true >= lower) & (true <= upper)) * pm
                    coverage90 = np.full(len(true), np.nan, dtype=float)
                    coverage_hits = inb.sum(axis=1)
                    coverage90[pm_valid] = coverage_hits[pm_valid] / pm_row[pm_valid]
                    sample_df["coverage90"] = coverage90
                    sample_df["coverage90_hits"] = np.where(pm_valid, coverage_hits, np.nan)
            # Mean absolute difference between empirical and nominal coverage across levels.
            metrics["Calibration_Error"] = float(np.mean(cov_gaps)) if cov_gaps else float("nan")
            # Split-conformal evaluation requires a separate calibration set
            # disjoint from the test set.
            # compute_metrics receives no such set and does not fit a test quantile.
            # A caller can calibrate on validation records and evaluate the resulting
            # intervals on test records without using test residuals for calibration.
            # Pointwise coverage pools time points;
            # it differs from simultaneous whole-trajectory coverage.
            # Gaussian predictive NLL on continuation points.
            nll = 0.5 * (np.log(2 * np.pi * std ** 2) + ((true - pred) ** 2) / (std ** 2))
            metrics["Traj_NLL"] = (float(np.sum(nll * pm) / pm_total) if pm_total > 0 else float("nan"))
            # Gaussian predictive CRPS on continuation points.
            z0 = (true - pred) / std
            phi = np.exp(-0.5 * z0 ** 2) / np.sqrt(2 * np.pi)
            Phi = 0.5 * (1.0 + erf(z0 / np.sqrt(2.0)))
            crps = std * (z0 * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
            metrics["Traj_CRPS"] = (float(np.sum(crps * pm) / pm_total) if pm_total > 0 else float("nan"))
            std90 = 1.6449 * std
            interval_width = np.full(len(true), np.nan, dtype=float)
            mean_std = np.full(len(true), np.nan, dtype=float)
            interval_width[pm_valid] = np.sum((2 * std90) * pm, axis=1)[pm_valid] / pm_row[pm_valid]
            mean_std[pm_valid] = np.sum(std * pm, axis=1)[pm_valid] / pm_row[pm_valid]
            sample_df["interval_width"] = interval_width
            sample_df["mean_pred_std_continuation"] = mean_std
        else:
            for level in (80, 90, 95):
                metrics[f"Coverage_{level}"] = float("nan")
                metrics[f"Interval_Width_{level}"] = float("nan")
            metrics["Calibration_Error"] = float("nan")
            metrics["Traj_NLL"] = float("nan")
            metrics["Traj_CRPS"] = float("nan")
            sample_df["interval_width"] = np.nan
    else:
        metrics["Traj_MSE"] = float("nan")
        metrics["Traj_MAE"] = float("nan")
        metrics["Traj_RMSE"] = float("nan")
        metrics["Traj_MSE_continuation"] = float("nan")
        metrics["Traj_MAE_continuation"] = float("nan")
        metrics["Traj_RMSE_continuation"] = float("nan")
        metrics["N_liq_Curve_Coherence_MAE"] = float("nan")
        metrics["Risk_Curve_Coherence_Rate"] = float("nan")
        metrics["Physics_Violation_Rate"] = float("nan")
        for level in (80, 90, 95):
            metrics[f"Coverage_{level}"] = float("nan")
            metrics[f"Interval_Width_{level}"] = float("nan")
        metrics["Calibration_Error"] = float("nan")
        metrics["Traj_NLL"] = float("nan")
        metrics["Traj_CRPS"] = float("nan")
        sample_df["traj_rmse"] = np.nan
        sample_df["traj_rmse_continuation"] = np.nan
        sample_df["interval_width"] = np.nan
        sample_df["physics_violation"] = np.nan
        sample_df["physics_decrease_violation"] = np.nan
        sample_df["physics_bounds_violation"] = np.nan
        sample_df["physics_nonfinite_violation"] = np.nan

    # Compare the model resistance coordinate with measured CRR targets
    # where available; this diagnostic does not identify material CRR.
    crr_rmse = float("nan")
    n_crr_test = 0
    n_crr_objects = 0
    if "crr" in outputs and "crr_obs" in split and "crr_obs_mask" in split:
        crr_pred = outputs["crr"]
        crr_true = split["crr_obs"].cpu().numpy()
        crr_m = split["crr_obs_mask"].cpu().numpy()
        tmask = split["mask"].cpu().numpy()
        per = np.sqrt(np.sum(((crr_pred - crr_true) ** 2) * tmask, axis=1) / np.maximum(tmask.sum(axis=1), 1.0))
        sel = crr_m > 0
        n_crr_test = int(sel.sum())
        # Count the independent sites contributing measured CRR targets
        # alongside the specimen count to describe the diagnostic's support.
        # Use object if site_id is unavailable.
        _crr_grp = "site_id" if "site_id" in sample_df.columns else ("object" if "object" in sample_df.columns else None)
        if _crr_grp is not None and n_crr_test > 0:
            n_crr_objects = int(pd.Series(sample_df[_crr_grp].to_numpy()[sel]).nunique())
        if sel.any():
            crr_rmse = float(per[sel].mean())
    metrics["CRR_RMSE"] = crr_rmse
    metrics["N_CRR_test"] = n_crr_test # Test specimens with measured CRR.
    metrics["N_CRR_objects"] = n_crr_objects # Objects/sites represented by those specimens.
    metrics["Produces_CRR"] = "crr" in outputs
    # Onset consistency diagnostic: compare predicted resistance
    # with applied CSR at the predicted event cycle.
    if "crr" in outputs and int(liq.sum()) > 0:
        cyc = split["cycles"].cpu().numpy(); csr_seq = split["csr"].cpu().numpy()
        crr_pred = np.asarray(outputs["crr"])
        crr_at = np.array([np.interp(nliq_pred[i], cyc[i], crr_pred[i]) for i in range(len(cyc))])
        csr_at = np.array([np.interp(nliq_pred[i], cyc[i], csr_seq[i]) for i in range(len(cyc))])
        metrics["CRR_Onset_Coherence_MAE"] = float(np.mean(np.abs(crr_at[liq] - csr_at[liq])))
    else:
        metrics["CRR_Onset_Coherence_MAE"] = float("nan")

    # Site-macro summaries give each site equal weight:
    # first average within a site, then across sites,
    # alongside the specimen-weighted summaries.
    # Calculate for continuation RMSE and log event-time error.
    if "site_id" in sample_df.columns:
        def _site_macro(col: str) -> float:
            if col not in sample_df.columns:
                return float("nan")
            sub = sample_df[["site_id", col]].dropna(subset=[col])
            if sub.empty:
                return float("nan")
            return float(sub.groupby("site_id")[col].mean().mean())
        metrics["Traj_RMSE_continuation_siteMacro"] = _site_macro("traj_rmse_continuation")
        if "nliq_log_err" in sample_df.columns:
            _tmp = sample_df[["site_id", "nliq_log_err"]].copy()
            _tmp["nliq_log_err"] = _tmp["nliq_log_err"].abs()
            _tmp = _tmp.dropna(subset=["nliq_log_err"])
            metrics["N_liq_logMAE_siteMacro"] = (
                float(_tmp.groupby("site_id")["nliq_log_err"].mean().mean()) if not _tmp.empty else float("nan"))
        else:
            metrics["N_liq_logMAE_siteMacro"] = float("nan")
        metrics["N_sites_test"] = int(sample_df["site_id"].nunique())

    return metrics, sample_df


