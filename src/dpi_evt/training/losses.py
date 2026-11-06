"""Masked training losses, predictive density scores and checkpoint-copy helpers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["masked_mean", "masked_mse", "masked_mae", "gaussian_nll", "gaussian_mixture_nll",
           "masked_bce_with_logits", "energy_crps", "clone_state_dict",
           "observed_aux_loss", "soft_auc_loss", "monotone_clip", "beta_nll", "censored_nliq_loss",
           "masked_censored_nliq_loss", "risk_observation_mask", "nliq_censor_mask",
           "interpolated_crossing", "crossing_margin_loss", "monotone_residual_scale",
           "normalized_free_increments", "trajectory_supervision_masks", "forecast_gaussian_nll"]


def risk_observation_mask(batch: Dict[str, torch.Tensor]):
    """Return the mask of observed binary by-horizon outcomes."""
    return batch["risk_label_observed"]


def trajectory_supervision_masks(batch: Dict[str, torch.Tensor]):
    """Return continuation and prefix masks for trajectory supervision.

    The primary term scores forecasting after the observed window; the auxiliary
    term reconstructs the observed prefix.

    :param batch: prepared input batch
    :return: (continuation_mask, prefix_mask), shaped like mask
    """
    return batch["continuation_mask"], batch["prefix_mask"]


def forecast_gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                          batch: Dict[str, torch.Tensor], prefix_weight: float = 0.15) -> torch.Tensor:
    """Combine post-prefix Gaussian NLL with a weak prefix-reconstruction term.

    The main term uses the forecasting window. The prefix term supports continuity
    at the observed/forecast boundary.
    """
    continuation, prefix = trajectory_supervision_masks(batch)
    loss = gaussian_nll(mean, logvar, target, continuation)
    if prefix_weight > 0.0:
        loss = loss + float(prefix_weight) * gaussian_nll(mean, logvar, target, prefix)
    return loss


def nliq_censor_mask(batch: Dict[str, torch.Tensor]):
    """Return the exact/censored event-time mask, distinct from binary-risk observability."""
    return batch["nliq_censor_valid"]


def soft_auc_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss approximating 1-AUROC.

    Penalize positive/negative pairs whose logits have the wrong order. Return zero
    when a batch contains only one class.

    :param logit: risk logits, shape (batch,)
    :param label: binary event labels, shape (batch,)
    :return: scalar ranking loss
    """
    pos = logit[label > 0.5]
    neg = logit[label < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logit.new_zeros(())
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return F.softplus(-diff).mean()


def beta_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Beta-NLL (Seitzer et al., 2022) weighted by var.detach()**beta.

    Apply the observation mask. beta=0 recovers ordinary Gaussian NLL.

    :param mean: predicted means, shape (batch, seq_len)
    :param logvar: predicted log variances, shape (batch, seq_len)
    :param target: measured values, shape (batch, seq_len)
    :param mask: validity mask, shape (batch, seq_len)
    :param beta: variance-weight exponent, commonly 0.5
    :return: scalar beta-NLL
    """
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    weighted = nll * var.detach() ** beta
    return masked_mean(weighted, mask)


def censored_nliq_loss(nliq_pred: torch.Tensor, nliq_target: torch.Tensor,
                       liq_label: torch.Tensor) -> torch.Tensor:
    """Censored event-time loss with a one-sided non-event penalty.

    Observed events use Smooth-L1. Right-censored records penalize predictions
    earlier than the censoring point and do not penalize later times.

    :param nliq_pred: normalized predictions, shape (batch,)
    :param nliq_target: normalized event/censoring times, shape (batch,)
    :param liq_label: binary event labels, shape (batch,)
    :return: scalar censored loss
    """
    obs = F.smooth_l1_loss(nliq_pred, nliq_target, reduction="none")
    cens = F.relu(nliq_target - nliq_pred)
    return (liq_label * obs + (1.0 - liq_label) * cens).mean()


def masked_censored_nliq_loss(nliq_pred: torch.Tensor, nliq_target: torch.Tensor,
                              liq_label: torch.Tensor,
                              observed: torch.Tensor = None) -> torch.Tensor:
    """Censored event-time loss with its own observation mask.

    Observed events use Smooth-L1 to exact time. Valid censored records use a
    one-sided penalty for predictions before the last observed cycle. Exclude
    records with observed==0. Completion status does not determine whether ordinary
    right censoring is valid.

    :param nliq_pred: normalized predictions, shape (batch,)
    :param nliq_target: normalized event/censoring times, shape (batch,)
    :param liq_label: binary event labels, shape (batch,)
    :param observed: nliq_censor_valid mask; None includes all records
    :return: scalar mean loss over included records
    """
    obs_term = F.smooth_l1_loss(nliq_pred, nliq_target, reduction="none")
    cens = F.relu(nliq_target - nliq_pred)
    per_sample = liq_label * obs_term + (1.0 - liq_label) * cens
    if observed is None:
        return per_sample.mean()
    weight = observed.to(per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp(min=1.0)


def monotone_clip(traj: torch.Tensor, lo: float = 0.0, hi: float = 1.05) -> torch.Tensor:
    """Project PPR to a nondecreasing trajectory bounded by [lo, hi].

    Use a cumulative maximum and clipping. The nondecreasing envelope is a modeling
    assumption of this study; drainage or variable loading can lower measured pressure.

    :param traj: pressure trajectory, shape (batch, seq_len)
    :return: bounded nondecreasing trajectory of the same shape
    """
    return torch.clamp(torch.cummax(traj, dim=1).values, lo, hi)


def monotone_residual_scale(traj: torch.Tensor, residual: torch.Tensor, span: float = 0.10,
                            free_increment: torch.Tensor = None) -> torch.Tensor:
    """Correct nonnegative pressure increments while retaining monotonicity.

    Multiply clamped increments by
    g=1+span*tanh(residual), in [1-span,1+span]>0, sum them and clip to [lo,hi].

    :param traj: base nondecreasing trajectory, shape (batch, seq_len)
    :param residual: correction-head output, shape (batch, seq_len)
    :param span: relative rate bound; 0.10 allows +/-10 percent
    :param free_increment: optional nonnegative additive increments, shape
        (batch, seq_len); None disables the term. Final bounds still apply.
    :return: corrected nondecreasing trajectory
    """
    if not 0.0 <= float(span) < 1.0:
        raise ValueError(f"span must satisfy 0 <= span < 1, got {span}")
    base0 = traj[:, :1]
    dr = torch.clamp(traj[:, 1:] - traj[:, :-1], min=0.0)
    gate = 1.0 + span * torch.tanh(residual[:, 1:])
    incr = dr * gate
    if free_increment is not None:
        incr = incr + torch.clamp(free_increment[:, 1:], min=0.0) # Model increments are nonnegative; the clamp guards against invalid inputs.
    return torch.cat([base0, base0 + torch.cumsum(incr, dim=1)], dim=1)


def normalized_free_increments(logits: torch.Tensor, raw_gate: torch.Tensor) -> torch.Tensor:
    """Distribute one learned non-negative correction mass over trajectory increments.

    The total mass is ``softplus(raw_gate)`` regardless of grid length. The first trajectory point
    has no preceding increment and therefore receives exactly zero.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (batch, time), got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        return torch.zeros_like(logits)
    weights = torch.softmax(logits[:, 1:], dim=1)
    return torch.cat([
        torch.zeros_like(logits[:, :1]),
        F.softplus(raw_gate) * weights,
    ], dim=1)


def interpolated_crossing(
    r: torch.Tensor,
    cycles: torch.Tensor,
    threshold: float = 0.95,
    beta: float = 25.0,
) -> torch.Tensor:
    """Interpolate the pressure-threshold crossing between bracketing grid nodes.

    The forward value lies on the curve. A smooth first-hitting CDF supplies the
    backward surrogate, including before a crossing occurs.

    :param r: pressure trajectory, shape (batch, seq_len)
    :param cycles: cycle grid, shape (batch, seq_len)
    :param threshold: event threshold
    :param beta: backward-surrogate steepness
    :return: straight-through N_liq, shape (batch,)
    """
    r_mono = torch.cummax(r, dim=1).values
    above = r_mono >= threshold
    has = above.any(dim=1)
    idx = torch.argmax(above.to(torch.int64), dim=1)
    idx0 = torch.clamp(idx - 1, min=0)
    ar = torch.arange(r.shape[0], device=r.device)
    r1 = r_mono[ar, idx]; r0 = r_mono[ar, idx0]
    c1 = cycles[ar, idx]; c0 = cycles[ar, idx0]
    dr = r1 - r0
    # dr is positive for a proper bracket. A first-node or degenerate bracket
    # has c0==c1; the safe denominator avoids 0/0 without changing the forward value.
    safe_dr = torch.where(dr > 0.0, dr, torch.ones_like(dr))
    frac = torch.clamp((threshold - r0) / safe_dr, 0.0, 1.0)
    hard = torch.where(has, c0 + frac * (c1 - c0), cycles[:, -1])

    # Use the smooth surrogate only in backward. Before a crossing,
    # residual horizon mass still depends on PPR and supplies a timing gradient.
    cdf = torch.sigmoid(float(beta) * (r_mono - threshold))
    prev = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], dim=1)
    event_mass = torch.clamp(cdf - prev, min=0.0)
    residual_mass = torch.clamp(1.0 - cdf[:, -1], min=0.0, max=1.0)
    soft = (event_mass * cycles).sum(dim=1) + residual_mass * cycles[:, -1]
    return soft + (hard - soft).detach()


def crossing_margin_loss(
    traj: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.95,
    margin: float = 0.0,
    observed: torch.Tensor = None,
    stable: torch.Tensor = None,
) -> torch.Tensor:
    """Penalize false crossings on completed non-event records only.

    Observed events need not exceed threshold+margin, since the record can end at
    onset. Unfinished records are excluded because their pressure can grow later.

    :param traj: forecast pressure, shape (batch, seq_len)
    :param label: binary labels, shape (batch,)
    :param margin: optional margin below threshold for completed non-events, default 0
    :param observed: observed binary-outcome mask when stable is not supplied
    :param stable: explicit completed-non-event mask
    :return: scalar penalty
    """
    m = traj.amax(dim=1)
    non = (label <= 0.5).to(traj.dtype)
    if stable is not None:
        stable_non = non * (stable > 0.5).to(traj.dtype)
    elif observed is not None:
        stable_non = non * (observed > 0.5).to(traj.dtype)
    else:
        stable_non = non
    penalty = torch.clamp(m - (threshold - margin), min=0.0) * stable_non
    return penalty.sum() / torch.clamp(stable_non.sum(), min=1.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean: sum(values*mask)/max(sum(mask),1).

    :param values: input tensor
    :param mask: same-shape binary validity mask
    :return: scalar masked mean
    """
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE = masked_mean((pred-target)^2).

    :param pred: predictions
    :param target: observed targets
    :param mask: validity mask
    :return: scalar MSE
    """
    return masked_mean((pred - target) ** 2, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MAE = masked_mean(abs(pred-target)).

    :param pred: predictions
    :param target: observed targets
    :param mask: validity mask
    :return: scalar MAE
    """
    return masked_mean(torch.abs(pred - target), mask)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Gaussian negative log likelihood.

    NLL = masked_mean(0.5*(logvar+(target-mean)^2*exp(-logvar))).
    Clip log variance to [-6, 3] for numerical stability.

    :param mean: predicted mean
    :param logvar: predicted log variance
    :param target: observed targets
    :param mask: validity mask
    :return: scalar NLL
    """
    logvar = torch.clamp(logvar, min=-6.0, max=3.0)
    inv_var = torch.exp(-logvar)
    return masked_mean(0.5 * (logvar + (target - mean) ** 2 * inv_var), mask)


def masked_bce_with_logits(logit: torch.Tensor, label: torch.Tensor,
                           observed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Risk BCE for observed binary outcomes.

    Exclude unfinished records with unknown by-H outcomes. An entirely unobserved
    batch returns a zero loss connected to the gradient graph.

    :param logit: risk logits, shape (N,)
    :param label: binary event labels, shape (N,)
    :param observed: risk_label_observed mask; None includes all records
    :return: scalar BCE, zero if no outcomes are observed
    """
    if observed is not None:
        m = observed > 0.5
        if bool(m.any()):
            return F.binary_cross_entropy_with_logits(logit[m], label[m])
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit, label)


def gaussian_mixture_nll(means: torch.Tensor, logvars: torch.Tensor,
                         target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NLL of an equally weighted Gaussian mixture.

    For S coefficient samples, p(y)=(1/S)*sum_s N(y;mu_s,sigma_s^2).
    Use logsumexp and average over masked observations.

    :param means: component means, shape (S, B, T)
    :param logvars: component log variances, shape (S, B, T)
    :param target: observations, shape (B, T)
    :param mask: validity mask, shape (B, T)
    :return: scalar mixture NLL
    """
    import math
    logvars = torch.clamp(logvars, min=-6.0, max=3.0)
    s = means.shape[0]
    t = target.unsqueeze(0)
    comp_logp = -0.5 * (math.log(2.0 * math.pi) + logvars
                        + (t - means) ** 2 * torch.exp(-logvars)) # (S, B, T)
    logp = torch.logsumexp(comp_logp, dim=0) - math.log(float(s)) # (B, T)
    return masked_mean(-logp, mask)


def energy_crps(samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked sample estimate of CRPS, the energy score with beta=1.

    CRPS ~= mean_s abs(y_s-y) - 0.5*mean_(s,s') abs(y_s-y_s').
    Samples are mu_s+sigma_s*epsilon. Equal samples reduce the score to MAE.

    :param samples: predictive samples, shape (S, B, T)
    :param target: observations, shape (B, T)
    :param mask: validity mask, shape (B, T)
    :return: scalar CRPS
    """
    t = target.unsqueeze(0)
    term1 = torch.abs(samples - t).mean(dim=0) # (B, T)
    diff = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1)).mean(dim=(0, 1)) # (B, T)
    return masked_mean(term1 - 0.5 * diff, mask)


def observed_aux_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    use_states: bool = True,
    w_g: float = 0.10,
    w_risk: float = 0.0,
    w_crr: float = 0.10,
) -> torch.Tensor:
    """Auxiliary losses for targets derived from measurements.

    Support g_obs, continuous risk_proxy and optional crr_obs/crr_obs_mask. Activate
    terms only when targets exist and weights are nonzero. Peak-PPR risk_proxy is
    distinct from the binary by-H event target.

    :param outputs: model outputs, including risk_prob and optionally g/crr
    :param batch: auxiliary targets and masks
    :param use_states: supervise g and the resistance coordinate
    :param w_g: trigger weight
    :param w_risk: continuous risk-proxy weight
    :param w_crr: measured-CRR weight
    :return: scalar auxiliary loss
    """
    device = outputs["risk_prob"].device if "risk_prob" in outputs else outputs["traj_mean"].device
    total = torch.zeros((), device=device)
    # risk_proxy=PPR_max is continuous and distinct from the by-H event.
    # Default w_risk=0 trains event probability with masked BCE.
    # If enabled, proxy loss uses the observed-outcome mask.
    if w_risk > 0 and "risk_proxy" in batch and "risk_prob" in outputs:
        _o = risk_observation_mask(batch)
        if _o is not None and bool((_o > 0.5).any()):
            _m = _o > 0.5
            total = total + w_risk * F.mse_loss(outputs["risk_prob"][_m], batch["risk_proxy"][_m])
        elif _o is None:
            total = total + w_risk * F.mse_loss(outputs["risk_prob"], batch["risk_proxy"])
    if use_states and "g_obs" in batch and "g" in outputs:
        total = total + w_g * masked_mse(outputs["g"], batch["g_obs"], batch["mask"])
    if use_states and "crr_obs" in batch and "crr" in outputs:
        mask = batch["mask"]
        per_sample = (((outputs["crr"] - batch["crr_obs"]) ** 2) * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        crr_mask = batch.get("crr_obs_mask")
        if crr_mask is not None:
            total = total + w_crr * (per_sample * crr_mask).sum() / torch.clamp(crr_mask.sum(), min=1.0)
        else:
            total = total + w_crr * per_sample.mean()
    return total


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copy model weights to detached CPU tensors.

    Used to retain a validation checkpoint without device or computation-graph ties.

    :param model: torch model
    :return: copied CPU state dictionary
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
