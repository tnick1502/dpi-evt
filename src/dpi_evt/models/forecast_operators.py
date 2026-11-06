"""Shared prefix anchoring/projection operators for internal and external evaluation."""
from __future__ import annotations

import torch

from dpi_evt.training.losses import interpolated_crossing
from dpi_evt.models.baselines import TransformerBaseline


class CalibratedTransformer(TransformerBaseline):
    """Unchanged Transformer architecture with a saved validation-only variance multiplier."""
    def __init__(self, static_dim, seq_dim, seq_len, hidden_dim=96, n_heads=4, n_layers=2):
        super().__init__(static_dim, seq_dim, seq_len, hidden_dim, n_heads, n_layers)
        self.register_buffer("calib_log_scale", torch.tensor(0.))

    def forward_batch(self, batch):
        out = super().forward_batch(batch)
        out["traj_logvar"] = out["traj_logvar"] + 2 * self.calib_log_scale
        return out


def anchor_and_cap(r, batch, anchor=True, cap=True):
    r = torch.cummax(r.clamp(0, 1), dim=1).values
    pm = batch["prefix_mask"] > .5
    has = pm.any(dim=1)
    idx = torch.arange(r.shape[1], device=r.device).expand_as(r)
    j = torch.where(pm, idx, -torch.ones_like(idx)).max(dim=1).values.clamp_min(0)
    pobs = torch.cummax(torch.where(pm, batch["prefix_obs"], 0.), dim=1).values
    r_o = pobs.gather(1, j[:, None])
    if anchor:
        r0 = r.gather(1, j[:, None])
        inc = ((r - r0) / (1 - r0).clamp_min(1e-3)).clamp_min(0)
        anchored = torch.where(idx > j[:, None], r_o + (1 - r_o) * inc, pobs)
        r = torch.where(has[:, None], anchored, r).clamp(0, 1)
    if cap:
        if "ru_cap" not in batch:
            raise ValueError("C operator requires a prescribed ru_cap")
        ceiling = batch["ru_cap"].to(r)[:, None]
        # A measured value above the cap is kept, and the caller flags the record.
        if anchor:
            ceiling = torch.maximum(ceiling, r_o)
        r = torch.minimum(r, ceiling)
    return r


def make_transfer_class(Base, family="generic"):
    class Transfer(Base):
        def __init__(self, *args, t_anchor=True, t_cap=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.t_anchor, self.t_cap = bool(t_anchor), bool(t_cap)

        def forward_batch(self, batch):
            out = super().forward_batch(batch)
            if "traj_mean" not in out:
                return out
            r = anchor_and_cap(out["traj_mean"], batch, self.t_anchor, self.t_cap)
            out["traj_mean"] = r
            if "ru" in out:
                out["ru"] = r
            thr = float(getattr(self, "liq_threshold", .95))
            if hasattr(self, "_nliq_from_curve"):
                n, pdf, mass = self._nliq_from_curve(r, batch["cycles"])
                out["cross_pdf"], out["cross_mass"] = pdf, mass
            else:
                n = interpolated_crossing(r, batch["cycles"], thr)
            H = float(getattr(self, "max_cycle_reference", 3000.))
            out["nliq"] = n
            out["nliq_norm"] = torch.log1p(n) / torch.log1p(torch.as_tensor(H, device=r.device))
            out["nliq_norm_curve"] = out["nliq_norm"]
            out["curve_has_crossing"] = (r >= thr).any(1).to(r.dtype)
            if self.t_cap and "risk_prob" in out:
                # The smooth gate is a classifier heuristic and is not used as F(H).
                gate = torch.sigmoid((batch["ru_cap"].to(r) - thr) / .01)
                p = (out["risk_prob"] * gate).clamp(1e-6, 1 - 1e-6)
                out["risk_prob"], out["risk_logit"] = p, torch.logit(p)
            return out

    Transfer.__name__ = Base.__name__ + "_T"
    return Transfer
