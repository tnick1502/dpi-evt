"""Matched Transformer controls and the reciprocal-cycle drift ablation."""
import torch
from dpi_evt.models.forecast_operators import CalibratedTransformer, anchor_and_cap
from dpi_evt.models.dpi_evt import DPIEvtNet
from dpi_evt.training.losses import (
    interpolated_crossing, forecast_gaussian_nll, masked_bce_with_logits,
    masked_censored_nliq_loss, masked_mean, nliq_censor_mask, risk_observation_mask,
)

class CrossingTransformer(CalibratedTransformer):
    """Same Transformer; projection in the forward pass and selectable time target.

    Head and curve controls have identical capacity, initialization, optimizer,
    pressure/risk/smoothness losses, and budgets. Only the primary time readout
    supplied to the .25 censored loss changes. Anchoring is a separate variant.
    """
    def __init__(self, static_dim, seq_dim, seq_len, hidden_dim=160,
                 n_heads=4, n_layers=2, curve_time_loss=True, anchor=False,
                 max_cycle_reference=3000.):
        super().__init__(static_dim, seq_dim, seq_len, hidden_dim, n_heads, n_layers)
        self.curve_time_loss=bool(curve_time_loss)
        self.anchor=bool(anchor)
        self.max_cycle_reference=float(max_cycle_reference)

    def forward_batch(self,batch):
        out=super().forward_batch(batch)
        r=anchor_and_cap(out['traj_mean'],batch,anchor=self.anchor,cap=False)
        n=interpolated_crossing(r,batch['cycles'],.95,beta=25.)
        out.update(traj_mean=r,nliq=n,nliq_norm=torch.log1p(n)/torch.log1p(n.new_tensor(self.max_cycle_reference)))
        return out

    def compute_loss(self,batch):
        out=self.forward_batch(batch)
        pressure=forecast_gaussian_nll(out['traj_mean'],out['traj_logvar'],batch['r_obs'],batch)
        risk=masked_bce_with_logits(out['risk_logit'],batch['label'],risk_observation_mask(batch))
        time_prediction=out['nliq_norm'] if self.curve_time_loss else out['nliq_pred']
        timing=masked_censored_nliq_loss(time_prediction,batch['n_liq_norm'],batch['label'],nliq_censor_mask(batch))
        smooth=masked_mean(torch.abs(out['traj_mean'][:,1:]-out['traj_mean'][:,:-1]),batch['mask'][:,1:])
        out.update(loss=pressure+.35*risk+.25*timing+.02*smooth,time_loss=timing)
        return out

class DPINoReciprocalDrift(DPIEvtNet):
    """Single-term ablation: zero reciprocal-cycle pressure drift in both regimes.

    All heads, parameter bounds, initialization, losses, integration, sampling,
    and resistance coordinates remain identical to DPI-EVT. The removed terms are
    set to zero without any outcome-based choice of constants, loss weights or drift.
    """
    def _engine(self,encoded,params,crr_dyn,batch,n_steps=None):
        params=dict(params)
        params['beta_pre']=torch.zeros_like(params['beta_pre'])
        params['beta_post']=torch.zeros_like(params['beta_post'])
        return super()._engine(encoded,params,crr_dyn,batch,n_steps=n_steps)
