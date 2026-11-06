"""DPI-EVT: conditional coefficients and a two-regime pressure-accumulation decoder.

The study uses a diagonal Gaussian latent law, a learned resistance-shaped
coordinate and threshold timing read from the pressure curve. This coordinate
does not identify material CRR. Study settings are in ``experiments.final_run.model_spec``."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpi_evt.models.latent import ConditionalCouplingFlow, flow_kl_per_dim
from dpi_evt.models.evt_ssm import EVTNeuralSSM
from dpi_evt.models.heads import physics_summary
from dpi_evt.models.blocks import ResidualMLP
from dpi_evt.training.losses import (crossing_margin_loss, energy_crps, gaussian_nll, gaussian_mixture_nll,
                                             interpolated_crossing,
                                             masked_bce_with_logits, masked_censored_nliq_loss, masked_mean,
                                             masked_mse, monotone_clip, monotone_residual_scale,
                                             normalized_free_increments, nliq_censor_mask,
                                             risk_observation_mask, soft_auc_loss,
                                             trajectory_supervision_masks)

__all__ = ["DPIEvtNet"]


class DPIEvtNet(EVTNeuralSSM):
    """Infer decoder coefficients and jointly forecast PPR and threshold-crossing time."""

    def __init__(self, static_dim: int, prefix_dim: int, seq_dim: int, seq_len: int, prefix_len: int,
                 max_cycle_reference: float, hidden_dim: int = 144, probabilistic: bool = True,
                 use_flow: bool = True, crr_from_damage: bool = True, crr_mode: str = None,
                 nliq_from_curve: bool = True, calibration_steps: int = 0, calibration_lr: float = 0.05,
                 use_traj_residual: bool = False, traj_residual_span: float = 0.10,
                 use_free_increment: bool = False, liq_threshold: float = 0.95,
                 use_observed_aux_loss: bool = True,
                 use_analytical_layer: bool = True, use_monotone_clip: bool = True,
                 use_discriminative_risk: bool = True, use_censored_nliq: bool = True,
                 crossing_margin_weight: float = 0.10, crossing_margin: float = 0.0,
                 kl_weight: float = 0.02,
                 mc_train_samples: int = 0, mc_crps_weight: float = 0.0, mc_predict_samples: int = 0,
                 report_nliq_from_curve: bool = True, nliq_head_aux_weight: float = 0.10,
                 **kwargs):
        """:param crr_mode: damage, empirical, hybrid or decoupled resistance mode
        :param nliq_from_curve: derive N_liq from the PPR threshold crossing
        :param calibration_steps: differentiable prefix-refinement steps; zero disables refinement
        :param use_traj_residual: enable a small zero-initialized trajectory correction
        :param liq_threshold: pressure threshold for curve-derived event timing
        :param report_nliq_from_curve: report the curve crossing while retaining an auxiliary time head
        :param nliq_head_aux_weight: censored-loss weight for the auxiliary N_liq head
        """
        super().__init__(static_dim, prefix_dim, seq_dim, seq_len, prefix_len, max_cycle_reference,
                         hidden_dim=hidden_dim, **kwargs)
        self.probabilistic = probabilistic
        self.use_flow = use_flow
        self.crr_mode = crr_mode if crr_mode is not None else ("damage" if crr_from_damage else "empirical")
        self.crr_from_damage = self.crr_mode in ("damage", "hybrid")
        self.nliq_from_curve = nliq_from_curve
        self.calibration_steps = calibration_steps
        self.calibration_lr = calibration_lr
        self.use_traj_residual = use_traj_residual
        self.liq_threshold = liq_threshold
        self.use_observed_aux_loss = use_observed_aux_loss
        self.use_analytical_layer = bool(use_analytical_layer)
        self.use_monotone_clip = bool(use_monotone_clip)
        self.use_discriminative_risk = bool(use_discriminative_risk)
        self.use_censored_nliq = bool(use_censored_nliq)
        self.crossing_margin_weight = float(crossing_margin_weight)
        self.crossing_margin = float(crossing_margin)
        self.kl_weight = float(kl_weight)
        # Optional mixture over theta samples: mixture NLL and energy CRPS.
        # Zero mc_train_samples uses the single-pass loss.
        self.mc_train_samples = int(mc_train_samples)
        self.mc_crps_weight = float(mc_crps_weight)
        self.mc_predict_samples = int(mc_predict_samples)
        self.report_nliq_from_curve = bool(report_nliq_from_curve)
        self.nliq_head_aux_weight = float(nliq_head_aux_weight)
        self._force_sample = False
        self.logvar_head = nn.Linear(hidden_dim, 33)
        # Conditional RealNVP with latent-dependent coupling and Jacobian log determinant.
        self.flow = ConditionalCouplingFlow(33, hidden_dim, n_layers=4, hidden=64)
        self.crr_ref_head = nn.Linear(hidden_dim, 3) # [CRR_ref, λ_crr, m_crr]
        self.traj_residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, seq_len))
        nn.init.zeros_(self.traj_residual[-1].weight)
        nn.init.zeros_(self.traj_residual[-1].bias)
        self.traj_residual_span = float(traj_residual_span)
        # Gate sets the total nonnegative correction mass; softmax distributes it
        # across nodes. Gate=-6 gives about 0.0025 total PPR, independent of seq_len.
        # Nonnegative increments and clipping retain monotonicity and ru<=1.
        self.use_free_increment = bool(use_free_increment)
        self.free_increment_head = nn.Linear(hidden_dim, seq_len)
        self.free_increment_gate = nn.Parameter(torch.tensor(-6.0))
        if not self.use_analytical_layer:
            context_dim = static_dim + prefix_dim + 2 * prefix_len
            self.direct_decoder = ResidualMLP(context_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
            self.direct_traj_head = nn.Linear(hidden_dim, seq_len)

    # Infer theta.
    def infer_theta(self, encoded):
        mu = self.param_head(encoded)
        raw_logvar = torch.clamp(self.logvar_head(encoded), min=-5.0, max=3.0)
        if self.probabilistic:
            stochastic = self.training or self._force_sample # _force_sample requests Monte Carlo sampling at inference.
            eps = torch.randn_like(mu) if stochastic else torch.zeros_like(mu)
            latent = mu + torch.exp(0.5 * raw_logvar) * eps
        else:
            latent = mu
            raw_logvar = torch.zeros_like(mu)
        return latent, mu, raw_logvar

    def _theta_from_latent(self, latent, encoded):
        """Transform the latent through conditional RealNVP; return theta and log absolute Jacobian determinant."""
        if self.use_flow:
            return self.flow(latent, encoded)
        return latent, torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype)

    # ---------- CRR ----------
    def crr_from_damage_law(self, params, crr_ref, cycles):
        inv_m = 1.0 / torch.clamp(params["m"], min=0.5).unsqueeze(1)
        arg = 1.0 + params["lambda_pre"].unsqueeze(1) * cycles
        return torch.clamp(crr_ref * torch.pow(arg, -inv_m), min=1e-4)

    def _compute_crr_pair(self, encoded, params, cycles):
        """Return dynamic resistance, reported resistance and consistency loss of shape (B,)."""
        head = self.crr_ref_head(encoded)
        crr_ref = 0.15 + 0.45 * torch.sigmoid(head[:, 0:1])
        cons = torch.zeros(encoded.shape[0], device=encoded.device)

        def power_crr():
            lam = 0.0005 + 0.02 * torch.sigmoid(head[:, 1:2])
            m = 1.0 + 2.4 * torch.sigmoid(head[:, 2:3])
            return torch.clamp(crr_ref * torch.pow(1.0 + lam * cycles, -1.0 / m), min=1e-4), lam, m

        if self.crr_mode == "empirical":
            crr_dyn = crr_out = self.compute_crr(params, cycles)
        elif self.crr_mode == "hybrid":
            crr_out, lam, m = power_crr(); crr_dyn = crr_out
            lp = params["lambda_pre"].unsqueeze(1); mp = params["m"].unsqueeze(1)
            cons = ((torch.log(lam) - torch.log(lp)) ** 2 + (1.0 / m - 1.0 / mp) ** 2).squeeze(1)
        elif self.crr_mode == "decoupled":
            crr_dyn = self.compute_crr(params, cycles)
            crr_out, _, _ = power_crr()
            cons = ((torch.log(crr_out + 1e-6) - torch.log(crr_dyn + 1e-6)) ** 2).mean(dim=1)
        else: # damage
            crr_dyn = crr_out = self.crr_from_damage_law(params, crr_ref, cycles)
        return crr_dyn, crr_out, cons

    # Recurrent EVT rollout.
    def _engine(self, encoded, params, crr_dyn, batch, n_steps=None):
        """Roll out states (z, r, c, g) for n_steps, defaulting to the complete sequence."""
        batch_size, seq_len = batch["cycles"].shape
        n_steps = seq_len if n_steps is None else min(n_steps, seq_len)
        eps = 1e-6
        cycles, dcyc, csr, seq_in = batch["cycles"], batch["delta_cycles"], batch["csr"], batch["seq_in"]

        def trigger(z_s):
            return torch.sigmoid(params["kappa"] * (z_s - params["z0"])) if self.use_trigger_head else torch.zeros_like(z_s)

        def derivatives(z_s, r_s, c_s, g_s, step):
            ratio = csr[:, step] / (crr_dyn[:, step] + eps)
            phi = F.softplus(6.0 * (ratio - 0.90)) / 6.0
            corr_in = torch.cat([encoded, seq_in[:, step, :], z_s[:, None], r_s[:, None], c_s[:, None], g_s[:, None]], dim=-1)
            raw = torch.tanh(self.correction_net(corr_in))
            dz_c = 0.004 * raw[:, 0]; dr_c = 0.004 * raw[:, 1]; dc_c = 0.5 * raw[:, 2]
            if self.use_crr_damage:
                dz_pre = (params["lambda_pre"] * torch.pow(torch.clamp(ratio, min=eps), params["m"])
                          * torch.pow(torch.clamp(1.0 - z_s, min=eps), params["nu"]) + dz_c)
            else:
                dz_pre = 0.10 * torch.tanh(dz_c) + 0.02 * phi
            dr_pre = (params["alpha_pre"] * phi * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_pre"])
                      + params["beta_pre"] / (cycles[:, step] + params["tau_pre"]) - params["gamma_pre"] * r_s + dr_c)
            if self.structured_post_event:
                dz_post = params["lambda_post"] * torch.clamp(1.0 - z_s, min=eps) + 0.5 * dz_c
                dr_post = (params["alpha_post"] * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_post"])
                           + params["beta_post"] / (cycles[:, step] + 0.70 * params["tau_pre"]) - params["gamma_post"] * r_s + 0.5 * dr_c)
            else:
                dz_post, dr_post = dz_pre, dr_pre
            return (1.0 - g_s) * dz_pre + g_s * dz_post, (1.0 - g_s) * dr_pre + g_s * dr_post, dc_c

        z_curr = torch.zeros(batch_size, device=cycles.device)
        r_curr = 0.10 * torch.sigmoid(self.r0_head(encoded)).squeeze(-1)
        c_curr = torch.zeros(batch_size, device=cycles.device)
        zs, rs, cs, gs = [z_curr], [r_curr], [c_curr], []
        for step in range(n_steps):
            g_step = trigger(z_curr); gs.append(g_step)
            if step == n_steps - 1:
                break
            dn = dcyc[:, step + 1]
            dz1, dr1, dc1 = derivatives(z_curr, r_curr, c_curr, g_step, step)
            if self.integrator == "heun":
                z_e = torch.clamp(z_curr + dn * dz1, 0.0, 0.999); r_e = torch.clamp(r_curr + dn * dr1, 0.0, 1.0)
                c_e = torch.tanh(params["c_decay"] * c_curr + dc1)
                dz2, dr2, dc2 = derivatives(z_e, r_e, c_e, trigger(z_e), step)
                dz_s, dr_s, dc_s = 0.5 * (dz1 + dz2), 0.5 * (dr1 + dr2), 0.5 * (dc1 + dc2)
            else:
                dz_s, dr_s, dc_s = dz1, dr1, dc1
            z_curr = torch.clamp(z_curr + dn * dz_s, 0.0, 0.999)
            # Project damage to prevent a negative neural correction from decreasing z.
            z_curr = torch.maximum(z_curr, zs[-1])
            r_curr = torch.clamp(r_curr + dn * dr_s, 0.0, 1.0)
            c_curr = torch.tanh(params["c_decay"] * c_curr + dc_s)
            zs.append(z_curr); rs.append(r_curr); cs.append(c_curr)
        # The cumulative maximum makes the event trigger nondecreasing.
        return torch.stack(zs, 1), torch.stack(rs, 1), torch.stack(cs, 1), torch.cummax(torch.stack(gs, 1), dim=1).values

    # Differentiable coefficient refinement on the prefix.
    def calibrate_theta(self, latent, encoded, batch):
        if self.calibration_steps <= 0 or not self.use_analytical_layer:
            return latent
        anchor = latent
        work = latent.detach()
        for _ in range(self.calibration_steps):
            work = work.detach().requires_grad_(True)
            with torch.enable_grad():
                theta, _ = self._theta_from_latent(work, encoded) # Select theta from (theta, log_det).
                params = self.unpack_params(theta)
                crr_dyn, _, _ = self._compute_crr_pair(encoded, params, batch["cycles"])
                _, r_pref, _, _ = self._engine(encoded, params, crr_dyn, batch, n_steps=self.prefix_len)
                r_pref = monotone_clip(r_pref)
                loss = masked_mse(r_pref, batch["prefix_obs"][:, : self.prefix_len],
                                  batch["prefix_mask"][:, : self.prefix_len])
            grad = torch.autograd.grad(loss, work, retain_graph=False, create_graph=False)[0]
            work = work - self.calibration_lr * grad
        return anchor + (work.detach() - anchor.detach()) # straight-through

    # Read N_liq from the PPR curve.
    def _nliq_from_curve(self, r, cycles, beta: float = 25.0):
        """Read threshold time from a monotone PPR curve; return (N_liq, pdf, mass).

        interpolated_crossing interpolates between the nodes bracketing the threshold.
        Soft pdf/mass support resistance supervision at the crossing.
        """
        p = torch.sigmoid(beta * (r - self.liq_threshold)) # Nondecreasing because r is nondecreasing.
        dp = torch.clamp(p[:, 1:] - p[:, :-1], min=0.0)
        pdf = torch.cat([p[:, :1], dp], dim=1) # Soft crossing mass for resistance-at-crossing supervision.
        mass = pdf.sum(dim=1)
        nliq = interpolated_crossing(r, cycles, self.liq_threshold, beta=beta)
        return nliq, pdf, mass

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = self.build_context(batch)
        encoded = self.context_encoder(context)
        latent, mu, raw_logvar = self.infer_theta(encoded)
        latent = self.calibrate_theta(latent, encoded, batch) # Prefix refinement; zero steps leaves theta unchanged.
        theta, log_det = self._theta_from_latent(latent, encoded)
        params = self.unpack_params(theta)
        crr_dyn, crr_out, crr_cons = self._compute_crr_pair(encoded, params, batch["cycles"])

        if self.use_analytical_layer:
            z, r, c, g = self._engine(encoded, params, crr_dyn, batch)
        else:
            decoded = self.direct_decoder(context)
            r = torch.sigmoid(self.direct_traj_head(decoded))
            z = torch.sigmoid(6.0 * r - 2.5)
            g = torch.sigmoid(8.0 * r - 4.0)
            c = torch.zeros_like(r)
            crr_out = torch.full_like(r, 0.20)
            crr_cons = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
        if self.use_traj_residual or self.use_free_increment:
            # Scale accumulation rates and add optional nonnegative increments
            # while preserving the nondecreasing curve.
            resid = self.traj_residual(encoded) if self.use_traj_residual else torch.zeros_like(r)
            free = None
            if self.use_free_increment:
                free = normalized_free_increments(
                    self.free_increment_head(encoded), self.free_increment_gate)
            if self.use_monotone_clip:
                r = monotone_residual_scale(r, resid, span=self.traj_residual_span, free_increment=free)
            else:
                r = r + self.traj_residual_span * torch.tanh(resid)
        if self.use_monotone_clip:
            r = monotone_clip(r, hi=1.0) # Clip to ru<=1.
            z = monotone_clip(z, hi=1.0)
            g = monotone_clip(g, hi=1.0)
        else:
            r = torch.clamp(r, 0.0, 1.0)
        if not self.use_analytical_layer:
            # Keep the black-box control internally coherent after its optional projection/residual.
            z = torch.sigmoid(6.0 * r - 2.5)
            g = torch.sigmoid(8.0 * r - 4.0)

        mcr = torch.as_tensor(self.max_cycle_reference, device=r.device, dtype=r.dtype)
        if self.nliq_from_curve:
            nliq_curve, cross_pdf, cross_mass = self._nliq_from_curve(r, batch["cycles"])
        else:
            nliq_curve = self.soft_first_hitting(r, g, batch["cycles"])
            cross_pdf = torch.zeros_like(r); cross_mass = torch.zeros(r.shape[0], device=r.device)
        nliq_norm_curve = torch.log1p(nliq_curve) / torch.log1p(mcr)
        # Report N_liq from the final pressure-curve crossing.
        # The direct head supplies an auxiliary training target.
        nliq_head, nliq_norm_head, nliq_norm_curve = self._apply_nliq_head(encoded, nliq_norm_curve, mcr)
        if self.report_nliq_from_curve:
            nliq, nliq_norm = nliq_curve, nliq_norm_curve
        else:
            nliq, nliq_norm = nliq_head, nliq_norm_head
        summary = physics_summary(r, z, g, nliq_norm)
        risk_prior = 6.0 * (0.50 * r.amax(dim=1) + 0.25 * g.amax(dim=1) + 0.25 * z.amax(dim=1) - 0.75)
        risk_logit = self.prior_gate * risk_prior + self.risk_head(encoded, summary)
        if self.use_discriminative_risk:
            risk_logit = risk_logit + self.risk_clf(encoded).squeeze(-1)
        traj_logvar = self.logvar_head_seq(encoded) + 2.0 * self.calib_log_scale
        # Flow KL includes log_det; without flow use the Gaussian KL.
        if not self.probabilistic:
            kl = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
        elif self.use_flow:
            kl = flow_kl_per_dim(latent, mu, raw_logvar, theta, log_det)
        else:
            kl = 0.5 * (torch.exp(raw_logvar) + mu.pow(2) - 1.0 - raw_logvar).mean(dim=1)
        return {
            "traj_mean": r, "ru": r, "damage": z, "traj_logvar": traj_logvar,
            "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit),
            "nliq": nliq, "nliq_norm": nliq_norm, "nliq_norm_curve": nliq_norm_curve,
            "nliq_head": nliq_head, "nliq_norm_head": nliq_norm_head,
            "z": z, "g": g, "c": c, "crr": crr_out, "kl": kl,
            "crr_consistency": crr_cons, "cross_pdf": cross_pdf, "cross_mass": cross_mass,
        }

    def predictive(self, batch: Dict[str, torch.Tensor], mc_samples: int = 8) -> Dict[str, torch.Tensor]:
        """Monte Carlo prediction from K sampled coefficient vectors.

        Trajectory variance combines mean emission variance exp(logvar) and variance
        between sampled trajectory means. Use forward_batch when the model is
        deterministic or mc_samples<=1.
        """
        if not self.probabilistic or mc_samples <= 1:
            return self.forward_batch(batch)
        prev = self._force_sample
        self._force_sample = True
        try:
            means, alea, risks, nliqs, crrs, last = [], [], [], [], [], None
            for _ in range(int(mc_samples)):
                o = self.forward_batch(batch)
                means.append(o["traj_mean"]); alea.append(torch.exp(o["traj_logvar"]))
                risks.append(o["risk_prob"])
                if "nliq_norm" in o:
                    nliqs.append(o["nliq_norm"])
                if "crr" in o and torch.is_tensor(o["crr"]):
                    crrs.append(o["crr"])
                last = o
        finally:
            self._force_sample = prev
        M = torch.stack(means, 0)
        pred_var = torch.stack(alea, 0).mean(0) + M.var(0, unbiased=False)
        out = dict(last)
        out["traj_mean"] = M.mean(0)
        out["traj_logvar"] = torch.log(pred_var.clamp_min(1e-12))
        out["traj_epistemic_var"] = M.var(0, unbiased=False)
        from dpi_evt.evaluation.event_distribution import hitting_time_distribution
        out.update(hitting_time_distribution(M, batch["cycles"], self.liq_threshold))
        rp = torch.stack(risks, 0).mean(0).clamp(1e-6, 1 - 1e-6)
        out["risk_prob"] = rp
        out["risk_logit"] = torch.log(rp) - torch.log1p(-rp)
        if crrs:
            out["crr"] = torch.stack(crrs, 0).mean(0)
        if nliqs:
            nn_samples = torch.stack(nliqs, 0)
            mcr = torch.as_tensor(self.max_cycle_reference, device=out["traj_mean"].device,
                                  dtype=out["traj_mean"].dtype)
            if self.report_nliq_from_curve:
                # Align the point time with the reported Monte Carlo mean curve.
                # Quantiles retain the variation of individual sampled-path crossings.
                nliq_point, cross_pdf, cross_mass = self._nliq_from_curve(
                    out["traj_mean"], batch["cycles"],
                )
                out["nliq"] = nliq_point
                out["nliq_norm"] = torch.log1p(nliq_point) / torch.log1p(mcr)
                out["nliq_norm_curve"] = out["nliq_norm"]
                out["cross_pdf"] = cross_pdf
                out["cross_mass"] = cross_mass
            else:
                nn_ = nn_samples.mean(0)
                out["nliq_norm"] = nn_
                out["nliq"] = torch.expm1(nn_ * torch.log1p(mcr))
            cyc_samples = torch.expm1(nn_samples * torch.log1p(mcr))
            # q05/q95 above come from actual path crossings, including an explicit T>H tail.
        return out

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from dpi_evt.training.losses import observed_aux_loss
        out = self.forward_batch(batch)
        # Primary trajectory loss uses the continuation mask;
        # prefix reconstruction has weight 0.15. Positive mc_train_samples
        # enables mixture NLL over theta samples and optional CRPS.
        cont_mask, pref_mask = trajectory_supervision_masks(batch)
        _PREFIX_AUX_W = 0.15
        if self.mc_train_samples > 0 and self.probabilistic:
            mus, logvars = [], []
            prev = self._force_sample
            self._force_sample = True
            try:
                for _ in range(self.mc_train_samples):
                    o = self.forward_batch(batch)
                    mus.append(o["traj_mean"]); logvars.append(o["traj_logvar"])
            finally:
                self._force_sample = prev
            mc_means = torch.stack(mus, 0); mc_logvars = torch.stack(logvars, 0)
            traj_loss = gaussian_mixture_nll(mc_means, mc_logvars, batch["r_obs"], cont_mask)
            traj_loss = traj_loss + _PREFIX_AUX_W * gaussian_mixture_nll(mc_means, mc_logvars, batch["r_obs"], pref_mask)
            if self.mc_crps_weight > 0.0:
                samples = mc_means + torch.exp(0.5 * mc_logvars) * torch.randn_like(mc_means)
                traj_loss = traj_loss + self.mc_crps_weight * energy_crps(samples, batch["r_obs"], cont_mask)
        else:
            traj_loss = gaussian_nll(out["traj_mean"], out["traj_logvar"], batch["r_obs"], cont_mask)
            traj_loss = traj_loss + _PREFIX_AUX_W * gaussian_nll(
                out["traj_mean"], out["traj_logvar"], batch["r_obs"], pref_mask)
        # Risk and soft-AUC losses use observed binary outcomes only.
        _robs = risk_observation_mask(batch)
        risk_loss = masked_bce_with_logits(out["risk_logit"], batch["label"], _robs)
        _m = (_robs > 0.5) if _robs is not None else torch.ones_like(batch["label"], dtype=torch.bool)
        if self.use_discriminative_risk:
            rank_loss = (soft_auc_loss(out["risk_logit"][_m], batch["label"][_m]) if bool(_m.any())
                         else out["risk_logit"].sum() * 0.0)
        else:
            rank_loss = torch.zeros((), device=out["traj_mean"].device)
        def event_time_loss(prediction):
            if self.use_censored_nliq:
                return masked_censored_nliq_loss(
                    prediction, batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch),
                )
            obsm = nliq_censor_mask(batch)
            obsm = torch.ones_like(batch["n_liq_norm"]) if obsm is None else obsm
            loss = (((prediction - batch["n_liq_norm"]) ** 2) * obsm).sum()
            return loss / obsm.sum().clamp_min(1.0)

        nliq_loss = event_time_loss(out["nliq_norm"])
        # With curve-first output, train the direct head with a separate auxiliary loss.
        # In head-first mode the main event-time loss already trains that head.
        if self.report_nliq_from_curve and getattr(self, "use_nliq_head", False):
            nliq_head_loss = event_time_loss(out["nliq_norm_head"])
        else:
            nliq_head_loss = torch.zeros((), device=out["traj_mean"].device)
        switch_reg = torch.abs(out["g"][:, 1:] - out["g"][:, :-1]).mean()
        state_smooth = (torch.abs(out["traj_mean"][:, 2:] - 2 * out["traj_mean"][:, 1:-1] + out["traj_mean"][:, :-2]).mean()
                        + torch.abs(out["z"][:, 2:] - 2 * out["z"][:, 1:-1] + out["z"][:, :-2]).mean())
        kl_loss = out["kl"].mean() if self.probabilistic else torch.zeros((), device=out["traj_mean"].device)

        # Link outputs of the common state at the event time.
        liq = batch["label"]; denom = torch.clamp(liq.sum(), min=1.0)
        joint = torch.zeros((), device=out["traj_mean"].device)
        if self.use_analytical_layer and out["cross_mass"].sum() > 0:
            m = torch.clamp(out["cross_mass"], min=1e-4)
            crr_at = (out["cross_pdf"] * out["crr"]).sum(1) / m # Resistance at the event time.
            z_at = (out["cross_pdf"] * out["z"]).sum(1) / m # Damage at the event time.
            csr_at = (out["cross_pdf"] * batch["csr"]).sum(1) / m
            crr_cons_liq = (liq * (crr_at - csr_at) ** 2).sum() / denom # CRR(N_liq) ≈ CSR(N_liq)
            dmg_cons_liq = (liq * (z_at - 0.90) ** 2).sum() / denom # Damage(N_liq) approximates the trigger threshold.
            joint = crr_cons_liq + dmg_cons_liq

        # Overshoot and non-event trigger penalties.
        # Penalize pressure above the measured curve, including on censored records.
        # Suppress the trigger only for completed non-event records
        # with regime_stable==1; unfinished records can accumulate further pressure.
        noliq = (1.0 - batch["label"]).unsqueeze(1)
        overshoot = masked_mean(torch.relu(out["traj_mean"] - batch["r_obs"]) * noliq, batch["mask"])
        stable = batch.get("regime_stable")
        stab = noliq if stable is None else stable.unsqueeze(1)
        trigger_noliq = masked_mean(out["g"] * stab, batch["mask"]) if "g" in out else torch.zeros((), device=out["traj_mean"].device)

        # Align the direct time head with curve crossing on observed events.
        nliq_consistency = self._nliq_consistency_loss(out, batch)
        # Align predicted risk with threshold crossing of the pressure path
        # to penalize disagreement between the two outputs.
        p_cross = torch.sigmoid(12.0 * (out["traj_mean"].amax(dim=1) - self.liq_threshold))
        coherence = ((out["risk_prob"] - p_cross) ** 2).mean()
        if self.crossing_margin_weight > 0.0:
            margin_loss = crossing_margin_loss(
                out["traj_mean"], batch["label"], threshold=self.liq_threshold,
                margin=self.crossing_margin, observed=_robs, stable=batch.get("regime_stable"),
            )
        else:
            margin_loss = torch.zeros((), device=out["traj_mean"].device)
        loss = (traj_loss + 0.80 * risk_loss + 0.30 * rank_loss + 0.25 * nliq_loss
                + self.nliq_head_aux_weight * nliq_head_loss
                + 0.10 * nliq_consistency
                + 0.02 * switch_reg + 0.01 * state_smooth + self.kl_weight * kl_loss
                + 0.01 * out["crr_consistency"].mean() + 0.05 * joint
                + 0.20 * overshoot + 0.05 * trigger_noliq + 0.05 * coherence
                + self.crossing_margin_weight * margin_loss)
        if self.use_observed_aux_loss:
            loss = loss + observed_aux_loss(out, batch, use_states=self.use_analytical_layer)
        out["loss"] = loss
        return out
