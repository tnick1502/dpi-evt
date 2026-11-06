"""EVT-NeuralSSM: a state-space model with smoothly switched regimes.

h_t = [z_t, r_t, c_t] contains damage, pressure and auxiliary memory. The CRR/CSR
relation and a neural correction through c_t determine the dynamics.
g_t = sigmoid(kappa*(z_t-z0)) mixes pre-event and post-event regimes.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpi_evt.models.blocks import ResidualMLP
from dpi_evt.models.heads import RiskHead, SeqLogvarHead, physics_summary
from dpi_evt.training.losses import (gaussian_nll,
                                             interpolated_crossing,
                                             masked_bce_with_logits,
                                             masked_censored_nliq_loss, masked_mean, monotone_clip,
                                             nliq_censor_mask, observed_aux_loss,
                                             risk_observation_mask, soft_auc_loss,
                                             trajectory_supervision_masks)

__all__ = ["EVTNeuralSSM"]


class EVTNeuralSSM(nn.Module):
    """Event-switched structured state-space model.

    A residual MLP maps context to constrained parameters. Recurrently integrate
    damage z and pressure r with smooth regime switching. Constructor switches
    support mechanism ablations.
    """

    def __init__(
        self,
        static_dim: int,
        prefix_dim: int,
        seq_dim: int,
        seq_len: int,
        prefix_len: int,
        max_cycle_reference: float,
        hidden_dim: int = 144,
        use_trigger_head: bool = True,
        structured_post_event: bool = True,
        use_crr_damage: bool = True,
        integrator: str = "heun",
        nliq_from_curve: bool = True,
        liq_threshold: float = 0.95, # Pressure event threshold, ru>=0.95.
        use_observed_aux_loss: bool = True,
        use_nliq_head: bool = True,
        report_nliq_from_curve: bool = True,
        nliq_head_aux_weight: float = 0.10,
    ):
        """:param static_dim: static-feature count
        :param prefix_dim: prefix-summary width
        :param seq_dim: sequence channels per node
        :param seq_len: sequence length
        :param prefix_len: observed prefix length
        :param max_cycle_reference: horizon for event-time normalization
        :param hidden_dim: encoder width
        :param use_trigger_head: enable the smooth event trigger g
        :param structured_post_event: enable structured post-event dynamics
        :param use_crr_damage: enable resistance-based damage dynamics
        :param integrator: heun (RK2) or euler integration
        """
        super().__init__()
        self.seq_len = seq_len
        self.use_trigger_head = use_trigger_head
        self.structured_post_event = structured_post_event
        self.use_crr_damage = use_crr_damage
        self.integrator = integrator
        # Curve-derived event time and optional joint-consistency loss.
        self.nliq_from_curve = nliq_from_curve # Readout shared with DPI-EVT.
        self.liq_threshold = liq_threshold
        self.use_observed_aux_loss = use_observed_aux_loss
        # Joint consistency is disabled by default for plain EVT;
        # empirical CRR directly governs this model's dynamics.
        self.use_joint_consistency = False
        self.prefix_len = prefix_len
        self.max_cycle_reference = max_cycle_reference

        context_dim = static_dim + prefix_dim + 2 * self.prefix_len
        self.context_encoder = ResidualMLP(context_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.param_head = nn.Linear(hidden_dim, 33)
        # Correction network with LayerNorm.
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim + seq_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        # Learned initial PPR state.
        self.r0_head = nn.Linear(hidden_dim, 1)
        # Risk and heteroscedastic-uncertainty heads.
        self.risk_head = RiskHead(hidden_dim)
        # Context risk head with a learned gate on the structured prior.
        self.risk_clf = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                      nn.Linear(hidden_dim // 2, 1))
        self.prior_gate = nn.Parameter(torch.zeros(1))
        self.logvar_head_seq = SeqLogvarHead(hidden_dim, seq_len)
        # Predict nliq_norm directly from context; consistency loss
        # aligns it with curve crossing on observed events.
        # Changing this head changes the architecture and requires refitting.
        self.use_nliq_head = use_nliq_head
        # Curve-first reporting uses the final pressure-curve crossing.
        # The direct head remains an auxiliary predictor.
        self.report_nliq_from_curve = bool(report_nliq_from_curve)
        self.nliq_head_aux_weight = float(nliq_head_aux_weight)
        self.nliq_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                       nn.Linear(hidden_dim // 2, 1))
        # Fold-local variance scale: std *= exp(calib_log_scale).
        # Scaling alone supplies no coverage guarantee.
        self.register_buffer("calib_log_scale", torch.zeros(1))

    def build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine static, prefix-summary, observation and mask features into context.

        :param batch: input dictionary
        :return: shape (batch, static_dim + prefix_dim + 2*prefix_len)
        """
        return torch.cat(
            [
                batch["static"],
                batch["prefix_summary"],
                batch["prefix_obs"][:, : self.prefix_len],
                batch["prefix_mask"][:, : self.prefix_len],
            ],
            dim=-1,
        )

    def unpack_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map raw coefficients to constrained decoder parameters.

        Includes resistance-mixture weights, amplitudes, rates and tails, damage/pressure
        ODE parameters in both regimes, memory decay and noise.

        :param theta: raw tensor, shape (batch, 33)
        :return: constrained parameter dictionary
        """
        weights = torch.softmax(theta[:, 0:4], dim=-1)
        amp = 0.03 + 0.28 * torch.sigmoid(theta[:, 4:8])
        rate = 0.05 + 1.15 * torch.sigmoid(theta[:, 8:12])
        tail = 0.005 + 0.18 * torch.sigmoid(theta[:, 12:16])
        lambda_pre = 0.0001 + 0.0050 * torch.sigmoid(theta[:, 16])
        exponent_m = 1.0 + 2.4 * torch.sigmoid(theta[:, 17])
        exponent_nu = 1.0 + 1.8 * torch.sigmoid(theta[:, 18])
        alpha_pre = 0.0004 + 0.0065 * torch.sigmoid(theta[:, 19])
        beta_pre = 0.01 + 0.18 * torch.sigmoid(theta[:, 20])
        gamma_pre = 0.00005 + 0.0020 * torch.sigmoid(theta[:, 21])
        exponent_p_pre = 0.95 + 1.8 * torch.sigmoid(theta[:, 22])
        tau_pre = 3.0 + 42.0 * torch.sigmoid(theta[:, 23])
        lambda_post = 0.0002 + 0.0075 * torch.sigmoid(theta[:, 24])
        alpha_post = 0.0004 + 0.0080 * torch.sigmoid(theta[:, 25])
        beta_post = 0.01 + 0.16 * torch.sigmoid(theta[:, 26])
        gamma_post = 0.00005 + 0.0024 * torch.sigmoid(theta[:, 27])
        exponent_p_post = 0.80 + 1.6 * torch.sigmoid(theta[:, 28])
        kappa = 6.0 + 20.0 * torch.sigmoid(theta[:, 29])
        z0 = 0.35 + 0.50 * torch.sigmoid(theta[:, 30])
        c_decay = 0.50 + 0.45 * torch.sigmoid(theta[:, 31])
        noise = 0.01 + 0.08 * torch.sigmoid(theta[:, 32])
        return {
            "weights": weights,
            "amp": amp,
            "rate": rate,
            "tail": tail,
            "lambda_pre": lambda_pre,
            "m": exponent_m,
            "nu": exponent_nu,
            "alpha_pre": alpha_pre,
            "beta_pre": beta_pre,
            "gamma_pre": gamma_pre,
            "p_pre": exponent_p_pre,
            "tau_pre": tau_pre,
            "lambda_post": lambda_post,
            "alpha_post": alpha_post,
            "beta_post": beta_post,
            "gamma_post": gamma_post,
            "p_post": exponent_p_post,
            "kappa": kappa,
            "z0": z0,
            "c_decay": c_decay,
            "noise": noise,
        }

    def compute_crr(self, params: Dict[str, torch.Tensor], cycles: torch.Tensor) -> torch.Tensor:
        """Construct resistance CRR(N) as a differentiable mixture of four families.

        :param params: weights, amplitudes, rates and tails
        :param cycles: query cycles, shape (batch, seq_len)
        :return: resistance curve, shape (batch, seq_len)
        """
        n_rel = cycles / 100.0
        crr_h = params["tail"][:, 0:1] + params["amp"][:, 0:1] / (1.0 + params["rate"][:, 0:1] * n_rel)
        crr_p = params["tail"][:, 1:2] + params["amp"][:, 1:2] * torch.pow(1.0 + n_rel, -(0.15 + params["rate"][:, 1:2]))
        crr_e = params["tail"][:, 2:3] + params["amp"][:, 2:3] * torch.exp(-params["rate"][:, 2:3] * n_rel)
        crr_l = params["tail"][:, 3:4] + params["amp"][:, 3:4] / (1.0 + params["rate"][:, 3:4] * torch.log1p(4.0 * n_rel))
        crr = (
            params["weights"][:, 0:1] * crr_h
            + params["weights"][:, 1:2] * crr_p
            + params["weights"][:, 2:3] * crr_e
            + params["weights"][:, 3:4] * crr_l
        )
        return torch.clamp(crr, min=1e-4)

    def soft_first_hitting(self, r: torch.Tensor, g: torch.Tensor, cycles: torch.Tensor) -> torch.Tensor:
        """Estimate N_liq by differentiable first hitting.

        :param r: pressure path, shape (batch, seq_len)
        :param g: trigger path, shape (batch, seq_len)
        :param cycles: query cycles, shape (batch, seq_len)
        :return: N_liq, shape (batch,)
        """
        event_prob = torch.maximum(torch.sigmoid(12.0 * (r - self.liq_threshold)), g)
        shifted_survival = torch.cumprod(
            torch.cat([torch.ones_like(event_prob[:, :1]), torch.clamp(1.0 - event_prob[:, :-1], min=1e-4)], dim=1),
            dim=1,
        )
        event_mass = event_prob * shifted_survival
        residual_mass = torch.clamp(1.0 - event_mass.sum(dim=1), min=0.0, max=1.0)
        return (event_mass * cycles).sum(dim=1) + residual_mass * cycles[:, -1]

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Integrate the switched state dynamics and return forecast outputs.

        Each step evaluates CSR/CRR, the memory-mediated correction, and pre/post-event
        increments mixed by g.

        :param batch: input dictionary
        :return: traj_mean, traj_logvar, risk_logit, risk_prob, nliq, nliq_norm,
            z, g, c and crr
        """
        context = self.build_context(batch)
        encoded = self.context_encoder(context)
        params = self.unpack_params(self.param_head(encoded))
        crr = self.compute_crr(params, batch["cycles"])

        batch_size, seq_len = batch["cycles"].shape
        eps = 1e-6
        cycles, dcyc, csr, seq_in = batch["cycles"], batch["delta_cycles"], batch["csr"], batch["seq_in"]

        def trigger(z_state: torch.Tensor) -> torch.Tensor:
            """Smooth event trigger g = sigmoid(kappa*(z-z0))."""
            if self.use_trigger_head:
                return torch.sigmoid(params["kappa"] * (z_state - params["z0"]))
            return torch.zeros_like(z_state)

        def derivatives(z_s, r_s, c_s, g_s, step):
            """Evaluate (dz, dr, dc) from the structured dynamics and neural correction."""
            ratio = csr[:, step] / (crr[:, step] + eps)
            phi = F.softplus(6.0 * (ratio - 0.90)) / 6.0
            corr_in = torch.cat([encoded, seq_in[:, step, :], z_s[:, None], r_s[:, None], c_s[:, None], g_s[:, None]], dim=-1)
            raw_corr = torch.tanh(self.correction_net(corr_in))
            # Scale neural rate corrections relative to the structured rates
            # before multiplying by potentially large cycle increments.
            dz_corr = 0.004 * raw_corr[:, 0]
            dr_corr = 0.004 * raw_corr[:, 1]
            dc_corr = 0.5 * raw_corr[:, 2]
            if self.use_crr_damage:
                dz_pre = (params["lambda_pre"] * torch.pow(torch.clamp(ratio, min=eps), params["m"])
                          * torch.pow(torch.clamp(1.0 - z_s, min=eps), params["nu"]) + dz_corr)
            else:
                dz_pre = 0.10 * torch.tanh(dz_corr) + 0.02 * phi
            dr_pre = (params["alpha_pre"] * phi * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_pre"])
                      + params["beta_pre"] / (cycles[:, step] + params["tau_pre"])
                      - params["gamma_pre"] * r_s + dr_corr)
            if self.structured_post_event:
                dz_post = params["lambda_post"] * torch.clamp(1.0 - z_s, min=eps) + 0.5 * dz_corr
                dr_post = (params["alpha_post"] * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_post"])
                           + params["beta_post"] / (cycles[:, step] + 0.70 * params["tau_pre"])
                           - params["gamma_post"] * r_s + 0.5 * dr_corr)
            else:
                dz_post, dr_post = dz_pre, dr_pre
            dz = (1.0 - g_s) * dz_pre + g_s * dz_post
            dr = (1.0 - g_s) * dr_pre + g_s * dr_post
            return dz, dr, dc_corr

        # Learned small initial pressure level.
        z_curr = torch.zeros(batch_size, device=cycles.device)
        r_curr = 0.10 * torch.sigmoid(self.r0_head(encoded)).squeeze(-1)
        c_curr = torch.zeros(batch_size, device=cycles.device)
        z_states: List[torch.Tensor] = [z_curr]
        r_states: List[torch.Tensor] = [r_curr]
        c_states: List[torch.Tensor] = [c_curr]
        g_states: List[torch.Tensor] = []

        for step in range(seq_len):
            g_step = trigger(z_curr)
            g_states.append(g_step)
            if step == seq_len - 1:
                break
            dn = dcyc[:, step + 1]
            dz1, dr1, dc1 = derivatives(z_curr, r_curr, c_curr, g_step, step)
            if self.integrator == "heun":
                # Heun RK2: predict, then correct with the average slope.
                z_e = torch.clamp(z_curr + dn * dz1, 0.0, 0.999)
                r_e = torch.clamp(r_curr + dn * dr1, 0.0, 1.0)
                c_e = torch.tanh(params["c_decay"] * c_curr + dc1)
                dz2, dr2, dc2 = derivatives(z_e, r_e, c_e, trigger(z_e), step)
                dz_step, dr_step, dc_step = 0.5 * (dz1 + dz2), 0.5 * (dr1 + dr2), 0.5 * (dc1 + dc2)
            else:
                dz_step, dr_step, dc_step = dz1, dr1, dc1
            z_curr = torch.clamp(z_curr + dn * dz_step, 0.0, 0.999)
            # Enforce irreversible damage by projecting z to a nondecreasing path;
            # neural corrections can otherwise decrease it in the tail,
            # which would also reduce the event trigger.
            z_curr = torch.maximum(z_curr, z_states[-1])
            r_curr = torch.clamp(r_curr + dn * dr_step, 0.0, 1.0)
            c_curr = torch.tanh(params["c_decay"] * c_curr + dc_step)
            z_states.append(z_curr)
            r_states.append(r_curr)
            c_states.append(c_curr)

        z = torch.stack(z_states, dim=1)
        r = torch.stack(r_states, dim=1)
        c = torch.stack(c_states, dim=1)
        g = torch.stack(g_states, dim=1)
        # Keep the absorbing-event trigger nondecreasing after projecting z.
        # cummax enforces this property on the reported trigger path.
        g = torch.cummax(g, dim=1).values
        # Project PPR to a nondecreasing path bounded by ru<=1.
        r = monotone_clip(r, hi=1.0)
        # Read event time from the pressure threshold crossing.
        mcr = torch.as_tensor(self.max_cycle_reference, device=r.device, dtype=r.dtype)
        if self.nliq_from_curve:
            nliq_curve, cross_pdf, cross_mass = self._nliq_from_curve(r, cycles)
        else:
            nliq_curve = self.soft_first_hitting(r, g, cycles)
            cross_pdf = torch.zeros_like(r); cross_mass = torch.zeros(r.shape[0], device=r.device)
        nliq_norm_curve = torch.log1p(nliq_curve) / torch.log1p(mcr)
        # Direct head is an auxiliary target.
        # Report N_liq from the pressure-curve crossing.
        nliq_head, nliq_norm_head, nliq_norm_curve = self._apply_nliq_head(encoded, nliq_norm_curve, mcr)
        if self.report_nliq_from_curve:
            nliq, nliq_norm = nliq_curve, nliq_norm_curve
        else:
            nliq, nliq_norm = nliq_head, nliq_norm_head
        summary = physics_summary(r, z, g, nliq_norm)
        # Context-based risk, gated structured prior and probability scaling.
        risk_prior = 6.0 * (0.50 * r.amax(dim=1) + 0.25 * g.amax(dim=1) + 0.25 * z.amax(dim=1) - 0.75)
        risk_logit = (self.risk_clf(encoded).squeeze(-1)
                      + self.prior_gate * risk_prior
                      + self.risk_head(encoded, summary))
        traj_logvar = self.logvar_head_seq(encoded) + 2.0 * self.calib_log_scale
        return {
            "traj_mean": r,
            "traj_logvar": traj_logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq": nliq,
            "nliq_norm": nliq_norm,
            "nliq_norm_curve": nliq_norm_curve,
            "nliq_head": nliq_head,
            "nliq_norm_head": nliq_norm_head,
            "z": z,
            "g": g,
            "c": c,
            "crr": crr,
            "cross_pdf": cross_pdf,
            "cross_mass": cross_mass,
        }

    def _apply_nliq_head(self, encoded: torch.Tensor, nliq_norm_curve: torch.Tensor,
                         mcr: torch.Tensor):
        """Apply the optional direct event-time head.

        Return (nliq, nliq_norm_reported, nliq_norm_curve). Without the head,
        the reported normalized time equals the curve-derived time.
        """
        if not getattr(self, "use_nliq_head", False):
            nliq = torch.expm1(nliq_norm_curve * torch.log1p(mcr))
            return nliq, nliq_norm_curve, nliq_norm_curve
        nn_head = torch.sigmoid(self.nliq_head(encoded)).squeeze(-1) # nliq_norm ∈ (0,1)
        nliq = torch.expm1(nn_head * torch.log1p(mcr))
        return nliq, nn_head, nliq_norm_curve

    def _nliq_consistency_loss(self, outputs: Dict[str, torch.Tensor],
                               batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Align the direct head with curve crossing on observed events.

        Apply nonzero weight when label==1, event time is observed and cross_mass>=0.5.
        Detach the curve target so gradients update the direct head.
        """
        if not getattr(self, "use_nliq_head", False) or "nliq_norm_curve" not in outputs:
            return torch.zeros((), device=outputs["nliq_norm"].device)
        obs = nliq_censor_mask(batch)
        obs = obs if obs is not None else torch.ones_like(batch["label"])
        crossed = ((outputs["cross_mass"] >= 0.5).to(outputs["nliq_norm"].dtype)
                   if "cross_mass" in outputs else torch.ones_like(batch["label"]))
        w = batch["label"] * obs * crossed
        target = outputs["nliq_norm_curve"].detach() # Stop gradients through the curve-derived target.
        # Curve-first models still train the direct head
        # with an auxiliary consistency term.
        head = outputs.get("nliq_norm_head", outputs["nliq_norm"])
        diff = (head - target) ** 2
        return (diff * w).sum() / w.sum().clamp_min(1.0)

    def _nliq_from_curve(self, r: torch.Tensor, cycles: torch.Tensor, beta: float = 25.0):
        """Derive N_liq and soft crossing mass from a monotone pressure curve.

        For nondecreasing r, sigmoid(beta*(r-threshold)) is nondecreasing. Differences
        give crossing mass by node. Return the interpolated time, pdf and total mass.

        :param r: monotone PPR, shape (B, T)
        :param cycles: query cycles, shape (B, T)
        :return: nliq (B,), pdf (B, T), mass (B,)
        """
        p = torch.sigmoid(beta * (r - self.liq_threshold))
        dp = torch.clamp(p[:, 1:] - p[:, :-1], min=0.0)
        pdf = torch.cat([p[:, :1], dp], dim=1) # Soft crossing mass for resistance-at-crossing and consistency losses.
        mass = pdf.sum(dim=1)
        # Use linear interpolation between threshold-bracketing nodes,
        # shared with DPI-EVT and the curve-coherence diagnostic.
        nliq = interpolated_crossing(r, cycles, self.liq_threshold, beta=beta)
        return nliq, pdf, mass

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate outputs and the total batch loss.

        Use measured PPR, event labels, event-time targets and observation masks.
        Additional terms regulate smoothness, state bounds, output consistency and
        optional auxiliary targets.

        :param batch: observed targets, inputs and masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        # Main trajectory loss is post-prefix;
        # auxiliary prefix reconstruction has weight 0.15.
        _cont_mask, _pref_mask = trajectory_supervision_masks(batch)
        traj_loss = gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], _cont_mask)
        traj_loss = traj_loss + 0.15 * gaussian_nll(
            outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], _pref_mask)
        # Risk and soft-AUC losses use observed outcomes only.
        _robs = risk_observation_mask(batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], _robs)
        _m = (_robs > 0.5) if _robs is not None else torch.ones_like(batch["label"], dtype=torch.bool)
        rank_loss = (soft_auc_loss(outputs["risk_logit"][_m], batch["label"][_m]) if bool(_m.any())
                     else outputs["risk_logit"].sum() * 0.0) # Pairwise ranking objective for AUROC.
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_norm"], batch["n_liq_norm"],
                                              batch["label"], nliq_censor_mask(batch))
        # Curve-first loss trains the reported curve, plus an auxiliary direct head;
        # head-first mode already applies the main timing loss to that head.
        if self.report_nliq_from_curve and getattr(self, "use_nliq_head", False):
            nliq_head_loss = masked_censored_nliq_loss(
                outputs["nliq_norm_head"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch),
            )
        else:
            nliq_head_loss = torch.zeros((), device=outputs["traj_mean"].device)
        switch_reg = torch.abs(outputs["g"][:, 1:] - outputs["g"][:, :-1]).mean()
        state_smoothness = (
            torch.abs(outputs["traj_mean"][:, 2:] - 2.0 * outputs["traj_mean"][:, 1:-1] + outputs["traj_mean"][:, :-2]).mean()
            + torch.abs(outputs["z"][:, 2:] - 2.0 * outputs["z"][:, 1:-1] + outputs["z"][:, :-2]).mean()
        )
        boundedness = (
            torch.relu(outputs["traj_mean"] - 1.0)  # ru<=1, matching monotone_clip(hi=1.0).
            + torch.relu(-outputs["traj_mean"])
            + torch.relu(outputs["z"] - 1.0)
            + torch.relu(-outputs["z"])
        ).mean()
        # Joint consistency at onset: CRR(N_liq)~=CSR and damage near its threshold.
        joint = torch.zeros((), device=outputs["traj_mean"].device)
        if self.use_joint_consistency and outputs.get("cross_mass") is not None and outputs["cross_mass"].sum() > 0:
            liq = batch["label"]; denom = torch.clamp(liq.sum(), min=1.0)
            m = torch.clamp(outputs["cross_mass"], min=1e-4)
            crr_at = (outputs["cross_pdf"] * outputs["crr"]).sum(1) / m
            z_at = (outputs["cross_pdf"] * outputs["z"]).sum(1) / m
            csr_app = batch["csr"].amax(dim=1)
            joint = ((liq * (crr_at - csr_app) ** 2).sum() + (liq * (z_at - 0.90) ** 2).sum()) / denom
        # Overshoot and non-event trigger penalties.
        # Penalize positive deviations from the observed pressure curve
        # so post-event dynamics do not create unsupported pressure increases
        # on records without observed liquefaction.
        # The targets are measured pressure and observed record status.
        # The one-sided overshoot term also applies to unfinished censored records,
        # penalizing only values above available measurements.
        noliq = (1.0 - batch["label"]).unsqueeze(1) # (B,1)
        overshoot = masked_mean(torch.relu(outputs["traj_mean"] - batch["r_obs"]) * noliq, batch["mask"])
        # Trigger suppression applies only to completed non-event records;
        # a valid censoring time alone does not establish this status.
        # An unfinished record can accumulate pressure and reach the threshold later,
        # so its trigger is not suppressed by this term.
        stable = batch.get("regime_stable")
        stab = noliq if stable is None else stable.unsqueeze(1)
        trigger_noliq = masked_mean(outputs["g"] * stab, batch["mask"])
        # Match the direct event-time head to observed curve crossings.
        nliq_consistency = self._nliq_consistency_loss(outputs, batch)
        # Link event probability with threshold crossing of the forecast curve,
        # penalizing disagreement between risk and the pressure path.
        p_cross = torch.sigmoid(12.0 * (outputs["traj_mean"].amax(dim=1) - self.liq_threshold))
        coherence = ((outputs["risk_prob"] - p_cross) ** 2).mean()
        loss = (
            traj_loss
            + 0.80 * risk_loss
            + 0.30 * rank_loss
            + 0.25 * nliq_loss
            + self.nliq_head_aux_weight * nliq_head_loss
            + 0.10 * nliq_consistency
            + 0.02 * switch_reg
            + 0.01 * state_smoothness
            + 0.03 * boundedness
            + 0.05 * joint
            + 0.20 * overshoot
            + 0.05 * trigger_noliq
            + 0.05 * coherence
        )
        if self.use_observed_aux_loss:
            loss = loss + observed_aux_loss(outputs, batch, use_states=True)
        outputs["loss"] = loss
        return outputs
