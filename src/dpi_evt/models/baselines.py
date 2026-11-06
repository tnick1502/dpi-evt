"""Black-box comparators implementing forward_batch and compute_loss.

Includes a static risk/event-time MLP and GRU, LSTM, Transformer and causal
temporal-convolution models with pressure-distribution heads.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpi_evt.models.blocks import CausalTemporalBlock, ResidualMLP
from dpi_evt.training.losses import (forecast_gaussian_nll, masked_bce_with_logits,
                                             masked_censored_nliq_loss, masked_mean,
                                             nliq_censor_mask, risk_observation_mask)

__all__ = ["RiskMLP", "GRUBaseline", "TCNBaseline", "LSTMBaseline", "TransformerBaseline"]


class RiskMLP(nn.Module):
    """Static MLP for event probability and N_liq.

    Use soil/loading descriptors and prefix summaries to predict risk logits,
    normalized event time and an uncertainty proxy.
    """

    def __init__(self, static_dim: int, prefix_dim: int, hidden_dim: int = 128):
        """:param static_dim: static-feature count
        :param prefix_dim: prefix-summary width
        :param hidden_dim: hidden representation width
        """
        super().__init__()
        self.backbone = ResidualMLP(static_dim + prefix_dim, hidden_dim=hidden_dim, depth=3)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate a batch of static and prefix-summary inputs.

        :param batch: input dictionary with static and prefix_summary
        :return: risk_logit, risk_prob, nliq_pred and uncertainty
        """
        x = torch.cat([batch["static"], batch["prefix_summary"]], dim=-1)
        h = self.backbone(x)
        risk_logit = self.risk_head(h).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(h).squeeze(-1))
        uncertainty = F.softplus(self.uncertainty_head(h).squeeze(-1)) + 1e-3
        return {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
            "uncertainty": uncertainty,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate risk BCE and event-time loss from observed targets.

        :param batch: inputs, label, n_liq_norm and observation masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        loss = risk_loss + 0.45 * nliq_loss
        outputs["loss"] = loss
        return outputs


class GRUBaseline(nn.Module):
    """GRU sequence comparator with a probabilistic PPR head.

    Concatenate projected static features with sequence inputs at every node.
    A two-layer GRU predicts pressure means/log variances; the final hidden state
    supplies event probability and N_liq.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        """:param static_dim: static-feature count
        :param seq_dim: sequence channels per node
        :param hidden_dim: GRU hidden-state width
        """
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRU(input_size=seq_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2, dropout=0.10)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate static and sequence inputs.

        :param batch: input dictionary with static and seq_in
        :return: traj_mean, traj_logvar, risk_logit, risk_prob and nliq_pred
        """
        static_embed = self.static_proj(batch["static"]).unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        x = torch.cat([batch["seq_in"], static_embed], dim=-1)
        h, _ = self.gru(x)
        mean = torch.sigmoid(self.mean_head(h).squeeze(-1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        pooled = h[:, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine Gaussian pressure NLL, risk BCE, event-time and smoothness losses.

        :param batch: targets, inputs and masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        traj_loss = forecast_gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs


class LSTMBaseline(nn.Module):
    """LSTM sequence comparator with a probabilistic PPR head.

    Concatenate projected static features with sequence inputs. A two-layer LSTM
    predicts nodewise pressure means/log variances and uses its final hidden state
    for event probability and N_liq.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        """:param static_dim: static-feature count
        :param seq_dim: sequence channels per node
        :param hidden_dim: LSTM hidden-state width
        """
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.lstm = nn.LSTM(input_size=seq_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2, dropout=0.10)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate static and sequence inputs.

        :param batch: input dictionary with static and seq_in
        :return: traj_mean, traj_logvar, risk_logit, risk_prob and nliq_pred
        """
        static_embed = self.static_proj(batch["static"]).unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        x = torch.cat([batch["seq_in"], static_embed], dim=-1)
        h, _ = self.lstm(x)
        mean = torch.sigmoid(self.mean_head(h).squeeze(-1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        pooled = h[:, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine Gaussian pressure NLL, risk BCE, event-time and smoothness losses.

        :param batch: targets, inputs and masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        traj_loss = forecast_gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs


class TransformerBaseline(nn.Module):
    """Causal Transformer encoder with a probabilistic PPR head.

    Combine projected sequence inputs, learned positions and static features.
    Attention at t sees nodes <=t. Predict pressure means/log variances at each node
    and event probability/N_liq from the final node.
    """

    def __init__(self, static_dim: int, seq_dim: int, seq_len: int, hidden_dim: int = 96,
                 n_heads: int = 4, n_layers: int = 2):
        """:param static_dim: static-feature count
        :param seq_dim: sequence channels per node
        :param seq_len: sequence length for positional embeddings
        :param hidden_dim: model width d_model
        :param n_heads: attention-head count
        :param n_layers: encoder-layer count
        """
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(seq_dim, hidden_dim)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
                                           dropout=0.10, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate a batch with causal attention.

        :param batch: input dictionary with static and seq_in
        :return: traj_mean, traj_logvar, risk_logit, risk_prob and nliq_pred
        """
        seq = batch["seq_in"]
        T = seq.shape[1]
        static_embed = self.static_proj(batch["static"]).unsqueeze(1)
        h = self.input_proj(seq) + self.pos_embed[:, :T] + static_embed
        mask = torch.triu(torch.ones(T, T, device=seq.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        mean = torch.sigmoid(self.mean_head(h).squeeze(-1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=2.0)
        pooled = h[:, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine Gaussian pressure NLL, risk BCE, event-time and smoothness losses.

        :param batch: targets, inputs and masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        traj_loss = forecast_gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs


class TCNBaseline(nn.Module):
    """Causal temporal-convolution comparator with a probabilistic PPR head.

    Dilation factors 1, 2 and 4 extend the receptive field. Predict nodewise pressure
    means/log variances and event probability/N_liq from the last time step.
    """

    def __init__(self, static_dim: int, seq_dim: int, hidden_dim: int = 96):
        """:param static_dim: static-feature count
        :param seq_dim: sequence channels per node
        :param hidden_dim: convolution-channel count
        """
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        in_channels = seq_dim + hidden_dim
        self.blocks = nn.Sequential(
            CausalTemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1),
            CausalTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            CausalTemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
        )
        self.mean_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.logvar_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.nliq_head = nn.Linear(hidden_dim, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Evaluate static and sequence inputs.

        :param batch: input dictionary with static and seq_in
        :return: traj_mean, traj_logvar, risk_logit, risk_prob and nliq_pred
        """
        static_embed = self.static_proj(batch["static"]).unsqueeze(1).expand(-1, batch["seq_in"].shape[1], -1)
        x = torch.cat([batch["seq_in"], static_embed], dim=-1).transpose(1, 2)
        h = self.blocks(x)
        mean = torch.sigmoid(self.mean_head(h).squeeze(1))
        logvar = torch.clamp(self.logvar_head(h).squeeze(1), min=-6.0, max=2.0)
        pooled = h[:, :, -1]
        risk_logit = self.risk_head(pooled).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(pooled).squeeze(-1))
        return {
            "traj_mean": mean,
            "traj_logvar": logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_pred": nliq_pred,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine Gaussian pressure NLL, risk BCE, event-time and smoothness losses.

        :param batch: targets, inputs and masks
        :return: model outputs with loss
        """
        outputs = self.forward_batch(batch)
        traj_loss = forecast_gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        smoothness = masked_mean(
            torch.abs(outputs["traj_mean"][:, 1:] - outputs["traj_mean"][:, :-1]), batch["mask"][:, 1:]
        )
        loss = traj_loss + 0.35 * risk_loss + 0.25 * nliq_loss + 0.02 * smoothness
        outputs["loss"] = loss
        return outputs
