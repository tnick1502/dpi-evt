"""Output heads for structured models.

The risk head combines context with pressure, damage, trigger and N_liq summaries.
Separate heads predict nodewise heteroscedastic pressure uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["physics_summary", "RiskHead", "SeqLogvarHead"]

PHYSICS_SUMMARY_DIM = 8
"""State-summary width; see physics_summary."""


def physics_summary(r: torch.Tensor, z: torch.Tensor, g: torch.Tensor, nliq_norm: torch.Tensor) -> torch.Tensor:
    """Summarize the modeled state trajectories for the risk head.

    Use peak/final/mean PPR, peak/final damage z, peak/mean trigger g, and normalized
    N_liq.

    :param r: pressure path, shape (batch, seq_len)
    :param z: damage path, shape (batch, seq_len)
    :param g: trigger path, shape (batch, seq_len)
    :param nliq_norm: normalized event time, shape (batch,)
    :return: summary tensor, shape (batch, 8)
    """
    return torch.stack(
        [
            r.amax(dim=1), r[:, -1], r.mean(dim=1),
            z.amax(dim=1), z[:, -1],
            g.amax(dim=1), g.mean(dim=1),
            nliq_norm,
        ],
        dim=-1,
    )


class RiskHead(nn.Module):
    """Learn a risk-logit correction from encoded context and state summaries."""

    def __init__(self, context_dim: int, summary_dim: int = PHYSICS_SUMMARY_DIM, hidden_dim: int = 64):
        """:param context_dim: encoded-context width
        :param summary_dim: state-summary width
        :param hidden_dim: head hidden width
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Zero-initialized output starts as a near-zero correction to the prior;
        # training adjusts the residual risk logit.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """Predict a residual risk-logit correction, initialized near zero.

        :param context: shape (batch, context_dim)
        :param summary: shape (batch, summary_dim)
        :return: residual logit, shape (batch,)
        """
        return self.net(torch.cat([context, summary], dim=-1)).squeeze(-1)


class SeqLogvarHead(nn.Module):
    """Predict a heteroscedastic log variance at each PPR node from the context."""

    def __init__(self, context_dim: int, seq_len: int, hidden_dim: int = 64,
                 logvar_min: float = -8.0, logvar_max: float = -1.0, init_logvar: float = -4.0):
        """:param context_dim: encoded-context width
        :param seq_len: sequence length
        :param hidden_dim: hidden width
        :param logvar_min: lower log-variance bound
        :param logvar_max: upper log-variance bound
        :param init_logvar: initial constant log variance
        """
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len),
        )
        # Start near constant init_logvar and learn heteroscedastic corrections.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_logvar)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict bounded nodewise log variance.

        :param context: shape (batch, context_dim)
        :return: clipped log variance, shape (batch, seq_len)
        """
        return torch.clamp(self.net(context), min=self.logvar_min, max=self.logvar_max)
