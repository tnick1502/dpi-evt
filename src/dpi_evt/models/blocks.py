"""Shared residual MLP and causal dilated temporal-convolution blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResidualMLP", "CausalTemporalBlock"]


class ResidualMLP(nn.Module):
    """Residual MLP with LayerNorm, GELU and dropout.

    Project inputs to hidden_dim, then apply depth residual blocks of
    Linear -> LayerNorm -> GELU -> Dropout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.10):
        """:param input_dim: input-feature count
        :param hidden_dim: hidden width
        :param depth: residual-block count
        :param dropout: dropout probability within each block
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor.

        :param x: shape (batch, input_dim)
        :return: hidden representation, shape (batch, hidden_dim)
        """
        h = F.gelu(self.input_layer(x))
        for block in self.blocks:
            h = h + block(h)
        return h


class CausalTemporalBlock(nn.Module):
    """Causal dilated temporal-convolution block.

    Two convolutions trim right padding so the output at t depends only on nodes <=t.
    The residual branch uses a 1x1 projection when channel counts differ.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.10):
        """:param in_channels: input-channel count
        :param out_channels: output-channel count
        :param kernel_size: convolution-kernel width
        :param dilation: convolution dilation
        :param dropout: dropout probability between convolutions
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Trim right padding to retain causality.

        :param x: shape (batch, channels, time + padding)
        :return: shape (batch, channels, time)
        """
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual temporal block.

        :param x: shape (batch, in_channels, time)
        :return: shape (batch, out_channels, time)
        """
        out = self.chomp(self.conv1(x))
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.chomp(self.conv2(out))
        out = F.gelu(out)
        out = self.dropout(out)
        residual = self.downsample(x)
        if self.padding > 0:
            residual = residual[:, :, : out.shape[2]]
        return F.gelu(out + residual)
