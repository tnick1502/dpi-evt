"""Optional normalizing-flow latent for DPI-EVT.

The study uses use_flow=False.
"""
from typing import Tuple
import torch
import torch.nn as nn

class ConditionalCouplingFlow(nn.Module):
    """Conditional RealNVP with latent-dependent coupling and Jacobian log determinant.

    Each layer holds one masked latent partition fixed and uses it with context to
    predict scale s and shift t for the active partition. Sum active scales across
    layers to obtain log|det dtheta/dz|. Zero-initialized final MLP layers start the
    transform at identity, s=t=0.
    """

    def __init__(self, latent_dim: int, context_dim: int, n_layers: int = 4,
                 hidden: int = 128, scale_clamp: float = 2.0):
        """:param latent_dim: latent coefficient dimension
        :param context_dim: context width
        :param n_layers: alternating-mask coupling-layer count
        :param hidden: coupling-MLP width
        :param scale_clamp: scale bound implemented with scale_clamp*tanh
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.scale_clamp = scale_clamp
        masks = []
        for i in range(n_layers):
            m = torch.zeros(latent_dim)
            m[i % 2::2] = 1.0 # Alternate the latent partition held fixed.
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks))
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + context_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * latent_dim),
            )
            for _ in range(n_layers)
        ])
        for net in self.nets: # Zero-initialize the final layer to start at identity.
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform z to theta and accumulate the Jacobian log determinant.

        :param z: base latent, shape (batch, latent_dim)
        :param context: shape (batch, context_dim)
        :return: theta (batch, latent_dim) and log absolute determinant (batch,)
        """
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        h = z
        for net, mask in zip(self.nets, self.masks.to(z.dtype)):
            h_frozen = h * mask
            s, t = net(torch.cat([h_frozen, context], dim=-1)).chunk(2, dim=-1)
            active = 1.0 - mask
            s = self.scale_clamp * torch.tanh(s) * active # Scale only the active partition.
            t = t * active
            h = h_frozen + active * (h * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=-1)
        return h, log_det

def flow_kl_per_dim(latent: torch.Tensor, mu: torch.Tensor, raw_logvar: torch.Tensor,
                    theta: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """Estimate per-dimension KL(q(theta) || N(0, I)) for theta=f(z).

    For z~N(mu, sigma^2), log q(theta)=log N(z;mu,sigma^2)-log_det.
    One-sample KL is log q(theta)-log N(theta;0,I). The log(2*pi) constants cancel;
    divide by the latent dimension to retain the per-dimension loss scale.

    :param latent: base sample z=mu+sigma*epsilon, shape (batch, dim)
    :param mu: base Gaussian mean, shape (batch, dim)
    :param raw_logvar: base log variance, shape (batch, dim)
    :param theta: transformed sample, shape (batch, dim)
    :param log_det: Jacobian log determinant, shape (batch,)
    :return: specimen-level per-dimension KL estimate, shape (batch,)
    """
    var = torch.exp(raw_logvar)
    log_q0 = -0.5 * (((latent - mu) ** 2) / var + raw_logvar).sum(dim=-1) # log N(z;mu,sigma^2), omitting the constant.
    log_prior = -0.5 * (theta ** 2).sum(dim=-1) # log N(theta;0,I), omitting the constant.
    kl = (log_q0 - log_det - log_prior) / float(latent.shape[-1])
    return kl
