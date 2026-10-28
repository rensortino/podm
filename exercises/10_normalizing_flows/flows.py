"""Module 10 - Normalizing flows (book §5.1, pp. 141-149).

Convention: `forward` maps data x -> latent z (the density-evaluation direction) and
returns log|det dz/dx|; `inverse` maps z -> x (the sampling direction).
    log p(x) = log N(z; 0, I) + sum_layers log|det dz/dx|
"""

import math

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    """RealNVP coupling layer. With a binary mask m (1 = passed through unchanged):

        z = m * x + (1 - m) * ( x * exp(s(m * x)) + t(m * x) )

    s is passed through tanh for stability. The Jacobian is triangular, so
    log|det| = sum over the transformed dims of s.
    """

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * dim))

    def _s_t(self, x_masked):
        s, t = self.net(x_masked).chunk(2, dim=-1)
        return torch.tanh(s), t

    def forward(self, x):
        """x -> (z, log|det dz/dx|) with log-det shape (B,)."""
        raise NotImplementedError("TODO")

    def inverse(self, z):
        """z -> x. Note s, t only depend on the unchanged half, which z and x share."""
        raise NotImplementedError("TODO")


class RealNVP(nn.Module):
    def __init__(self, dim: int = 2, n_layers: int = 8, hidden: int = 128):
        super().__init__()
        masks = [torch.arange(dim) % 2 == (i % 2) for i in range(n_layers)]
        self.layers = nn.ModuleList([AffineCoupling(dim, m, hidden) for m in masks])
        self.dim = dim

    def forward(self, x):
        """x -> (z, total log-det)."""
        raise NotImplementedError("TODO")

    def log_prob(self, x):
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None):
        raise NotImplementedError("TODO")
