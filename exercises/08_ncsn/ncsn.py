"""Module 08 - Noise-Conditional Score Networks (book §3.4-3.5, pp. 91-97).

One network s_theta(x, sigma) for a geometric ladder sigma_1 > ... > sigma_L.
The network is called as score_model(x, sigma) with sigma a (B,) float tensor.
"""

import math

import torch


def geometric_sigmas(sigma_max: float, sigma_min: float, L: int) -> torch.Tensor:
    """L noise levels, geometrically spaced from sigma_max down to sigma_min (descending)."""
    raise NotImplementedError("TODO")


def ncsn_loss(score_model, x: torch.Tensor, sigmas: torch.Tensor, idx: torch.Tensor | None = None,
              noise: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-level DSM with lambda(sigma) = sigma^2 weighting (book Eq. in §3.4):

        L = E_i E_x E_eps [ sigma_i^2 * 1/2 || s(x + sigma_i eps, sigma_i) + eps / sigma_i ||^2 ]
          = E [ 1/2 || sigma_i s(x~, sigma_i) + eps ||^2 ]

    Draw a level index per sample if idx is None. Why sigma^2? It makes every level's
    loss O(1), so no noise level dominates.
    """
    raise NotImplementedError("TODO")


@torch.no_grad()
def annealed_langevin(score_model, x: torch.Tensor, sigmas: torch.Tensor, eps: float = 2e-5,
                      n_steps_each: int = 100, denoise_last: bool = True,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Annealed Langevin dynamics (Song & Ermon 2019; book Algorithm in §3.4):

        for each sigma_i (largest first):
            alpha_i = eps * sigma_i^2 / sigma_L^2
            repeat n_steps_each:  x <- x + alpha_i s(x, sigma_i) + sqrt(2 alpha_i) z

    If denoise_last, finish with one Tweedie step at sigma_L: x <- x + sigma_L^2 s(x, sigma_L).
    """
    raise NotImplementedError("TODO")


def ddpm_eps_to_score(eps: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    """The DDPM <-> NCSN dictionary: since x_t = sqrt(ab) x0 + sqrt(1-ab) eps,
    grad log p_t(x_t) = -E[eps | x_t] / sqrt(1 - ab).  alpha_bar_t broadcasts against eps.
    """
    raise NotImplementedError("TODO")
