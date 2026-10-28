"""Module 05 - DDPM (book §2.2, pp. 54-67).

Index convention (same as most code bases, slightly different from the book):
    t in {0, ..., T-1};  betas[t] is the noise added at step t;
    alpha[t] = 1 - betas[t];  alpha_bar[t] = prod_{s <= t} alpha[s];
    "x_t" in code = the latent after t+1 noising steps, so q(x_t | x_0) = N(sqrt(ab[t]) x0, (1 - ab[t]) I).
    For the posterior at t = 0 use alpha_bar_prev[0] = 1: it collapses to a point mass at x0.

The network is called as model(x_t, t) with t a LongTensor of shape (B,)
(use TimeMLP(t_scale=1.0) or SmallUNet(t_scale=1.0) so the embedding sees integers).
`t` and `alpha_bar` lookups must broadcast against x of any shape (B, ...).
"""

import math

import torch


def _bcast(v: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gather v[t] and reshape to (B, 1, 1, ...) so it broadcasts against x."""
    return v.to(x.device)[t].reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Ho et al. (2020): betas linearly spaced in [beta_start, beta_end]. Returns (T,) float64."""
    raise NotImplementedError("TODO")


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal (2021). Define  f(u) = cos^2( (u/T + s) / (1 + s) * pi/2 ),
    alpha_bar(u) = f(u) / f(0) for u = 0..T, then betas[t] = 1 - alpha_bar(t+1) / alpha_bar(t),
    clipped to max_beta. Returns (T,) float64.
    """
    raise NotImplementedError("TODO")


def alpha_bar_from_betas(betas: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_0) in one shot (the closed form from module 04)."""
    raise NotImplementedError("TODO")


def q_posterior(x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, betas: torch.Tensor):
    """Mean and variance of q(x_{t-1} | x_t, x_0) (book Eq. in §2.2.2).

        mean = sqrt(ab_prev) beta_t / (1 - ab_t) * x0 + sqrt(alpha_t) (1 - ab_prev) / (1 - ab_t) * x_t
        var  = beta_t (1 - ab_prev) / (1 - ab_t)

    Returns (mean shaped like x0, var broadcastable to x0).
    """
    raise NotImplementedError("TODO")


def predict_x0_from_eps(xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Invert q_sample: x0 = (x_t - sqrt(1 - ab) eps) / sqrt(ab)."""
    raise NotImplementedError("TODO")


def ddpm_loss(model, x0: torch.Tensor, alpha_bar: torch.Tensor, t: torch.Tensor | None = None,
              noise: torch.Tensor | None = None) -> torch.Tensor:
    """The "simple" loss  E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2  (mean over everything).

    If t / noise are None, draw t ~ Uniform{0..T-1} and noise ~ N(0, I).
    """
    raise NotImplementedError("TODO")


def p_mean_var(model, xt: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, var_type: str = "posterior"):
    """Reverse step p_theta(x_{t-1} | x_t) = N(mean, var):
    mean = q_posterior mean with x0 replaced by the eps-model's prediction (equivalently
    1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - ab_t) * eps_theta)).
    var = posterior variance ("posterior") or beta_t ("beta").
    """
    raise NotImplementedError("TODO")


@torch.no_grad()
def ddpm_sample(model, shape, betas: torch.Tensor, var_type: str = "posterior",
                generator: torch.Generator | None = None, device="cpu", return_trajectory: bool = False):
    """Ancestral sampling: x_{T-1} ~ N(0, I), then x_{t-1} ~ p_theta(x_{t-1} | x_t) down to t = 0.
    No noise is added at the final step. Returns x (and the list of all x_t if return_trajectory).
    """
    raise NotImplementedError("TODO")


def vlb_term(model, x0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """L_{t-1} = KL( q(x_{t-1} | x_t, x0) || p_theta(x_{t-1} | x_t) ) per sample (B,), for t >= 1,
    with x_t = q_sample(x0, t, ..., noise) and var_type="posterior".

    The test checks that this equals  beta_t^2 / (2 var_t alpha_t (1 - ab_t)) * ||eps - eps_theta||^2,
    i.e. the simple loss is a *re-weighted* ELBO. Derive that identity yourself first.
    """
    raise NotImplementedError("TODO")
