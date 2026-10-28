"""Module 02 - ODE/SDE crash course and density evolution
(book Appendix A pp. 436-452, Appendix B pp. 453-466).

Sampling from every diffusion/flow model in this course *is* solving an ODE or
SDE, and likelihoods come from how densities change along the flow.

Conventions: `f(x, t)` returns a tensor shaped like x; `t` is a Python float.
Time may run backwards (t1 < t0) - your solvers must handle that.
"""

import torch


def euler_ode(f, x0: torch.Tensor, t0: float, t1: float, n_steps: int) -> torch.Tensor:
    """Explicit Euler: x_{k+1} = x_k + h f(x_k, t_k),  h = (t1 - t0) / n_steps. Returns x(t1)."""
    raise NotImplementedError("TODO")


def heun_ode(f, x0: torch.Tensor, t0: float, t1: float, n_steps: int) -> torch.Tensor:
    """Heun (explicit trapezoid, 2nd order): predictor Euler step, then average the two slopes."""
    raise NotImplementedError("TODO")


def rk4_ode(f, x0: torch.Tensor, t0: float, t1: float, n_steps: int) -> torch.Tensor:
    """Classic 4th-order Runge-Kutta."""
    raise NotImplementedError("TODO")


def euler_maruyama(f, g, x0: torch.Tensor, t0: float, t1: float, n_steps: int,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler-Maruyama for dx = f(x, t) dt + g(t) dW (scalar diffusion coefficient g).

    x_{k+1} = x_k + f(x_k, t_k) h + g(t_k) sqrt(|h|) z,  z ~ N(0, I).
    Use torch.randn(..., generator=generator) for the noise.
    """
    raise NotImplementedError("TODO")


def pushforward_log_prob(y: torch.Tensor, inverse_fn, base_log_prob) -> torch.Tensor:
    """Change of variables for an invertible map y = T(x) (book App. B.1, p. 455).

    log p_Y(y) = log p_X(T^{-1}(y)) + log |det d T^{-1}(y) / dy|

    Args:
        y: (N, D) points.
        inverse_fn: T^{-1}, maps (N, D) -> (N, D), acts row-wise.
        base_log_prob: (N, D) -> (N,).
    Hint: torch.func.jacrev + torch.func.vmap give per-sample Jacobians;
    torch.linalg.slogdet gives log|det|.
    """
    raise NotImplementedError("TODO")


def divergence(f, x: torch.Tensor, t: float) -> torch.Tensor:
    """Exact divergence  sum_i d f_i / d x_i  for each row of x (N, D) -> (N,).

    Fine for small D (one backward pass per dimension). Later modules swap this
    for Hutchinson's estimator  E_v[v^T (df/dx) v]  in high dimension.
    """
    raise NotImplementedError("TODO")


def ode_with_log_density(f, x0: torch.Tensor, t0: float, t1: float, n_steps: int):
    """Integrate dx/dt = f(x, t) together with the instantaneous change of variables

        d/dt log p_t(x_t) = - div f(x_t, t)       (continuity equation, App. B.2, p. 463)

    using Euler steps. Returns (x(t1), delta) where delta = log p_{t1}(x(t1)) - log p_{t0}(x0).
    This is exactly how continuous normalising flows and the diffusion PF-ODE compute likelihoods.
    """
    raise NotImplementedError("TODO")
