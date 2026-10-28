import math

import torch
from torch.distributions import Normal

from diffeq import (divergence, euler_maruyama, euler_ode, heun_ode, ode_with_log_density,
                    pushforward_log_prob, rk4_ode)

f_decay = lambda x, t: -x  # noqa: E731  exact solution x0 * exp(-(t1 - t0))


def _order(solver):
    x0 = torch.ones(1, dtype=torch.float64)
    exact = math.exp(-1.0)
    errs = [abs(solver(f_decay, x0, 0.0, 1.0, n).item() - exact) for n in (10, 20)]
    return math.log2(errs[0] / errs[1])


def test_euler_first_order():
    assert 0.8 < _order(euler_ode) < 1.2


def test_heun_second_order():
    assert 1.8 < _order(heun_ode) < 2.2


def test_rk4_fourth_order():
    assert 3.7 < _order(rk4_ode) < 4.3


def test_solvers_run_backwards_in_time():
    x1 = torch.ones(1, dtype=torch.float64)
    x0 = rk4_ode(f_decay, x1, 1.0, 0.0, 50)
    assert abs(x0.item() - math.e) < 1e-6


def test_euler_maruyama_ou_marginal():
    theta, sigma, c, T = 1.5, 0.8, 2.0, 1.0
    g = torch.Generator().manual_seed(0)
    x = euler_maruyama(lambda x, t: -theta * x, lambda t: sigma, torch.full((50_000, 1), c, dtype=torch.float64),
                       0.0, T, 500, generator=g)
    mean = c * math.exp(-theta * T)
    var = sigma**2 / (2 * theta) * (1 - math.exp(-2 * theta * T))
    assert abs(x.mean().item() - mean) < 0.01
    assert abs(x.var().item() - var) < 0.01


def test_pushforward_log_prob_lognormal():
    y = torch.rand(100, 1, dtype=torch.float64) * 3 + 0.1
    out = pushforward_log_prob(y, torch.log, lambda x: Normal(0.0, 1.0).log_prob(x).sum(-1))
    ref = torch.distributions.LogNormal(0.0, 1.0).log_prob(y)[:, 0]
    torch.testing.assert_close(out, ref)


def test_pushforward_log_prob_linear_2d():
    A = torch.tensor([[2.0, 1.0], [0.0, 0.5]], dtype=torch.float64)
    Ainv = torch.linalg.inv(A)
    y = torch.randn(10, 2, dtype=torch.float64)
    out = pushforward_log_prob(y, lambda v: v @ Ainv.T, lambda x: Normal(0.0, 1.0).log_prob(x).sum(-1))
    ref = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), A @ A.T).log_prob(y)
    torch.testing.assert_close(out, ref)


def test_divergence():
    x = torch.randn(7, 3, dtype=torch.float64)
    f = lambda x, t: torch.stack([x[:, 0] ** 2, x[:, 0] * x[:, 1], torch.sin(x[:, 2])], 1)  # noqa: E731
    ref = 2 * x[:, 0] + x[:, 0] + torch.cos(x[:, 2])
    torch.testing.assert_close(divergence(f, x, 0.0), ref)


def test_ode_with_log_density_linear_contraction():
    a, D, T = 0.7, 2, 1.0
    x0 = torch.randn(5, D, dtype=torch.float64)
    x1, delta = ode_with_log_density(lambda x, t: -a * x, x0, 0.0, T, 200)
    # contraction concentrates mass: log-density grows by a * D * T
    torch.testing.assert_close(delta, torch.full((5,), a * D * T, dtype=torch.float64))
    assert torch.allclose(x1, x0 * math.exp(-a * T), atol=5e-3)
