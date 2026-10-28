import math

import torch

from fast_solvers import (alpha_sigma, ddim_step, dpm_solver1_step, lam, sample_ddim, sample_dpm_solver2,
                          sample_dpm_solver_pp_2m, t_from_lambda, timesteps)

D = torch.float64
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5


def _as(t):
    a, s = alpha_sigma(t)
    return a.reshape(-1, 1), s.reshape(-1, 1)


def eps_oracle(x, t):
    a, s = _as(t)
    return s * (x - a * MU) / (a**2 * S**2 + s**2)


def x0_oracle(x, t):
    a, s = _as(t)
    return (x - s * eps_oracle(x, t)) / a


def exact_flow(x, t, s):
    """Exact PF-ODE map for Gaussian data: x_s = a_s MU + sqrt(V_s / V_t) (x_t - a_t MU)."""
    a_t, s_t = alpha_sigma(t)
    a_s, s_s = alpha_sigma(s)
    Vt, Vs = a_t**2 * S**2 + s_t**2, a_s**2 * S**2 + s_s**2
    return a_s * MU + torch.sqrt(Vs / Vt) * (x - a_t * MU)


def test_lambda_helpers_consistent():
    t = torch.tensor([0.01, 0.3, 0.9], dtype=D)
    torch.testing.assert_close(t_from_lambda(lam(t)), t)


def test_timesteps():
    for kind in ("uniform_t", "uniform_lambda"):
        ts = timesteps(10, kind=kind)
        assert ts.shape == (11,) and abs(ts[0] - 1) < 1e-9 and abs(ts[-1] - 1e-3) < 1e-9
        assert (ts[1:] < ts[:-1]).all()
    lams = lam(timesteps(10))
    torch.testing.assert_close(lams[1:] - lams[:-1], (lams[1] - lams[0]).expand(10))


def test_ddim_exact_for_point_mass_data():
    mu = torch.tensor([0.3, -1.2], dtype=D)
    t, s = 0.8, 0.05
    a_t, s_t = alpha_sigma(t)
    a_s, s_s = alpha_sigma(s)
    eps = torch.randn(10, 2, dtype=D)
    x_t = a_t * mu + s_t * eps
    torch.testing.assert_close(ddim_step(x_t, eps, t, s), a_s * mu + s_s * eps)


def test_dpm_solver1_equals_ddim():
    x, e = torch.randn(10, 2, dtype=D), torch.randn(10, 2, dtype=D)
    torch.testing.assert_close(dpm_solver1_step(x, e, 0.7, 0.3), ddim_step(x, e, 0.7, 0.3, eta=0.0))


def test_ddim_eta1_marginals():
    g = torch.Generator().manual_seed(0)
    x = sample_ddim(eps_oracle, torch.randn(20_000, 2, generator=g, dtype=D), timesteps(200, kind="uniform_t"),
                    eta=1.0, generator=g)
    assert torch.allclose(x.mean(0), MU, atol=0.03) and torch.allclose(x.std(0), torch.full((2,), S, dtype=D), atol=0.03)


def _err(sampler, model, n):
    torch.manual_seed(0)
    xT = torch.randn(2000, 2, dtype=D)
    ts = timesteps(n)
    ref = exact_flow(xT, ts[0], ts[-1])
    return (sampler(model, xT, ts) - ref).norm(dim=1).mean().item()


def test_convergence_orders():
    ddim = [_err(sample_ddim, eps_oracle, n) for n in (20, 40)]
    dpm2 = [_err(sample_dpm_solver2, eps_oracle, n) for n in (20, 40)]
    pp2m = [_err(sample_dpm_solver_pp_2m, x0_oracle, n) for n in (20, 40)]
    order = lambda e: math.log2(e[0] / e[1])  # noqa: E731
    assert 0.7 < order(ddim) < 1.4, ddim
    assert order(dpm2) > 1.6, dpm2
    assert order(pp2m) > 1.6, pp2m
    assert dpm2[1] < ddim[1] and pp2m[1] < ddim[1]
