import math

import torch
from torch.distributions import MultivariateNormal

from score_sde import VESDE, VPSDE, dsm_loss_sde, pf_ode_log_likelihood, pf_ode_sample, reverse_sde_sample

D = torch.float64
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5


def oracle_score(sde):
    def score(x, t):
        m, std = sde.marginal(t)
        m, std = m.reshape(-1, 1).to(x.dtype), std.reshape(-1, 1).to(x.dtype)
        return -(x - m * MU) / (m**2 * S**2 + std**2)

    return score


def _simulate_forward(sde, x0, t_end, n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, h = x0.clone(), t_end / n
    for i in range(n):
        t = torch.full((x.shape[0],), i * h, dtype=D)
        x = x + sde.drift(x, t) * h + sde.diffusion(t).reshape(-1, 1) * math.sqrt(h) * torch.randn(
            x.shape, generator=g, dtype=D)
    return x


def test_vp_marginal_matches_simulation():
    sde = VPSDE()
    x = _simulate_forward(sde, torch.full((50_000, 1), 2.0, dtype=D), 0.4)
    m, std = sde.marginal(torch.tensor([0.4], dtype=D))
    assert abs(x.mean().item() - 2 * m.item()) < 0.02
    assert abs(x.std().item() - std.item()) < 0.02


def test_ve_marginal_matches_simulation():
    sde = VESDE(0.01, 5.0)
    x = _simulate_forward(sde, torch.full((50_000, 1), 2.0, dtype=D), 0.7)
    m, std = sde.marginal(torch.tensor([0.7], dtype=D))
    assert abs(x.mean().item() - 2.0) < 0.03 and abs(m.item() - 1) < 1e-12
    assert abs(x.std().item() / std.item() - 1) < 0.02


def test_dsm_loss_oracle_is_minimum():
    torch.manual_seed(0)
    sde = VPSDE()
    x0 = MU + S * torch.randn(50_000, 2, dtype=D)
    t, noise = 1e-3 + torch.rand(50_000, dtype=D) * (1 - 1e-3), torch.randn(50_000, 2, dtype=D)
    best = dsm_loss_sde(oracle_score(sde), x0, sde, t=t, noise=noise).item()
    for f in (0.9, 1.1):
        sc = lambda x, t, f=f: f * oracle_score(sde)(x, t)  # noqa: E731
        assert dsm_loss_sde(sc, x0, sde, t=t, noise=noise).item() > best


def _check_stats(x):
    assert torch.allclose(x.mean(0), MU, atol=0.03), x.mean(0)
    assert torch.allclose(x.std(0), torch.full((2,), S, dtype=D), atol=0.03), x.std(0)


def test_reverse_sde_vp_and_ve():
    g = torch.Generator().manual_seed(0)
    for sde in (VPSDE(), VESDE(0.01, 10.0)):
        xT = sde.prior_sample((20_000, 2), generator=g, dtype=D)
        _check_stats(reverse_sde_sample(oracle_score(sde), sde, xT, 1000, generator=g))


def test_pf_ode_vp_and_ve():
    g = torch.Generator().manual_seed(1)
    # sigma_max must dwarf the data scale: the ODE map carries any prior mismatch straight to the data
    for sde in (VPSDE(), VESDE(0.01, 50.0)):
        xT = sde.prior_sample((20_000, 2), generator=g, dtype=D)
        _check_stats(pf_ode_sample(oracle_score(sde), sde, xT, 200))


def test_pf_ode_is_deterministic():
    sde = VPSDE()
    xT = torch.randn(10, 2, dtype=D)
    torch.testing.assert_close(pf_ode_sample(oracle_score(sde), sde, xT, 50),
                               pf_ode_sample(oracle_score(sde), sde, xT, 50))


def test_pf_ode_log_likelihood_exact_for_gaussian():
    sde = VPSDE()
    x0 = MU + S * torch.randn(20, 2, dtype=D)
    eps = 1e-5
    m, std = sde.marginal(torch.tensor([eps], dtype=D))
    ref = MultivariateNormal(m * MU, (m**2 * S**2 + std**2) * torch.eye(2, dtype=D)).log_prob(x0)
    out = pf_ode_log_likelihood(oracle_score(sde), sde, x0, n_steps=2000, eps=eps)
    assert torch.allclose(out, ref, atol=0.05), (out - ref).abs().max()
