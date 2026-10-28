import math

import torch

from flow_matching import cfm_loss, cosine_schedule, interpolate, linear_schedule, minibatch_ot_pairing, sample_ode

D = torch.float64
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5


def oracle_velocity(schedule):
    """Marginal velocity E[alpha' x0 + sigma' eps | x_t] for Gaussian data N(MU, S^2 I)."""

    def v(x, t):
        a, s, da, ds = (u.reshape(-1, 1).to(x.dtype) for u in schedule(t))
        V = a**2 * S**2 + s**2
        r = x - a * MU
        return da * (MU + a * S**2 / V * r) + ds * (s / V * r)

    return v


def test_schedule_derivatives():
    t = torch.rand(10, dtype=D) * 0.98 + 0.01
    h = 1e-6
    for sch in (linear_schedule, cosine_schedule):
        a, s, da, ds = sch(t)
        ap, sp, _, _ = sch(t + h)
        am, sm, _, _ = sch(t - h)
        torch.testing.assert_close(da, (ap - am) / (2 * h), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ds, (sp - sm) / (2 * h), atol=1e-6, rtol=1e-6)
    a, s, _, _ = cosine_schedule(t)
    torch.testing.assert_close(a**2 + s**2, torch.ones_like(t))


def test_interpolate_endpoints():
    x0, eps = torch.randn(4, 2, dtype=D), torch.randn(4, 2, dtype=D)
    xt, u = interpolate(x0, eps, torch.zeros(4, dtype=D), linear_schedule)
    torch.testing.assert_close(xt, x0)
    xt, u = interpolate(x0, eps, torch.ones(4, dtype=D), linear_schedule)
    torch.testing.assert_close(xt, eps)
    torch.testing.assert_close(u, eps - x0)


def test_cfm_loss_oracle_minimises_and_conditional_target_is_zero():
    torch.manual_seed(0)
    x0 = MU + S * torch.randn(50_000, 2, dtype=D)
    t, eps = torch.rand(50_000, dtype=D), torch.randn(50_000, 2, dtype=D)
    v = oracle_velocity(linear_schedule)
    best = cfm_loss(v, x0, linear_schedule, t, eps).item()
    for f in (0.9, 1.1):
        assert cfm_loss(lambda x, t, f=f: f * v(x, t), x0, linear_schedule, t, eps).item() > best
    # a "cheating" model that knows x0 and eps gets zero loss
    cheat = lambda x, tt: eps - x0  # noqa: E731
    assert cfm_loss(cheat, x0, linear_schedule, t, eps).item() < 1e-20


def test_sampling_with_oracle_velocity():
    g = torch.Generator().manual_seed(0)
    for sch, method, n in ((linear_schedule, "euler", 100), (cosine_schedule, "heun", 50)):
        x1 = torch.randn(20_000, 2, generator=g, dtype=D)
        x = sample_ode(oracle_velocity(sch), x1, n, method)
        assert torch.allclose(x.mean(0), MU, atol=0.03)
        assert torch.allclose(x.std(0), torch.full((2,), S, dtype=D), atol=0.03)


def test_trajectory_shape():
    x, traj = sample_ode(oracle_velocity(linear_schedule), torch.randn(5, 2, dtype=D), 10, return_trajectory=True)
    assert traj.shape == (11, 5, 2)


def test_minibatch_ot_pairing_reduces_cost():
    torch.manual_seed(0)
    x0, eps = torch.randn(64, 2) + 3, torch.randn(64, 2)
    paired = minibatch_ot_pairing(x0, eps)
    assert sorted(map(tuple, paired.tolist())) == sorted(map(tuple, eps.tolist()))
    assert ((x0 - paired) ** 2).sum() < ((x0 - eps) ** 2).sum()
