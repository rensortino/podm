import math

import torch

from consistency import (cd_loss, cm_coeffs, consistency_fn, ct_loss, karras_timesteps, meanflow_loss,
                         meanflow_sample, meanflow_target, multistep_consistency_sample)

D = torch.float64
MU, S, EPS = torch.tensor([1.0, -0.5], dtype=D), 0.5, 0.002


def exact_step(x, t, s):
    """Exact VE PF-ODE map for Gaussian data N(MU, S^2): x_s = MU + sqrt(V_s / V_t)(x_t - MU)."""
    V = lambda u: S**2 + torch.as_tensor(u, dtype=D) ** 2  # noqa: E731
    r = torch.sqrt(V(s) / V(t))
    return MU + (r.reshape(-1, 1) if r.dim() else r) * (x - MU)


def exact_f(x, t):
    return exact_step(x, t, torch.full_like(t, EPS))


def test_karras_timesteps():
    ts = karras_timesteps(18)
    assert ts.shape == (18,) and abs(ts[0] - EPS) < 1e-12 and abs(ts[-1] - 80) < 1e-9
    assert (ts[1:] > ts[:-1]).all()


def test_boundary_condition():
    x = torch.randn(4, 2, dtype=D)
    t = torch.full((4,), EPS, dtype=D)
    torch.testing.assert_close(consistency_fn(lambda x, t: 100 * torch.ones_like(x), x, t), x)
    c_skip, c_out = cm_coeffs(torch.tensor(1.0, dtype=D))
    assert 0 < c_skip < 1 and c_out > 0


def test_cd_loss_zero_for_exact_consistency_function():
    ts = karras_timesteps(18)
    x0 = MU + S * torch.randn(256, 2, dtype=D)
    assert cd_loss(exact_f, exact_f, exact_step, x0, ts).item() < 1e-20
    bad = lambda x, t: exact_f(x, t) + 0.1  # noqa: E731
    assert cd_loss(bad, exact_f, exact_step, x0, ts).item() > 1e-3


def test_ct_loss_uses_shared_noise():
    ts = karras_timesteps(10)
    x0, z, n = torch.randn(8, 2, dtype=D), torch.randn(8, 2, dtype=D), torch.randint(0, 9, (8,))
    ident = lambda x, t: x  # noqa: E731
    ref = (((ts[n + 1] - ts[n]).reshape(-1, 1) * z) ** 2).sum(1).mean()
    torch.testing.assert_close(ct_loss(ident, ident, x0, ts, n, z), ref)


def test_multistep_sampling_with_exact_f():
    g = torch.Generator().manual_seed(0)
    xT = 80 * torch.randn(20_000, 2, dtype=D, generator=g)
    for sts in ([80.0], [80.0, 1.0, 0.3]):
        x = multistep_consistency_sample(exact_f, xT, sts, generator=g)
        assert torch.allclose(x.mean(0), MU, atol=0.03) and torch.allclose(x.std(0), torch.full((2,), S, dtype=D), atol=0.03)


A = 0.8  # toy flow dz/dt = A z  ->  z_r = z_t exp(-A (t - r))


def exact_u(z, r, t):
    h = (t - r).reshape(-1, 1)
    return z * -torch.expm1(-A * h) / h


def test_meanflow_identity_fixed_point():
    z = torch.randn(6, 2, dtype=D)
    t = torch.rand(6, dtype=D) * 0.5 + 0.5
    r = t - torch.rand(6, dtype=D) * 0.4 - 0.05
    torch.testing.assert_close(meanflow_target(exact_u, z, r, t, A * z), exact_u(z, r, t))


def test_meanflow_one_step_sample():
    z = torch.randn(6, 2, dtype=D)
    torch.testing.assert_close(meanflow_sample(exact_u, z, 1), z * math.exp(-A))
    torch.testing.assert_close(meanflow_sample(exact_u, z, 4), z * math.exp(-A))


def test_meanflow_loss_backprop():
    net = torch.nn.Linear(4, 2).double()
    u = lambda z, r, t: net(torch.cat([z, r[:, None], t[:, None]], 1))  # noqa: E731
    loss = meanflow_loss(u, torch.randn(16, 2, dtype=D))
    loss.backward()
    assert net.weight.grad is not None and torch.isfinite(loss)
