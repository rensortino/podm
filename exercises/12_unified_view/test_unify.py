import math

import torch

from unify import (edm_coeffs, edm_denoiser, eps_to_score, eps_to_x0, score_to_eps, to_velocity,
                   velocity_from_score, weight_eps_per_x0, weight_v_per_eps, x0_eps_from_velocity, x0_to_eps)

D = torch.float64
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5


def _sched(t):  # cosine path
    h = math.pi / 2
    return torch.cos(h * t), torch.sin(h * t), -h * torch.sin(h * t), h * torch.cos(h * t)


def _rand():
    t = (torch.rand(16, 1, dtype=D) * 0.9 + 0.05)
    return t, _sched(t), torch.randn(16, 2, dtype=D), torch.randn(16, 2, dtype=D)


def test_roundtrips():
    t, (a, s, da, ds), x0, eps = _rand()
    x = a * x0 + s * eps
    torch.testing.assert_close(eps_to_x0(x, eps, a, s), x0)
    torch.testing.assert_close(x0_to_eps(x, x0, a, s), eps)
    torch.testing.assert_close(score_to_eps(eps_to_score(eps, s), s), eps)
    v = to_velocity(x0, eps, da, ds)
    x0r, er = x0_eps_from_velocity(x, v, a, s, da, ds)
    torch.testing.assert_close(x0r, x0)
    torch.testing.assert_close(er, eps)


def test_velocity_from_oracle_score_equals_oracle_fm_velocity():
    """The PF-ODE (score view) and flow matching (velocity view) are the same field."""
    t = torch.rand(32, 1, dtype=D) * 0.9 + 0.05
    a, s, da, ds = _sched(t)
    x = torch.randn(32, 2, dtype=D)
    V = a**2 * S**2 + s**2
    score = -(x - a * MU) / V
    r = x - a * MU
    v_fm = da * (MU + a * S**2 / V * r) + ds * (s / V * r)
    torch.testing.assert_close(velocity_from_score(x, score, a, s, da, ds), v_fm)


def test_loss_weights():
    t, (a, s, da, ds), x0, eps = _rand()
    x = a * x0 + s * eps
    eps_hat = eps + 0.1 * torch.randn_like(eps)
    x0_hat = eps_to_x0(x, eps_hat, a, s)
    v_hat, v = to_velocity(x0_hat, eps_hat, da, ds), to_velocity(x0, eps, da, ds)
    e_eps = ((eps_hat - eps) ** 2).sum(1, keepdim=True)
    torch.testing.assert_close(e_eps, weight_eps_per_x0(a, s) * ((x0_hat - x0) ** 2).sum(1, keepdim=True))
    torch.testing.assert_close(((v_hat - v) ** 2).sum(1, keepdim=True), weight_v_per_eps(a, s, da, ds) * e_eps)


def test_edm_coeffs_and_unit_variance_target():
    sd = 0.5
    sigma = torch.tensor([0.01, 0.5, 2.0, 80.0], dtype=D)
    c_skip, c_out, c_in, c_noise = edm_coeffs(sigma, sd)
    torch.testing.assert_close(c_skip, sd**2 / (sigma**2 + sd**2))
    torch.testing.assert_close(c_noise, torch.log(sigma) / 4)
    torch.manual_seed(0)
    for i in range(4):
        x0 = sd * torch.randn(200_000, dtype=D)
        x = x0 + sigma[i] * torch.randn_like(x0)
        assert abs((c_in[i] * x).std().item() - 1) < 0.01  # network input has unit variance
        target = (x0 - c_skip[i] * x) / c_out[i]  # effective training target of F
        assert abs(target.std().item() - 1) < 0.01


def test_edm_denoiser_shapes_and_limit():
    sigma = torch.tensor([1e-4, 1e-4], dtype=D)
    x = torch.randn(2, 1, 3, 3, dtype=D)
    out = edm_denoiser(lambda x_in, c: torch.zeros_like(x_in), x, sigma)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=0)  # sigma -> 0: D(x) -> x
