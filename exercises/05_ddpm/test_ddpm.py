import math

import torch

from ddpm import (alpha_bar_from_betas, cosine_beta_schedule, ddpm_loss, ddpm_sample, linear_beta_schedule,
                  predict_x0_from_eps, q_posterior, q_sample, vlb_term)

D = torch.float64
T = 1000
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5  # Gaussian "data": N(MU, S^2 I)


def gaussian_oracle_eps(alpha_bar):
    """Exact eps-prediction E[eps | x_t] for Gaussian data - the best any network can do."""

    def model(xt, t):
        ab = alpha_bar[t].reshape(-1, *([1] * (xt.dim() - 1))).to(xt.dtype)
        a, b = ab.sqrt(), (1 - ab).sqrt()
        return b * (xt - a * MU.to(xt.dtype)) / (a**2 * S**2 + b**2)

    return model


def test_schedules():
    b = linear_beta_schedule(T)
    assert b.shape == (T,) and abs(b[0].item() - 1e-4) < 1e-12 and abs(b[-1].item() - 0.02) < 1e-12
    ab = alpha_bar_from_betas(b)
    assert (ab[1:] < ab[:-1]).all() and ab[-1] < 1e-4
    c = cosine_beta_schedule(T)
    abc = alpha_bar_from_betas(c)
    f = lambda u: math.cos((u / T + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    assert abs(abc[T // 2].item() - f(T // 2 + 1) / f(0)) < 1e-9
    assert c.max() <= 0.999


def test_q_sample_shapes_and_stats():
    ab = alpha_bar_from_betas(linear_beta_schedule(T))
    x0 = torch.ones(50_000, 1, 2, 2, dtype=D)
    t = torch.full((50_000,), 300)
    xt = q_sample(x0, t, ab, torch.randn_like(x0))
    assert xt.shape == x0.shape
    assert abs(xt.mean().item() - ab[300].sqrt().item()) < 0.01
    assert abs(xt.var().item() - (1 - ab[300]).item()) < 0.01


def test_q_posterior_matches_gaussian_conditioning():
    betas = linear_beta_schedule(T)
    ab = alpha_bar_from_betas(betas)
    i = 500
    x0, xt = torch.randn(4, 3, dtype=D), torch.randn(4, 3, dtype=D)
    mean, var = q_posterior(x0, xt, torch.full((4,), i), betas)
    # joint Gaussian of (x_{t-1}, x_t) given x0, conditioned on x_t
    m_prev, v_prev = ab[i - 1].sqrt() * x0, 1 - ab[i - 1]
    m_t, v_t = ab[i].sqrt() * x0, 1 - ab[i]
    cov = (1 - betas[i]).sqrt() * v_prev
    torch.testing.assert_close(mean, m_prev + cov / v_t * (xt - m_t))
    torch.testing.assert_close(torch.as_tensor(var).expand(4, 3), (v_prev - cov**2 / v_t).expand(4, 3))
    mean0, var0 = q_posterior(x0, xt, torch.zeros(4, dtype=torch.long), betas)
    torch.testing.assert_close(mean0, x0)
    assert torch.as_tensor(var0).abs().max() < 1e-12


def test_predict_x0_inverts_q_sample():
    ab = alpha_bar_from_betas(linear_beta_schedule(T))
    x0, eps, t = torch.randn(8, 2, dtype=D), torch.randn(8, 2, dtype=D), torch.randint(0, T, (8,))
    torch.testing.assert_close(predict_x0_from_eps(q_sample(x0, t, ab, eps), t, eps, ab), x0)


def test_ddpm_loss_zero_for_perfect_noise_predictor():
    ab = alpha_bar_from_betas(linear_beta_schedule(T))
    x0, noise, t = torch.randn(16, 2, dtype=D), torch.randn(16, 2, dtype=D), torch.randint(0, T, (16,))
    assert ddpm_loss(lambda xt, tt: noise, x0, ab, t, noise).item() < 1e-20
    assert abs(ddpm_loss(lambda xt, tt: torch.zeros_like(xt), x0, ab, t, noise).item() - (noise**2).mean().item()) < 1e-12


def test_ddpm_sample_with_oracle_recovers_data_distribution():
    betas = linear_beta_schedule(T)
    model = gaussian_oracle_eps(alpha_bar_from_betas(betas))
    x = ddpm_sample(model, (20_000, 2), betas, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(x.mean(0).double(), MU, atol=0.03)
    assert torch.allclose(x.std(0).double(), torch.full((2,), S, dtype=D), atol=0.03)


def test_vlb_term_is_reweighted_simple_loss():
    betas = linear_beta_schedule(T)
    ab = alpha_bar_from_betas(betas)
    x0, noise = torch.randn(6, 2, dtype=D), torch.randn(6, 2, dtype=D)
    t = torch.tensor([1, 10, 100, 400, 800, 999])
    model = lambda xt, tt: 0.3 * xt  # noqa: E731  any eps-model will do
    xt = q_sample(x0, t, ab, noise)
    ab_prev = torch.cat([torch.ones(1, dtype=D), ab[:-1]])[t]
    var = betas[t] * (1 - ab_prev) / (1 - ab[t])
    w = betas[t] ** 2 / (2 * var * (1 - betas[t]) * (1 - ab[t]))
    ref = w * ((noise - model(xt, t)) ** 2).sum(1)
    torch.testing.assert_close(vlb_term(model, x0, t, betas, noise), ref)
