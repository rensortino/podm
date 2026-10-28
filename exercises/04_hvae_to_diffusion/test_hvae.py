import math

import torch
from torch.distributions import MultivariateNormal

from hvae import composed_marginal_coeffs, fixed_noising_encoder, hvae_elbo

D = torch.float64


def _lin(scale, shift, var):
    return lambda z: (scale * z + shift, torch.full_like(z, math.log(var)))


def test_hvae_elbo_lower_bounds_exact_likelihood():
    # 2-layer linear-Gaussian chain in 1D: z2 ~ N(0,1); z1 = a z2 + N(0, s1); x = c z1 + N(0, s0)
    a, s1, c, s0 = 0.8, 0.5, 1.5, 0.2
    decoders = [_lin(c, 0.0, s0), _lin(a, 0.0, s1)]
    var_x = c**2 * (a**2 + s1) + s0
    x = torch.randn(8, 1, dtype=D) * math.sqrt(var_x)
    log_px = MultivariateNormal(torch.zeros(1, dtype=D), torch.tensor([[var_x]], dtype=D)).log_prob(x)
    encoders = [_lin(0.5, 0.0, 0.5), _lin(0.5, 0.0, 0.5)]  # deliberately wrong posterior
    g = torch.Generator().manual_seed(0)
    est = torch.stack([hvae_elbo(x, encoders, decoders, g) for _ in range(4000)]).mean(0)
    assert (est < log_px).all()
    assert (est > log_px - 5).all()


def test_hvae_elbo_single_layer_matches_vae_formula():
    # with L = 1 and an exact posterior the ELBO estimator is exactly log p(x)
    s0 = 0.5
    x = torch.randn(10, 1, dtype=D)
    post_var = s0 / (1 + s0)
    enc = lambda x: (x / (1 + s0), torch.full_like(x, math.log(post_var)))  # noqa: E731
    out = hvae_elbo(x, [enc], [_lin(1.0, 0.0, s0)], torch.Generator().manual_seed(1))
    ref = MultivariateNormal(torch.zeros(1, dtype=D), torch.tensor([[1 + s0]], dtype=D)).log_prob(x)
    torch.testing.assert_close(out, ref)


def test_fixed_noising_encoder():
    enc = fixed_noising_encoder(0.1)
    mu, lv = enc(torch.ones(3, 2))
    torch.testing.assert_close(mu, torch.full((3, 2), math.sqrt(0.9)))
    torch.testing.assert_close(lv, torch.full((3, 2), math.log(0.1)))


def test_composed_marginals_match_simulation():
    betas = torch.linspace(1e-3, 0.2, 30, dtype=D)
    a, s = composed_marginal_coeffs(betas)
    x = torch.full((200_000, 1), 2.0, dtype=D)
    z = x
    g = torch.Generator().manual_seed(0)
    for b in betas:
        z = math.sqrt(1 - b) * z + math.sqrt(b) * torch.randn(z.shape, generator=g, dtype=D)
    assert abs(z.mean().item() - 2.0 * a[-1].item()) < 0.01
    assert abs(z.std().item() - s[-1].item()) < 0.01
    torch.testing.assert_close(a**2 + s**2, torch.ones_like(a))
