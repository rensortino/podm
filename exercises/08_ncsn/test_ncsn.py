import math

import torch

from ncsn import annealed_langevin, ddpm_eps_to_score, geometric_sigmas, ncsn_loss

D = torch.float64
MU, S = torch.tensor([2.0, -1.0], dtype=D), 0.6


def oracle(x, sigma):
    v = (S**2 + sigma**2).reshape(-1, 1)
    return -(x - MU) / v


def test_geometric_sigmas():
    s = geometric_sigmas(50.0, 0.01, 10)
    assert s.shape == (10,)
    torch.testing.assert_close(s[0], torch.tensor(50.0, dtype=D))
    torch.testing.assert_close(s[-1], torch.tensor(0.01, dtype=D))
    r = s[1:] / s[:-1]
    torch.testing.assert_close(r, r[0].expand_as(r))


def test_ncsn_loss_oracle_beats_perturbations_and_value():
    torch.manual_seed(0)
    sig = geometric_sigmas(5.0, 0.05, 8)
    x = MU + S * torch.randn(100_000, 2, dtype=D)
    idx, noise = torch.randint(0, 8, (100_000,)), torch.randn(100_000, 2, dtype=D)
    best = ncsn_loss(oracle, x, sig, idx, noise).item()
    for f in (0.8, 1.2):
        assert ncsn_loss(lambda x, s: f * oracle(x, s), x, sig, idx, noise).item() > best
    # exact value for a fixed level: 1/2 E||sigma s + eps||^2 at optimum = D/2 * S^2 / (S^2 + sigma^2)
    i = torch.full((100_000,), 3)
    expect = 0.5 * 2 * S**2 / (S**2 + sig[3].item() ** 2)
    assert abs(ncsn_loss(oracle, x, sig, i, noise).item() - expect) < 0.01


def test_ncsn_loss_supports_images():
    sig = geometric_sigmas(1.0, 0.1, 4)
    out = ncsn_loss(lambda x, s: -x / (1 + s.reshape(-1, 1, 1, 1) ** 2), torch.randn(3, 1, 4, 4), sig.float())
    assert out.dim() == 0


def test_annealed_langevin_with_oracle():
    g = torch.Generator().manual_seed(0)
    sig = geometric_sigmas(10.0, 0.01, 10)
    x0 = 10 * torch.randn(10_000, 2, dtype=D, generator=g)
    x = annealed_langevin(oracle, x0, sig, eps=2e-5, n_steps_each=100, generator=g)
    assert torch.allclose(x.mean(0), MU, atol=0.03)
    assert torch.allclose(x.std(0), torch.full((2,), S, dtype=D), atol=0.03)


def test_ddpm_eps_to_score():
    ab = torch.tensor([0.3], dtype=D)
    eps = torch.randn(4, 2, dtype=D)
    torch.testing.assert_close(ddpm_eps_to_score(eps, ab), -eps / math.sqrt(0.7))
