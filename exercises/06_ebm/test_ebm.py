import torch

from ebm import contrastive_divergence_loss, langevin_sample, score_from_energy

MU, S = torch.tensor([1.0, -2.0], dtype=torch.float64), 0.7


def energy(x):  # Gaussian N(MU, S^2 I), up to a constant
    return ((x - MU) ** 2).sum(-1) / (2 * S**2)


def test_score_from_energy_gaussian():
    x = torch.randn(10, 2, dtype=torch.float64)
    torch.testing.assert_close(score_from_energy(energy, x), -(x - MU) / S**2)
    with torch.no_grad():
        torch.testing.assert_close(score_from_energy(energy, x), -(x - MU) / S**2)


def test_langevin_converges_to_target():
    g = torch.Generator().manual_seed(0)
    x0 = 5 * torch.randn(20_000, 2, dtype=torch.float64, generator=g)
    x = langevin_sample(lambda x: score_from_energy(energy, x), x0, 0.01, 1000, generator=g)
    assert torch.allclose(x.mean(0), MU, atol=0.03)
    # Langevin with finite step size has O(step) bias in the variance
    assert torch.allclose(x.std(0), torch.full((2,), S, dtype=torch.float64), atol=0.03)


def test_contrastive_divergence_gradient_direction():
    w = torch.tensor(1.0, requires_grad=True)
    e = lambda x: w * (x**2).sum(-1)  # noqa: E731
    x_data, x_model = 0.1 * torch.randn(100, 2), (3 * torch.randn(100, 2)).requires_grad_(True)
    loss = contrastive_divergence_loss(e, x_data, x_model)
    loss.backward()
    # data has low |x|, model samples high |x|: increasing w lowers data energy relative to model
    assert w.grad < 0
    assert x_model.grad is None
