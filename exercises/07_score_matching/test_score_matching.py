import torch

from score_matching import dsm_loss, implicit_score_matching_loss, sliced_score_matching_loss, tweedie_denoise

D = torch.float64


def linear(a):
    return lambda x: -a * x


def test_implicit_sm_exact_value():
    x = torch.randn(32, 3, dtype=D)
    a = 0.7
    ref = (-a * 3 + 0.5 * a**2 * (x**2).sum(1)).mean()
    torch.testing.assert_close(implicit_score_matching_loss(linear(a), x), ref)


def test_implicit_sm_nonlinear_trace():
    x = torch.randn(16, 2, dtype=D)
    f = lambda x: torch.stack([torch.sin(x[:, 0]) * x[:, 1], x[:, 1] ** 3], 1)  # noqa: E731
    s = f(x)
    tr = torch.cos(x[:, 0]) * x[:, 1] + 3 * x[:, 1] ** 2
    torch.testing.assert_close(implicit_score_matching_loss(f, x), (tr + 0.5 * (s**2).sum(1)).mean())


def test_ism_minimised_at_true_score():
    torch.manual_seed(0)
    x = 1.5 * torch.randn(100_000, 2, dtype=D)  # true score -x / 2.25
    losses = {a: implicit_score_matching_loss(linear(a), x).item() for a in (0.3, 1 / 2.25, 0.6)}
    assert losses[1 / 2.25] < losses[0.3] and losses[1 / 2.25] < losses[0.6]


def test_sliced_sm_matches_implicit_in_expectation():
    x = torch.randn(64, 4, dtype=D)
    f = lambda x: torch.tanh(x) * 2 - 0.5 * x  # noqa: E731
    ism = implicit_score_matching_loss(f, x)
    ssm = sliced_score_matching_loss(f, x, n_projections=4000, generator=torch.Generator().manual_seed(0))
    assert abs(ism.item() - ssm.item()) < 0.05


def test_sliced_sm_has_gradients():
    lin = torch.nn.Linear(2, 2).double()
    sliced_score_matching_loss(lin, torch.randn(8, 2, dtype=D)).backward()
    assert lin.weight.grad is not None


def test_dsm_minimiser_is_noisy_score():
    torch.manual_seed(0)
    s_data, sigma = 1.0, 0.5
    x = s_data * torch.randn(200_000, 2, dtype=D)
    a_star = 1 / (s_data**2 + sigma**2)
    noise = torch.randn_like(x)
    losses = {a: dsm_loss(linear(a), x, sigma, noise).item() for a in (a_star * 0.8, a_star, a_star * 1.2)}
    assert losses[a_star] < min(losses[a_star * 0.8], losses[a_star * 1.2])
    out = dsm_loss(linear(a_star), x[:4], sigma, torch.zeros(4, 2, dtype=D))
    torch.testing.assert_close(out, 0.5 * (a_star**2 * (x[:4] ** 2).sum(1)).mean())


def test_tweedie_gaussian_posterior_mean():
    s2, sigma = 2.0, 0.8
    x_noisy = torch.randn(5, 3, dtype=D)
    score = -x_noisy / (s2 + sigma**2)
    torch.testing.assert_close(tweedie_denoise(x_noisy, score, sigma), s2 / (s2 + sigma**2) * x_noisy)
