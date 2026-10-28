import math

import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from gen_basics import fit_gaussian, gaussian_kl_1d, gmm_log_prob, kl_monte_carlo


def _bimodal():
    means = torch.tensor([[-2.0], [2.0]])
    log_stds = torch.log(torch.tensor([[0.5], [0.5]]))
    logits = torch.zeros(2)
    return means, log_stds, logits


def test_gmm_log_prob_matches_torch():
    torch.manual_seed(0)
    K, D = 3, 2
    means, log_stds, logits = torch.randn(K, D), 0.3 * torch.randn(K, D), torch.randn(K)
    ref = MixtureSameFamily(Categorical(logits=logits), Independent(Normal(means, log_stds.exp()), 1))
    x = 2 * torch.randn(100, D)
    torch.testing.assert_close(gmm_log_prob(x, means, log_stds, logits), ref.log_prob(x), atol=1e-5, rtol=1e-5)


def test_gmm_log_prob_is_stable_far_away():
    means, log_stds, logits = _bimodal()
    out = gmm_log_prob(torch.tensor([[200.0]]), means, log_stds, logits)
    assert torch.isfinite(out).all()


def test_gaussian_kl_1d():
    p, q = Normal(0.3, 1.2), Normal(-1.0, 0.7)
    ref = torch.distributions.kl_divergence(p, q)
    out = gaussian_kl_1d(torch.tensor(0.3), torch.tensor(1.2), torch.tensor(-1.0), torch.tensor(0.7))
    torch.testing.assert_close(out, ref)


def test_kl_monte_carlo_converges():
    torch.manual_seed(0)
    p, q = Normal(0.0, 1.0), Normal(1.0, 2.0)
    est = kl_monte_carlo(lambda x: p.log_prob(x[:, 0]), lambda x: q.log_prob(x[:, 0]), p.sample((200_000, 1)))
    assert abs(est.item() - torch.distributions.kl_divergence(p, q).item()) < 0.01


def test_forward_kl_is_mode_covering_reverse_is_mode_seeking():
    means, log_stds, logits = _bimodal()
    target_log_prob = lambda x: gmm_log_prob(x, means, log_stds, logits)  # noqa: E731

    def sampler(n):
        comp = torch.randint(0, 2, (n, 1))
        return means[comp, 0] + 0.5 * torch.randn(n, 1)

    mu_f, s_f = fit_gaussian(target_log_prob, sampler, "forward", steps=1500)
    mu_r, s_r = fit_gaussian(target_log_prob, sampler, "reverse", steps=1500, init_mu=1.0, init_sigma=0.5)
    # forward KL: moment matching -> mean 0, std = sqrt(4 + 0.25)
    assert abs(mu_f) < 0.2 and abs(s_f - math.sqrt(4.25)) < 0.2
    # reverse KL: collapses onto one mode
    assert abs(abs(mu_r) - 2.0) < 0.2 and abs(s_r - 0.5) < 0.1
