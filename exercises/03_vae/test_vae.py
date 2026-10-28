import math

import torch
from torch.distributions import Bernoulli, MultivariateNormal, Normal, kl_divergence

from vae import (bernoulli_log_prob, elbo, gaussian_kl, gaussian_log_prob, kl_to_standard_normal,
                 reparameterize)


def test_reparameterize_stats_and_grad():
    mu = torch.tensor([1.0, -2.0], requires_grad=True)
    logvar = torch.log(torch.tensor([0.25, 4.0])).requires_grad_(True)
    z = reparameterize(mu.expand(100_000, 2), logvar.expand(100_000, 2), torch.Generator().manual_seed(0))
    assert torch.allclose(z.mean(0), mu.detach(), atol=0.02)
    assert torch.allclose(z.var(0), torch.tensor([0.25, 4.0]), rtol=0.03)
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def test_gaussian_log_prob():
    x, mu, lv = torch.randn(4, 3, 2), torch.randn(4, 3, 2), torch.randn(4, 3, 2)
    ref = Normal(mu, (0.5 * lv).exp()).log_prob(x).sum((1, 2))
    torch.testing.assert_close(gaussian_log_prob(x, mu, lv), ref)


def test_bernoulli_log_prob_stable():
    x = torch.randint(0, 2, (5, 10)).float()
    logits = 3 * torch.randn(5, 10)
    torch.testing.assert_close(bernoulli_log_prob(x, logits), Bernoulli(logits=logits).log_prob(x).sum(1))
    assert torch.isfinite(bernoulli_log_prob(torch.ones(1, 1), torch.tensor([[-200.0]]))).all()


def test_gaussian_kls():
    mq, lq, mp, lp = (torch.randn(6, 4) for _ in range(4))
    ref = kl_divergence(Normal(mq, (0.5 * lq).exp()), Normal(mp, (0.5 * lp).exp())).sum(1)
    torch.testing.assert_close(gaussian_kl(mq, lq, mp, lp), ref)
    ref0 = kl_divergence(Normal(mq, (0.5 * lq).exp()), Normal(0.0, 1.0)).sum(1)
    torch.testing.assert_close(kl_to_standard_normal(mq, lq), ref0)


# Linear-Gaussian model: z ~ N(0, I_2), x = W z + b + sigma * noise.
# p(x) = N(b, W W^T + sigma^2 I) is tractable, so we can check the bound exactly.
W = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
b = torch.tensor([0.5, -1.0, 0.2], dtype=torch.float64)
SIG2 = 0.3


def _decoder(z):
    mu = z @ W.T + b
    return mu, torch.full_like(mu, math.log(SIG2))


def _exact_posterior_encoder(x):
    # W^T W is diagonal here, so the true posterior is a diagonal Gaussian.
    prec = torch.eye(2, dtype=torch.float64) + W.T @ W / SIG2
    cov = torch.linalg.inv(prec)
    mu = (x - b) @ W @ cov.T / SIG2
    return mu, torch.log(torch.diagonal(cov)).expand_as(mu)


def _log_px(x):
    return MultivariateNormal(b, W @ W.T + SIG2 * torch.eye(3, dtype=torch.float64)).log_prob(x)


def test_elbo_tight_with_exact_posterior():
    x = torch.randn(16, 3, dtype=torch.float64)
    out = elbo(x, _exact_posterior_encoder, _decoder, "gaussian", n_samples=20_000,
               generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out, _log_px(x), atol=0.05, rtol=0)


def test_elbo_is_a_lower_bound_with_wrong_posterior():
    x = torch.randn(16, 3, dtype=torch.float64)
    bad = lambda x: (torch.zeros(x.shape[0], 2, dtype=x.dtype), torch.zeros(x.shape[0], 2, dtype=x.dtype))  # noqa: E731
    out = elbo(x, bad, _decoder, "gaussian", n_samples=2000, generator=torch.Generator().manual_seed(0))
    assert (out < _log_px(x)).all()
