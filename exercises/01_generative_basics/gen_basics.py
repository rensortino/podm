"""Module 01 - Deep generative modeling basics (book Ch. 1, pp. 21-33).

Fill in every function marked TODO. Run the checks with:
    uv run pytest exercises/01_generative_basics
"""

import math

import torch


def gmm_log_prob(x: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Log-density of a diagonal Gaussian mixture.

    p(x) = sum_k pi_k N(x; mu_k, diag(sigma_k^2)),  pi = softmax(logits)

    Args:
        x: (N, D) points.
        means: (K, D), log_stds: (K, D), logits: (K,).
    Returns:
        (N,) log p(x). Use logsumexp for numerical stability - never exp then log.
    """
    raise NotImplementedError("TODO")


def gaussian_kl_1d(mu_p, std_p, mu_q, std_q):
    """Closed-form KL(N(mu_p, std_p^2) || N(mu_q, std_q^2)) for scalars/tensors."""
    raise NotImplementedError("TODO")


def kl_monte_carlo(log_p, log_q, samples_p: torch.Tensor) -> torch.Tensor:
    """Monte Carlo estimate of KL(p || q) = E_p[log p(x) - log q(x)].

    Args:
        log_p, log_q: callables mapping (N, D) -> (N,).
        samples_p: (N, D) samples from p.
    """
    raise NotImplementedError("TODO")


def fit_gaussian(target_log_prob, target_sampler, mode: str, steps: int = 2000, lr: float = 0.05,
                 n: int = 512, seed: int = 0, init_mu: float = 0.0, init_sigma: float = 1.0):
    """Fit a 1D Gaussian q = N(mu, sigma^2) to a target p by minimising a KL divergence.

    mode == "forward": minimise KL(p || q)  ->  maximise E_p[log q(x)]   (= maximum likelihood).
        Needs samples from p, never evaluates p.
    mode == "reverse": minimise KL(q || p)  ->  E_q[log q(x) - log p(x)].
        Needs log p (unnormalised is fine), samples from q with the
        *reparameterisation trick* x = mu + sigma * eps so gradients reach mu, sigma.

    Args:
        target_log_prob: (N, 1) -> (N,)
        target_sampler: int n -> (n, 1)
    Returns:
        (mu, sigma) as Python floats. Start the optimisation from (init_mu, init_sigma).
    """
    raise NotImplementedError("TODO")
