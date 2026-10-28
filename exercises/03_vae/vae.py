"""Module 03 - Variational Autoencoder (book §2.1, pp. 44-54).

Conventions: Gaussians are parameterised by (mean, logvar), diagonal, and every
log-likelihood / KL returns one value per sample (summed over feature dims).
"""

import math

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample z ~ N(mu, diag(exp(logvar))) such that gradients flow to mu and logvar."""
    raise NotImplementedError("TODO")


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """log N(x; mu, diag(exp(logvar))), summed over all non-batch dims -> (B,)."""
    raise NotImplementedError("TODO")


def bernoulli_log_prob(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """log Bernoulli(x; sigmoid(logits)) summed over non-batch dims -> (B,). Must be stable for large |logits|."""
    raise NotImplementedError("TODO")


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL( N(mu_q, var_q) || N(mu_p, var_p) ) for diagonal Gaussians, summed over non-batch dims -> (B,).

    You will reuse this for the DDPM loss in module 05.
    """
    raise NotImplementedError("TODO")


def kl_to_standard_normal(mu, logvar) -> torch.Tensor:
    """KL( N(mu, var) || N(0, I) ) -> (B,). (Special case of gaussian_kl - derive the simplified formula.)"""
    raise NotImplementedError("TODO")


def elbo(x: torch.Tensor, encoder, decoder, likelihood: str = "gaussian", n_samples: int = 1,
         generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-sample Evidence Lower BOund (book Eq. for ELBO in §2.1.2):

        ELBO(x) = E_{q(z|x)}[ log p(x|z) ] - KL( q(z|x) || p(z) ),   p(z) = N(0, I)

    Args:
        encoder: x -> (mu_z, logvar_z)
        decoder: z -> (mu_x, logvar_x) if likelihood == "gaussian", or logits if "bernoulli".
        n_samples: Monte Carlo samples for the reconstruction term (average them).
    Returns:
        (B,) ELBO values. The training loss is -elbo(...).mean().
    """
    raise NotImplementedError("TODO")
