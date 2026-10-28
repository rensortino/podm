"""Module 04 - From hierarchical VAEs to diffusion (book §2.1.4-§2.2.1, pp. 52-58).

A Markovian hierarchical VAE (HVAE) with latents z_1, ..., z_L:

    generative: p(z_L) p(z_{L-1}|z_L) ... p(z_1|z_2) p(x|z_1)
    inference : q(z_1|x) q(z_2|z_1) ... q(z_L|z_{L-1})

DDPM = an HVAE where (i) every z_i has the same dimension as x,
(ii) the encoder q is *fixed* Gaussian noising, not learned, and
(iii) L is large (hundreds to thousands).
"""

import math

import torch


def _gauss_logp(x, mu, logvar):
    return (-0.5 * ((x - mu) ** 2 * torch.exp(-logvar) + logvar + math.log(2 * math.pi))).flatten(1).sum(1)


def hvae_elbo(x: torch.Tensor, encoders: list, decoders: list, generator: torch.Generator | None = None) -> torch.Tensor:
    """Single-sample ELBO of a Markovian HVAE with L = len(encoders) latent layers.

    Args:
        encoders: [q_1, ..., q_L]; q_i maps z_{i-1} (z_0 = x) -> (mu, logvar) of q(z_i | z_{i-1}).
        decoders: [p_0, ..., p_{L-1}]; p_i maps z_{i+1} -> (mu, logvar) of p(z_i | z_{i+1}) (p_0 is p(x|z_1)).
        The top prior p(z_L) is N(0, I).
    Returns:
        (B,) estimate of  E_q[ log p(x, z_{1:L}) - log q(z_{1:L} | x) ].

    Sample z_1..z_L ancestrally through the encoders (reparameterised), then add up
    every log p term and subtract every log q term. `_gauss_logp` is provided.
    """
    raise NotImplementedError("TODO")


def fixed_noising_encoder(beta: float):
    """Return a *non-learned* encoder q(z_i | z_{i-1}) = N( sqrt(1 - beta) z_{i-1}, beta I ),
    in the (mu, logvar) format used by hvae_elbo. This is the DDPM forward step.
    """
    raise NotImplementedError("TODO")


def composed_marginal_coeffs(betas: torch.Tensor):
    """Composing the fixed encoders gives a closed form for q(z_i | x):

        z_i = sqrt(alpha_bar_i) x + sqrt(1 - alpha_bar_i) eps,   alpha_bar_i = prod_{j<=i} (1 - beta_j)

    Return (sqrt(alpha_bar), sqrt(1 - alpha_bar)), each of shape (L,).
    Prove it by induction before coding: a sum of independent Gaussians is Gaussian.
    """
    raise NotImplementedError("TODO")
