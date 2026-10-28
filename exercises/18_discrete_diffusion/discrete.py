"""Module 18 (optional) - Discrete diffusion: masked (absorbing-state) diffusion
(book Epilogue §12.2 "Beyond Continuous States", pp. 415-434).

Tokens x in {0..K-1}; an extra MASK id = K. Linear schedule: at time t in [0, 1] each token is
independently masked with probability t (t = 0 clean, t = 1 fully masked).
The model maps x_t (B, L) -> logits (B, L, K) over clean tokens.
"""

import torch
import torch.nn.functional as F


def mask_tokens(x0: torch.Tensor, t: torch.Tensor, mask_id: int, generator: torch.Generator | None = None):
    """Forward process q(x_t | x_0): mask each token w.p. t (t is (B,)). Returns x_t."""
    raise NotImplementedError("TODO")


def masked_diffusion_loss(logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, mask_id: int):
    """Continuous-time ELBO for masked diffusion with the linear schedule (MDLM / book §12.2):
        L = E_t [ (1/t) * sum_{l : x_t^l = MASK} -log p_theta(x_0^l | x_t) ]
    Unmasked positions contribute nothing. Sum over positions, mean over batch.
    """
    raise NotImplementedError("TODO")


@torch.no_grad()
def sample_masked(model, shape, n_steps: int, mask_id: int, generator: torch.Generator | None = None):
    """Reverse process on a uniform grid t: 1 -> 0. From t to s < t, each still-masked token is
    revealed with probability (t - s) / t, with its value drawn from softmax(logits).
    Already-revealed tokens never change. Returns (B, L) with no MASK left."""
    raise NotImplementedError("TODO")
