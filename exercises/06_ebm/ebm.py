"""Module 06 - Energy-based models (book §3.1, pp. 70-76).

p_theta(x) = exp(-E_theta(x)) / Z_theta.  Z is intractable, but the *score*
grad_x log p(x) = -grad_x E(x) does not depend on Z - the seed of everything in Part B's score view.
"""

import torch


def score_from_energy(energy_fn, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """grad_x log p(x) = -grad_x E(x), for energy_fn: (N, D) -> (N,). Must work under torch.no_grad()."""
    raise NotImplementedError("TODO")


def langevin_sample(score_fn, x0: torch.Tensor, step_size: float, n_steps: int,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Unadjusted Langevin dynamics:  x <- x + step * score(x) + sqrt(2 * step) * z.

    As step -> 0 and n_steps -> inf, x is distributed as p. (It is Euler-Maruyama on
    dx = grad log p(x) dt + sqrt(2) dW, whose stationary distribution is p.)
    """
    raise NotImplementedError("TODO")


def contrastive_divergence_loss(energy_fn, x_data: torch.Tensor, x_model: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """Maximum-likelihood gradient for EBMs (book §3.1.2):

        grad_theta NLL = E_data[grad_theta E(x)] - E_model[grad_theta E(x)]

    so the surrogate loss is  mean E(x_data) - mean E(x_model)  with x_model *detached*
    (they are MCMC samples, not a function of theta), plus alpha * mean(E(x_data)^2 + E(x_model)^2)
    as a regulariser that keeps energies bounded.
    """
    raise NotImplementedError("TODO")
