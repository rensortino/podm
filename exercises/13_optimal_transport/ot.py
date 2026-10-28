"""Module 13 (optional) - Diffusion models and optimal transport (book Ch. 7, pp. 214-248).

Question the chapter answers: is the noise -> data map of a diffusion PF-ODE the
*optimal* transport map? (Short answer: not in general; it is in 1D and for Gaussians.)
"""

import torch


def monotone_map_1d(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Empirical OT map in 1D for any convex cost: match the k-th smallest source to the
    k-th smallest target. source, target: (N,). Returns T(source), aligned with source."""
    raise NotImplementedError("TODO")


def transport_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean cost of a pairing: mean_i ||x_i - y_i||^2."""
    raise NotImplementedError("TODO")


def path_straightness(traj: torch.Tensor) -> torch.Tensor:
    """Excess path length of a batch of trajectories traj: (K+1, N, D):
        mean_i [ sum_k ||x_{k+1} - x_k|| - ||x_K - x_0|| ]   (0 iff all paths are straight lines)."""
    raise NotImplementedError("TODO")
