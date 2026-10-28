"""Module 16 - Distillation for fast sampling (book Ch. 10, pp. 350-365).

Cosine VP path, t in [0, 1]: alpha_t = cos(pi t / 2), sigma_t = sin(pi t / 2).
Models predict x0: model(x, t) with t a (B,) float tensor. Times t, s below are Python floats
or (B,) tensors - write code that handles both by broadcasting through `_ab`.
"""

import math

import torch


def _ab(t, x):
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if t.dim() > 0:
        t = t.reshape(-1, *([1] * (x.dim() - 1)))
    return torch.cos(math.pi / 2 * t), torch.sin(math.pi / 2 * t)


def _full(x, t):
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    return t.expand(x.shape[0]) if t.dim() == 0 else t


def ddim_step_x0(x_t, x0_hat, t, s):
    """Deterministic DDIM step t -> s written with an x0-prediction:
        x_s = alpha_s x0_hat + sigma_s (x_t - alpha_t x0_hat) / sigma_t
    """
    raise NotImplementedError("TODO")


def progressive_distillation_target(teacher, x_t, t, s):
    """Salimans & Ho (2022), book §10.3. Run two teacher DDIM steps t -> (t+s)/2 -> s to get x_s, then
    return the x0 target x0_tilde such that ONE student DDIM step t -> s lands exactly on x_s:
        x0_tilde = (x_s - (sigma_s / sigma_t) x_t) / (alpha_s - (sigma_s / sigma_t) alpha_t)
    """
    raise NotImplementedError("TODO")


def pd_loss(student, teacher, x0: torch.Tensor, n_student_steps: int, i: torch.Tensor | None = None,
            noise: torch.Tensor | None = None) -> torch.Tensor:
    """Progressive distillation loss on the student grid t_i = i / N, i in {1..N}, s = t_i - 1/N:
        x_t = alpha_t x0 + sigma_t noise
        L = E[ max(SNR(t), 1) * || student(x_t, t) - x0_tilde ||^2 ]   ("truncated SNR" weighting)
    x0_tilde is a fixed target (no grad). Sum over dims, mean over batch.
    """
    raise NotImplementedError("TODO")
