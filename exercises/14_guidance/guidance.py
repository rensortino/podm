"""Module 14 - Guidance and controllable generation (book Ch. 8, pp. 251-282).

Everything is phrased with scores; the same formulas hold for eps-predictions
(eps = -sigma * score is linear) and velocities (affine in the score, module 12).
"""

import torch
import torch.nn.functional as F


def classifier_guided_score(uncond_score: torch.Tensor, classifier_log_prob, x: torch.Tensor, y: torch.Tensor,
                            scale: float) -> torch.Tensor:
    """Classifier guidance (book §8.2):  s_uncond(x) + w * grad_x log p_phi(y | x_t).

    classifier_log_prob(x, y) -> (B,) log p(y | x) of a classifier trained on *noisy* x_t.
    Must work under torch.no_grad() (samplers run without grad).
    """
    raise NotImplementedError("TODO")


def cfg_combine(cond: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-free guidance (book §8.3):  uncond + w * (cond - uncond).
    w = 0 -> unconditional, w = 1 -> conditional, w > 1 -> sharpened p(x) p(y|x)^w.
    """
    raise NotImplementedError("TODO")


def drop_labels(y: torch.Tensor, p_uncond: float, null_label: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """For CFG training: replace each label by null_label with probability p_uncond (returns a new tensor)."""
    raise NotImplementedError("TODO")


def dps_guidance(x: torch.Tensor, x0_predictor, measurement: torch.Tensor, operator, scale: float) -> torch.Tensor:
    """Training-free guidance a la Diffusion Posterior Sampling (book §8.4):
        approximate grad_x log p(y | x_t) by  -scale * grad_x || y - A( x0_hat(x_t) ) ||^2,
    differentiating *through* the denoiser x0_hat (Tweedie). Return that guidance term.
    """
    raise NotImplementedError("TODO")


def diffusion_dpo_loss(err_w, err_ref_w, err_l, err_ref_l, beta: float) -> torch.Tensor:
    """Diffusion-DPO (book §8.5): per-pair denoising errors ||eps - eps_hat||^2 of the policy and the
    frozen reference on the preferred (w) and dispreferred (l) samples at the same (t, eps).

        L = -log sigmoid( -beta * [ (err_w - err_ref_w) - (err_l - err_ref_l) ] )   (mean over pairs)
    """
    raise NotImplementedError("TODO")
