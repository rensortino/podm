import math

import torch

from distill import ddim_step_x0, pd_loss, progressive_distillation_target

D = torch.float64
MU, S = torch.tensor([1.0, -0.5], dtype=D), 0.5


def teacher(x, t):  # exact x0 = E[x0 | x_t] for Gaussian data on the cosine path
    a = torch.cos(math.pi / 2 * t).reshape(-1, 1)
    s = torch.sin(math.pi / 2 * t).reshape(-1, 1)
    return MU + a * S**2 / (a**2 * S**2 + s**2) * (x - a * MU)


def test_ddim_step_x0_consistent_with_exact_x0():
    x0, eps = torch.randn(5, 2, dtype=D), torch.randn(5, 2, dtype=D)
    t, s = 0.8, 0.3
    a_t, s_t = math.cos(math.pi / 2 * t), math.sin(math.pi / 2 * t)
    a_s, s_s = math.cos(math.pi / 2 * s), math.sin(math.pi / 2 * s)
    torch.testing.assert_close(ddim_step_x0(a_t * x0 + s_t * eps, x0, t, s), a_s * x0 + s_s * eps)


def test_one_student_step_matches_two_teacher_steps():
    x_t = torch.randn(10, 2, dtype=D)
    t, s = 0.9, 0.5
    target = progressive_distillation_target(teacher, x_t, t, s)
    m = 0.7
    x_m = ddim_step_x0(x_t, teacher(x_t, torch.full((10,), t, dtype=D)), t, m)
    x_s = ddim_step_x0(x_m, teacher(x_m, torch.full((10,), m, dtype=D)), m, s)
    torch.testing.assert_close(ddim_step_x0(x_t, target, t, s), x_s)


def test_tensor_times_supported():
    x_t = torch.randn(4, 2, dtype=D)
    t = torch.tensor([0.9, 0.6, 0.5, 0.3], dtype=D)
    out = progressive_distillation_target(teacher, x_t, t, t - 0.25)
    for k in range(4):
        ref = progressive_distillation_target(teacher, x_t[k:k + 1], t[k].item(), t[k].item() - 0.25)
        torch.testing.assert_close(out[k:k + 1], ref)


def test_pd_loss_zero_for_ideal_student():
    N = 4
    ideal = lambda x, t: progressive_distillation_target(teacher, x, t, t - 1.0 / N)  # noqa: E731
    x0 = MU + S * torch.randn(64, 2, dtype=D)
    assert pd_loss(ideal, teacher, x0, N).item() < 1e-20
    assert pd_loss(teacher, teacher, x0, N).item() > 1e-6
