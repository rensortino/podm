import torch
import torch.nn.functional as F

from guidance import cfg_combine, classifier_guided_score, diffusion_dpo_loss, dps_guidance, drop_labels

D = torch.float64
M = torch.tensor([[-2.0, 0.0], [2.0, 1.0]], dtype=D)  # two classes, equal priors
VAR = 0.6  # per-class variance of the *noisy* marginal at some fixed t


def class_score(x, y):
    return -(x - M[y]) / VAR


def class_loglik(x):  # (B, 2) log N(x; M_c, VAR I) up to a shared constant
    return -((x[:, None, :] - M[None]) ** 2).sum(-1) / (2 * VAR)


def uncond_score(x):
    w = torch.softmax(class_loglik(x), 1)  # p(c | x)
    return (w[:, :, None] * (-(x[:, None, :] - M[None]) / VAR)).sum(1)


def classifier_log_prob(x, y):
    return torch.log_softmax(class_loglik(x), 1).gather(1, y[:, None])[:, 0]


def test_classifier_guidance_scale_one_gives_conditional_score():
    x, y = torch.randn(20, 2, dtype=D) * 2, torch.randint(0, 2, (20,))
    with torch.no_grad():
        out = classifier_guided_score(uncond_score(x), classifier_log_prob, x, y, 1.0)
    torch.testing.assert_close(out, class_score(x, y))


def test_cfg_equals_classifier_guidance_with_exact_scores():
    x, y = torch.randn(20, 2, dtype=D) * 2, torch.randint(0, 2, (20,))
    for w in (0.0, 1.0, 3.5):
        cfg = cfg_combine(class_score(x, y), uncond_score(x), w)
        cg = classifier_guided_score(uncond_score(x), classifier_log_prob, x, y, w)
        torch.testing.assert_close(cfg, cg)


def test_drop_labels():
    y = torch.zeros(100_000, dtype=torch.long)
    out = drop_labels(y, 0.2, 10, torch.Generator().manual_seed(0))
    assert abs((out == 10).float().mean().item() - 0.2) < 0.01
    assert ((out == 0) | (out == 10)).all() and (y == 0).all()


def test_dps_guidance_linear_case():
    c = 0.7
    A = torch.tensor([[1.0, 0.0]], dtype=D)  # observe first coordinate
    x = torch.randn(5, 2, dtype=D)
    ymeas = torch.randn(5, 1, dtype=D)
    g = dps_guidance(x, lambda x: c * x, ymeas, lambda z: z @ A.T, scale=0.3)
    ref = 0.3 * 2 * c * (ymeas - c * x @ A.T) @ A
    torch.testing.assert_close(g, ref)


def test_diffusion_dpo_loss():
    ew, erw, el, erl = (torch.rand(8, dtype=D) for _ in range(4))
    ref = -F.logsigmoid(-2.0 * ((ew - erw) - (el - erl))).mean()
    torch.testing.assert_close(diffusion_dpo_loss(ew, erw, el, erl, 2.0), ref)
    # lowering the error on the preferred sample lowers the loss
    assert diffusion_dpo_loss(ew - 0.1, erw, el, erl, 2.0) < diffusion_dpo_loss(ew, erw, el, erl, 2.0)
