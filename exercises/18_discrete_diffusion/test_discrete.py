import torch
import torch.nn.functional as F

from discrete import mask_tokens, masked_diffusion_loss, sample_masked

K, MASK = 5, 5


def test_mask_tokens_rate_and_identity():
    x0 = torch.randint(0, K, (4, 50_000))
    t = torch.tensor([0.0, 0.25, 0.7, 1.0])
    xt = mask_tokens(x0, t, MASK, torch.Generator().manual_seed(0))
    rates = (xt == MASK).float().mean(1)
    assert torch.allclose(rates, t, atol=0.01)
    keep = xt != MASK
    assert (xt[keep] == x0[keep]).all()


def test_masked_diffusion_loss():
    torch.manual_seed(0)
    x0 = torch.randint(0, K, (3, 7))
    xt = mask_tokens(x0, torch.tensor([0.5, 0.5, 0.9]), MASK)
    logits = torch.randn(3, 7, K)
    t = torch.tensor([0.5, 0.5, 0.9])
    nll = -F.log_softmax(logits, -1).gather(-1, x0[..., None])[..., 0]
    ref = ((nll * (xt == MASK)).sum(1) / t).mean()
    torch.testing.assert_close(masked_diffusion_loss(logits, x0, xt, t, MASK), ref)
    # unmasked positions must not contribute
    logits2 = logits.clone()
    logits2[xt != MASK] = 100 * torch.randn_like(logits2[xt != MASK])
    torch.testing.assert_close(masked_diffusion_loss(logits2, x0, xt, t, MASK), ref)


def test_sampler_with_oracle_for_iid_tokens():
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    model = lambda x: torch.log(p).expand(*x.shape, K)  # noqa: E731
    x = sample_masked(model, (200, 100), 20, MASK, torch.Generator().manual_seed(0))
    assert (x != MASK).all()
    freq = torch.bincount(x.flatten(), minlength=K).float() / x.numel()
    assert torch.allclose(freq, p, atol=0.01)
