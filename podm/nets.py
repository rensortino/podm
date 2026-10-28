"""Small networks for 2D toys and 28x28 images.

Every net takes ``(x, t)`` (plus optional class labels ``y``) where ``t`` is a
float tensor of shape (B,). Scale ``t`` however your exercise needs (e.g.
t in [0, 1], an integer step, or log(sigma)); the sinusoidal embedding copes
with any range, but values of order 1-1000 work best with the default ``scale``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().reshape(-1) * self.scale
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TimeMLP(nn.Module):
    """MLP for low-dimensional data: f(x, t[, y]) -> R^out_dim.

    Args:
        dim: data dimension.
        out_dim: output dimension (defaults to dim: eps/score/velocity/x0 prediction).
        num_classes: if set, adds a label embedding; label ``num_classes`` is the
            "null" label used for classifier-free guidance.
        t_scale: multiplier applied to t before the sinusoidal embedding.
            Use 1000 for t in [0, 1], 1 for integer steps in [0, 1000).
    """

    def __init__(self, dim: int = 2, hidden: int = 256, depth: int = 4, out_dim: int | None = None,
                 num_classes: int | None = None, t_scale: float = 1000.0, emb_dim: int = 64):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmbedding(emb_dim, t_scale), nn.Linear(emb_dim, hidden), nn.SiLU())
        self.y_emb = nn.Embedding(num_classes + 1, hidden) if num_classes is not None else None
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, out_dim or dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, y: torch.Tensor | None = None):
        h = self.inp(x)
        cond = 0
        if t is not None:
            cond = cond + self.t_emb(t.expand(x.shape[0]) if t.dim() == 0 else t)
        if self.y_emb is not None and y is not None:
            cond = cond + self.y_emb(y)
        for block in self.blocks:
            h = h + block(h + cond)
        return self.out(h)


class MLP(nn.Module):
    """Plain MLP without time conditioning (encoders, decoders, energies, classifiers)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.cond = nn.Linear(cond_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(c)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """~2M-param UNet for (B, C, 28, 28) images. 28 -> 14 -> 7 and back."""

    def __init__(self, in_ch: int = 1, out_ch: int | None = None, base: int = 64,
                 num_classes: int | None = None, t_scale: float = 1000.0):
        super().__init__()
        cdim = base * 4
        self.t_emb = nn.Sequential(SinusoidalEmbedding(base, t_scale), nn.Linear(base, cdim), nn.SiLU(),
                                   nn.Linear(cdim, cdim))
        self.y_emb = nn.Embedding(num_classes + 1, cdim) if num_classes is not None else None
        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.d1 = _ResBlock(c1, c1, cdim)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.d2 = _ResBlock(c2, c2, cdim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.mid = _ResBlock(c3, c3, cdim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2 = _ResBlock(c2 * 2, c2, cdim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1 = _ResBlock(c1 * 2, c1, cdim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, out_ch or in_ch, 3, padding=1))

    def forward(self, x, t=None, y=None):
        c = 0
        if t is not None:
            c = c + self.t_emb(t.expand(x.shape[0]) if t.dim() == 0 else t)
        if self.y_emb is not None and y is not None:
            c = c + self.y_emb(y)
        if not torch.is_tensor(c):
            c = torch.zeros(x.shape[0], self.t_emb[-1].out_features, device=x.device)
        h1 = self.d1(self.inp(x), c)
        h2 = self.d2(self.down1(h1), c)
        h = self.mid(self.down2(h2), c)
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), c)
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), c)
        return self.out(h)


class EMA:
    """Exponential moving average of a model's parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy

        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p, 1 - self.decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
