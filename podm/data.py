"""Toy 2D distributions and MNIST.

All data is roughly zero-centred with unit-ish scale, which is what the
diffusion schedules in the exercises assume.
"""

import math
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOY_DATASETS = ("moons", "8gaussians", "swissroll", "checkerboard", "rings")


def sample_2d(name: str, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw n samples (n, 2) from a named toy distribution."""
    g = generator
    if name == "moons":
        t = torch.rand(n, generator=g) * math.pi
        upper = torch.rand(n, generator=g) < 0.5
        x = torch.where(upper, torch.cos(t), 1 - torch.cos(t))
        y = torch.where(upper, torch.sin(t), 0.5 - torch.sin(t))
        pts = torch.stack([x - 0.5, y - 0.25], dim=1) * 1.5
        return pts + 0.05 * torch.randn(n, 2, generator=g)
    if name == "8gaussians":
        angles = torch.randint(0, 8, (n,), generator=g) * (2 * math.pi / 8)
        centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.0
        return centers + 0.15 * torch.randn(n, 2, generator=g)
    if name == "swissroll":
        t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=g))
        pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) / 5.0
        return pts + 0.05 * torch.randn(n, 2, generator=g)
    if name == "checkerboard":
        x1 = torch.rand(n, generator=g) * 4 - 2
        x2 = torch.rand(n, generator=g) - torch.randint(0, 2, (n,), generator=g) * 2.0
        x2 = x2 + (torch.floor(x1) % 2)
        return torch.stack([x1, x2], dim=1)
    if name == "rings":
        r = torch.randint(1, 4, (n,), generator=g).float() * 0.7
        a = torch.rand(n, generator=g) * 2 * math.pi
        pts = torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)
        return pts + 0.05 * torch.randn(n, 2, generator=g)
    raise ValueError(f"unknown dataset {name!r}; choose from {TOY_DATASETS}")


def sample_2d_labeled(name: str, n: int, generator: torch.Generator | None = None):
    """Labeled toy data for conditional generation: returns (x, y).

    Only '8gaussians' (8 classes) and 'moons' (2 classes) are supported.
    """
    g = generator
    if name == "8gaussians":
        y = torch.randint(0, 8, (n,), generator=g)
        angles = y * (2 * math.pi / 8)
        centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.0
        return centers + 0.15 * torch.randn(n, 2, generator=g), y
    if name == "moons":
        t = torch.rand(n, generator=g) * math.pi
        y = (torch.rand(n, generator=g) < 0.5).long()
        upper = y == 0
        x = torch.where(upper, torch.cos(t), 1 - torch.cos(t))
        yy = torch.where(upper, torch.sin(t), 0.5 - torch.sin(t))
        pts = torch.stack([x - 0.5, yy - 0.25], dim=1) * 1.5
        return pts + 0.05 * torch.randn(n, 2, generator=g), y
    raise ValueError(f"no labels for {name!r}")


def mnist_loader(batch_size: int = 128, train: bool = True, binarize: bool = False, num_workers: int = 2):
    """MNIST as (B, 1, 28, 28) tensors in [-1, 1] (or {0, 1} if binarize)."""
    from torchvision import datasets, transforms

    if binarize:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x > 0.5).float())])
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(DATA_DIR, train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                                       num_workers=num_workers, persistent_workers=num_workers > 0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TOY_DATASETS), figsize=(3 * len(TOY_DATASETS), 3))
    for ax, name in zip(axes, TOY_DATASETS):
        x = sample_2d(name, 5000)
        ax.scatter(x[:, 0], x[:, 1], s=1)
        ax.set_title(name)
        ax.set_aspect("equal")
    out = Path("runs/toy_datasets.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
