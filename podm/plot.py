"""Plotting helpers. Everything writes to disk (runs/ is git-ignored)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _out(path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = RUNS / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def scatter2d(samples: dict[str, torch.Tensor], path, lim: float = 3.5, s: float = 1.0):
    """Side-by-side scatter plots: scatter2d({"data": x, "model": y}, "ddpm/samples.png")."""
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4), squeeze=False)
    for ax, (name, x) in zip(axes[0], samples.items()):
        x = x.detach().cpu()
        ax.scatter(x[:, 0], x[:, 1], s=s, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(name)
    out = _out(path)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def trajectories2d(traj: torch.Tensor, path, n_show: int = 200, lim: float = 3.5):
    """Plot sampling paths. traj: (steps, N, 2)."""
    traj = traj.detach().cpu()[:, :n_show]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(traj[:, :, 0], traj[:, :, 1], lw=0.4, alpha=0.5, color="gray")
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=3, label="start")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=3, label="end")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend()
    out = _out(path)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def vector_field2d(fn, path, lim: float = 3.0, n: int = 25, title: str = ""):
    """Quiver plot of fn: (N, 2) -> (N, 2), e.g. a score or velocity field at fixed t."""
    xs = torch.linspace(-lim, lim, n)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        v = fn(grid).cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], angles="xy")
    ax.set_aspect("equal")
    ax.set_title(title)
    out = _out(path)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def image_grid(x: torch.Tensor, path, nrow: int = 8):
    """Save (B, C, H, W) images in [-1, 1] as a grid."""
    from torchvision.utils import save_image

    out = _out(path)
    save_image((x.detach().cpu().clamp(-1, 1) + 1) / 2, out, nrow=nrow)
    return out
