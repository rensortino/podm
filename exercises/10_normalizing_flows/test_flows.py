import torch
from torch.func import jacrev, vmap

from flows import AffineCoupling, RealNVP


def _init(model, seed=0):
    torch.manual_seed(seed)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.double()


def test_coupling_invertible_and_logdet():
    layer = _init(AffineCoupling(3, torch.tensor([1, 0, 1])))
    x = torch.randn(10, 3, dtype=torch.float64)
    z, ld = layer(x)
    torch.testing.assert_close(layer.inverse(z), x)
    J = vmap(jacrev(lambda v: layer(v[None])[0][0]))(x)
    torch.testing.assert_close(ld, torch.linalg.slogdet(J).logabsdet)
    # masked dims are passed through unchanged
    torch.testing.assert_close(z[:, [0, 2]], x[:, [0, 2]])


def test_realnvp_roundtrip():
    flow = _init(RealNVP(2, 6, 32))
    x = torch.randn(20, 2, dtype=torch.float64)
    z, _ = flow(x)
    for layer in reversed(flow.layers):
        z = layer.inverse(z)
    torch.testing.assert_close(z, x)


def test_realnvp_density_normalises():
    flow = _init(RealNVP(2, 6, 32), seed=1)
    xs = torch.linspace(-12, 12, 600, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        mass = flow.log_prob(grid).exp().sum() * (xs[1] - xs[0]) ** 2
    assert abs(mass.item() - 1) < 0.02


def test_sample_shape():
    flow = RealNVP(2, 4, 16)
    assert flow.sample(7).shape == (7, 2)
