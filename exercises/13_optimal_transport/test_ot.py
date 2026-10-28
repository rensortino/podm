import torch

from ot import monotone_map_1d, path_straightness, transport_cost


def test_monotone_map_is_optimal_among_permutations():
    torch.manual_seed(0)
    x, y = torch.randn(200), torch.rand(200) * 4
    Tx = monotone_map_1d(x, y)
    assert sorted(Tx.tolist()) == sorted(y.tolist())
    assert (torch.argsort(x) == torch.argsort(Tx)).all()  # order preserving
    best = transport_cost(x[:, None], Tx[:, None])
    for _ in range(20):
        assert best <= transport_cost(x[:, None], y[torch.randperm(200)][:, None])


def test_path_straightness():
    t = torch.linspace(0, 1, 11)[:, None, None]
    a, b = torch.randn(1, 5, 2), torch.randn(1, 5, 2)
    assert path_straightness(a + t * (b - a)).abs() < 1e-5
    bent = a + t * (b - a) + torch.sin(torch.pi * t) * torch.tensor([1.0, 0.0])
    assert path_straightness(bent) > 0.1
