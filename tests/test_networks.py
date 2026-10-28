"""网络与谱方法基线（需要 PyTorch；无 torch 时自动跳过）。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch

torch = pytest.importorskip("torch")

from nnschrodinger.solver.networks import SpectralSolver, build_hhl_network, build_mlp  # noqa: E402


@requires_torch
def test_mlp_shape_and_dtype():
    net = build_mlp(width=16, depth=3, seed=0)
    x = torch.zeros(5, 1, dtype=torch.float64)
    y = net(x)
    assert y.shape == (5, 1)
    assert y.dtype == torch.float64


@requires_torch
def test_mlp_is_differentiable_twice():
    net = build_mlp(width=8, depth=2, seed=0)
    x = torch.rand(10, 1, dtype=torch.float64, requires_grad=True)
    y = net(x).sum()
    (dy,) = torch.autograd.grad(y, x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    assert torch.isfinite(d2y).all()


@requires_torch
def test_fourier_features_shape():
    net = build_mlp(width=16, depth=2, fourier=True, seed=0)
    out = net(torch.rand(4, 1, dtype=torch.float64))
    assert out.shape == (4, 1)


@requires_torch
def test_hhl_network_io():
    net = build_hhl_network(hidden=10, n_layers=3, out_dim=2)
    out = net(torch.rand(7, 2, dtype=torch.float64))
    assert out.shape == (7, 2)


def test_spectral_solver_harmonic():
    solver = SpectralSolver(-6.0, 6.0, lambda x: 0.5 * x**2, n_states=4, n_points=2001)
    for n in range(4):
        assert abs(solver.energy(n) - (n + 0.5)) < 1e-2
    x = np.linspace(-6, 6, 2001)
    psi = solver.psi(x, 0)
    assert float(np.trapezoid(psi**2, x)) == pytest.approx(1.0, abs=1e-6)


def test_spectral_solver_well():
    solver = SpectralSolver(0.0, 1.0, lambda x: np.zeros_like(x), energy_window=(0.0, 200.0), n_states=3)
    exact = [n * n * np.pi**2 / 2 for n in (1, 2, 3)]
    # 用“阱外 V₀ 有限”近似无限深阱：误差 ≈ E/V₀ + 离散误差，n=3 时约 0.11
    for n, e in enumerate(exact):
        assert abs(solver.energy(n) - e) < 0.2
