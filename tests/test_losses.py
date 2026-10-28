"""损失项的数值自检（需要 PyTorch）：对精确解，每一项都应为 0。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch

torch = pytest.importorskip("torch")

from nnschrodinger.physics.spectrum import get_reference  # noqa: E402
from nnschrodinger.solver.losses import LossWeights, Losses, grad_balance_norm, total_loss  # noqa: E402


def _exact_model(ref, x_grid: np.ndarray):
    """把精确解做成可求值、可求导的“理想模型”（分段线性插值，保证二阶导存在）。"""
    vals = ref.psi(x_grid)

    def model(x):
        idx = (x - x_grid[0]) / (x_grid[1] - x_grid[0])
        i0 = torch.clamp(idx.floor().long(), 0, len(x_grid) - 2)
        w = idx - i0.double()
        t = torch.tensor(vals, dtype=torch.float64)
        return ((1 - w) * t[i0] + w * t[i0 + 1]).reshape(-1, 1)

    return model


@requires_torch
def test_normalization_and_boundary_zero_for_exact_state():
    # ψ = √2·sin(πx) 才是 L2 归一化的基态（∫|ψ|²dx = 1）
    x_grid = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    losses = Losses(dx=float(x_grid[1] - x_grid[0]))
    model = lambda x: np.sqrt(2.0) * torch.sin(np.pi * x).reshape(-1, 1)  # noqa: E731
    value = float(losses.normalization(model, x_grid.reshape(-1, 1)))
    assert value < 1e-12
    boundary = float(losses.boundary(model, torch.tensor([[0.0], [1.0]], dtype=torch.float64)))
    assert boundary < 1e-12


def test_residual_vanishes_for_exact_solution():
    """ψ = π^(-1/4)·e^(-x²/2) 是谐振子基态（E=0.5），残差应严格为 0。"""
    x = torch.linspace(-6.0, 6.0, 1501, dtype=torch.float64).reshape(-1, 1)
    losses = Losses()
    scale = np.pi ** -0.25
    model = lambda t: torch.exp(-0.5 * t**2) * scale  # noqa: E731
    r = float(losses.residual(model, x, 0.5, lambda t: 0.5 * t**2))
    assert r < 1e-12


def test_energy_expectation_matches_reference():
    ref = get_reference("harmonic", 0)
    x = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float64).reshape(-1, 1)
    losses = Losses(dx=float(x[1, 0] - x[0, 0]))
    scale = float(1.0 / np.sqrt(np.trapezoid(ref.psi(x[:, 0].numpy()) ** 2, x[:, 0].numpy())))
    model = lambda t: torch.exp(-0.5 * t**2) * scale  # noqa: E731
    e, _ = losses.energy(model, x, lambda t: 0.5 * t**2)
    assert abs(float(e) - 0.5) < 1e-3

def test_compute_weighting_and_skips():
    losses = Losses(LossWeights(residual=2.0, normalization=0.0))
    a = torch.tensor(1.0, dtype=torch.float64)
    b = torch.tensor(3.0, dtype=torch.float64)
    breakdown = losses.compute(residual=a, normalization=b, boundary=None)
    assert breakdown.total == pytest.approx(2.0)
    assert "normalization" not in breakdown.terms
    assert "boundary" not in breakdown.terms


@requires_torch
def test_total_loss_requires_a_term():
    losses = Losses(LossWeights(residual=0.0))
    with pytest.raises(ValueError):
        total_loss(losses, residual=torch.tensor(0.0, dtype=torch.float64))


@requires_torch
def test_orthogonality():
    losses = Losses(dx=1.0 / 1000)
    x = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64).reshape(-1, 1)
    f1 = lambda t: torch.sin(np.pi * t) * np.sqrt(2.0)  # noqa: E731
    f2 = lambda t: torch.sin(2 * np.pi * t) * np.sqrt(2.0)  # noqa: E731
    assert float(losses.orthogonality([f1, f2], x)) < 1e-6
    assert float(losses.orthogonality([f1], x)) == 0.0


@requires_torch
def test_grad_balance_norm_shape():
    net = torch.nn.Linear(1, 1, dtype=torch.float64)
    x = torch.rand(20, 1, dtype=torch.float64)
    a = (net(x) ** 2).sum()
    b = (net(x) ** 4).sum()
    coeffs = grad_balance_norm([a, b], net)
    assert len(coeffs) == 2
    assert all(c > 0 for c in coeffs)


def test_energy_projected_preserves_derivative():
    """正交投影只应作用于 ψ，不能作用于 ψ′。

    这是一个真实的回归：曾经把 ψ′ 也用 ψ 的基投影，导致 dψ/dx 在中点被削掉一半
    （1.062 -> 0.531），Rayleigh 商算成 1.25 而不是 1.5。
    """
    import math as _math

    n = 1200
    gl = np.linspace(-6.0, 6.0, n)
    h = gl[1] - gl[0]
    dx = 12.0 / n
    psi0 = get_reference("harmonic", 0).psi(gl)
    psi0 = psi0 / _math.sqrt(float(np.sum(psi0**2)) * h)
    basis = torch.tensor(psi0.reshape(-1, 1), dtype=torch.float64)

    def projector(v):
        return v - basis @ (basis.T @ v * dx)

    x = torch.tensor(gl.reshape(-1, 1), dtype=torch.float64).requires_grad_(True)
    # 解析 ψ₁ ∝ x·e^{-x²/2}
    model = lambda t: t * torch.exp(-0.5 * t**2)  # noqa: E731
    losses = Losses(dx=dx)
    energy, _ = losses.energy_projected(model, x, lambda t: 0.5 * t**2, projector)
    # 对任意整体缩放，Rayleigh 商都应等于精确值 1.5
    assert abs(float(energy.detach()) - 1.5) < 1e-6


def test_energy_projected_rejects_lower_state():
    """投影掉基态后，基态本身的 Rayleigh 商应退化成无穷大（分母 -> 0）。"""
    n = 800
    gl = np.linspace(-6.0, 6.0, n)
    h = gl[1] - gl[0]
    dx = 12.0 / n
    psi0 = get_reference("harmonic", 0).psi(gl)
    psi0 = psi0 / np.sqrt(float(np.sum(psi0**2)) * h)
    basis = torch.tensor(psi0.reshape(-1, 1), dtype=torch.float64)

    def projector(v):
        return v - basis @ (basis.T @ v * dx)

    x = torch.tensor(gl.reshape(-1, 1), dtype=torch.float64).requires_grad_(True)
    model = lambda t: torch.exp(-0.5 * t**2)  # noqa: E731
    losses = Losses(dx=dx)
    _, norm = losses.energy_projected(model, x, lambda t: 0.5 * t**2, projector)
    # 投影基用离散求积构造，并非严格单位正交（含 O(1/N) 的求积误差），
    # 因此残差是 1e-3 量级而不是 1e-9。这里只要求它比未投影时（≈1）小 2 个量级。
    assert abs(float(norm.detach())) < 1e-2

@requires_torch
def test_discrete_terms_energy_has_a_lower_bound_with_nonzero_endpoints():
    """回归（工程记录 18）：端点非零时，Rayleigh 商必须仍有下界。

    旧实现把含非零端点的 ψ 直接代入二阶差分：二次型丢掉边界项，
    V=0 的无限深阱也能算出**负能量**，最小化能量项会把训练推向放大端点。
    正确做法是**只对能量项**用 Dirichlet 延拓（端点置零）；残差仍用真实
    端点——若残差也置零，会在边界制造人为跳变，残差压不下去。
    """
    from nnschrodinger.solver.certify import discrete_rayleigh_quotient
    from nnschrodinger.solver.losses import discrete_dirichlet_terms

    x = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    h = float(x[1] - x[0])
    v = torch.zeros_like(x)
    rng = np.random.default_rng(0)
    psi_np = np.sin(2.0 * np.pi * x.numpy()) + 1e-3 * rng.normal(size=x.numel())
    psi_np[0], psi_np[-1] = 3.0, -2.0  # 故意让边界项很大
    psi = torch.tensor(psi_np)

    terms = discrete_dirichlet_terms(psi, v, h, 1.0)
    energy = float(terms["energy"])
    # λ0(H_h) = 4.9348...；端点置零后的 Rayleigh 商不可能低于它
    assert energy >= 4.9
    # 与 numpy 证书口径（同样零端点）完全一致
    assert energy == pytest.approx(
        discrete_rayleigh_quotient(psi_np, x.numpy(), np.zeros_like(x.numpy())), rel=1e-12
    )
    assert float(terms["boundary"]) == pytest.approx(13.0)
    # 残差仍用**真实端点**：与手工 raw-endpoint 差分逐点一致
    inner = psi[1:-1]
    lap_raw = (psi[2:] - 2.0 * inner + psi[:-2]) / h**2
    r_raw = (-0.5 * lap_raw + v[1:-1] * inner) - 1.0 * inner
    assert float(terms["residual"]) == pytest.approx(float(torch.mean(r_raw**2)), rel=1e-12)

    # 旧写法（直接用非零端点作差分）在同一 ψ 上给出负能量——这正是 bug
    p = psi.reshape(-1)
    inner = p[1:-1]
    lap = (p[2:] - 2.0 * inner + p[:-2]) / h**2
    norm2 = h * torch.sum(inner**2)
    old_energy = float(h * torch.sum(inner * (-0.5 * lap)) / norm2)
    assert old_energy < 0.0
