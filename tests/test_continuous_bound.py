"""连续算子变分上界的测试。

核心是一条定理的数值化：对任意 ``u ∈ H¹_0``，``λ₁ ≤ R(u)``。测试锁住

1. 解析试探函数（盒子的 ``sin(πx)``）得到精确的 Rayleigh 商；
2. 网格本征向量的 P1 插值给出的上界**高于**真值，且随 ``h → 0`` 以二阶
   收敛；
3. 势能项求积不精确时（势在单元内有扭折且未给断点）**必须被拒绝**；
4. 二维 Q1 上界在非正方形网格上与两条一维上界之和一致；
5. ``box_lower_bound_1d`` 给出的离散值确实是连续值的下界且单调上升。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh_tridiagonal

from nnschrodinger.solver.continuous_bound import (
    box_lower_bound_1d,
    p1_rayleigh_bound,
    q1_rayleigh_bound_2d,
)


def _ground_vector(x: np.ndarray, potential: np.ndarray) -> np.ndarray:
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    _, vectors = eigh_tridiagonal(diag, off, eigvals_only=False)
    return vectors[:, 0]


def test_sine_trial_function_reproduces_the_box_energy():
    """u = sin(πx) 是盒子基态本身，Rayleigh 商应为 π²/2。"""
    x = np.linspace(0.0, 1.0, 401)
    psi = np.sin(math.pi * (x[1:-1] - 0.0))
    bound = p1_rayleigh_bound(x, lambda t: np.zeros_like(t), psi)
    # P1 插值的 sin 略高于真 sin，商略高于 π²/2，但差值是 O(h²)
    assert bound.upper >= math.pi**2 / 2.0 - 1e-12
    assert bound.upper - math.pi**2 / 2.0 < 1e-3
    assert bound.exact_quadrature


def test_bound_dominates_the_true_energy_and_converges_at_second_order():
    truth = 0.5
    widths = []
    for n in (201, 401, 801):
        x = np.linspace(-6.0, 6.0, n)
        potential = 0.5 * x**2
        psi = _ground_vector(x, potential)
        bound = p1_rayleigh_bound(x, lambda t: 0.5 * t**2, psi)
        assert bound.upper >= truth
        widths.append(bound.upper - truth)
    # h 减半，误差应下降约 4 倍
    for coarse, fine in zip(widths[:-1], widths[1:]):
        assert 3.0 < coarse / fine < 5.0


def test_quadrature_check_rejects_a_kinked_potential():
    """|x| 在含原点的单元上不是多项式；不给断点时求积不精确，必须拒绝。"""
    n = 401
    base = np.linspace(-1.0, 1.0, n)
    x = base + 0.5 * float(base[1] - base[0])  # 让扭折落在单元中点
    psi = _ground_vector(x, np.abs(x))
    bound = p1_rayleigh_bound(x, lambda t: np.abs(t), psi)
    assert bound.exact_quadrature is False
    assert "Gauss" in bound.reason


def test_kinked_potential_is_accepted_when_the_kink_is_a_breakpoint():
    n = 401
    base = np.linspace(-1.0, 1.0, n)
    x = base + 0.5 * float(base[1] - base[0])
    psi = _ground_vector(x, np.abs(x))
    bound = p1_rayleigh_bound(x, lambda t: np.abs(t), psi, breakpoints=(0.0,))
    assert bound.exact_quadrature is True


def test_two_dimensional_quadrature_check_rejects_a_non_polynomial():
    """二维也必须自查：非多项式势使两条 Gauss 规则给出不相交的区间。"""
    nx, ny = 41, 41
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    psi = np.outer(np.sin(math.pi * y[1:-1]), np.sin(math.pi * x[1:-1]))
    kinked = lambda X, Y: np.sqrt(np.abs(X - 0.37)) + 0.1 * Y
    bound = q1_rayleigh_bound_2d(x, y, kinked, psi)
    assert bound.exact_quadrature is False
    assert bound.upper == math.inf  # 拒绝时不返回一个可能被破坏的上界
    assert "Gauss" in bound.reason


def test_two_dimensional_quadrature_accepts_the_quartic_anharmonic():
    """V ~ x²y² 在单元内是低次多项式，3 点与 5 点 Gauss 都必须精确。"""
    x = np.linspace(-3.0, 3.0, 41)
    y = np.linspace(-3.0, 3.0, 41)
    psi = np.outer(np.exp(-0.5 * y[1:-1] ** 2), np.exp(-0.5 * x[1:-1] ** 2))
    potential = lambda X, Y: 0.5 * (X**2 + Y**2) + 0.5 * X**2 * Y**2
    bound = q1_rayleigh_bound_2d(x, y, potential, psi)
    assert bound.exact_quadrature is True
    assert math.isfinite(bound.upper)


def test_two_dimensional_q1_bound_matches_the_tensor_product_on_a_rectangle():
    """可分离势 + 矩形网格：Q1 上界应等于两条一维上界之和。"""
    nx, ny = 41, 61
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    sine_x = np.sin(math.pi * x[1:-1])
    sine_y = np.sin(math.pi * y[1:-1])
    psi = np.outer(sine_y, sine_x)  # 行对应 y，列对应 x
    bound_2d = q1_rayleigh_bound_2d(x, y, lambda X, Y: np.zeros_like(X), psi)
    bound_x = p1_rayleigh_bound(x, lambda t: np.zeros_like(t), sine_x)
    bound_y = p1_rayleigh_bound(y, lambda t: np.zeros_like(t), sine_y)
    assert abs(bound_2d.upper - (bound_x.upper + bound_y.upper)) < 1e-9


def test_two_dimensional_bound_dominates_pi_squared():
    nx, ny = 41, 61
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    psi = np.outer(np.sin(math.pi * y[1:-1]), np.sin(math.pi * x[1:-1]))
    bound = q1_rayleigh_bound_2d(x, y, lambda X, Y: np.zeros_like(X), psi)
    assert bound.upper >= math.pi**2
    assert bound.upper - math.pi**2 < 1e-2


def test_box_lower_bound_is_below_the_continuum_and_increases():
    previous = -1.0
    for n in (21, 41, 81, 161):
        h = 1.0 / (n - 1)
        value = box_lower_bound_1d(h, 0)
        assert value <= math.pi**2 / 2.0
        assert value >= previous
        previous = value
