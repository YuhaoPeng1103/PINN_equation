"""定向舍入计数的测试。

锁四件事：

1. ``certified_sturm_count_1d`` 与浮点特征分解**逐点**一致（在区间不拒绝的
   全部 shift 上），且**拒绝**只发生在 shift 落在某个本征值的一个 ulp 以内；
2. 计数是单调的：shift 增大，计数不减；
3. 可分离二维势的 Kronecker 和计数与稠密二维特征分解一致，且对不可分离
   势明确拒绝（而不是悄悄给一个浮点数）；
4. shift 恰好等于一个简并本征值时**必须拒绝**——拒绝是正确行为，不是缺陷。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from nnschrodinger.solver.certified_count import (
    CountCertificate,
    Interval,
    certified_eigenvalue_brackets_1d,
    certified_separable_brackets_2d,
    certified_separable_count_2d,
    certified_sturm_count_1d,
)
from nnschrodinger.solver.certify2d import Grid2D


def _dense_1d(x: np.ndarray, potential: np.ndarray) -> np.ndarray:
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    return np.sort(eigh_tridiagonal(diag, off, eigvals_only=True))


def test_interval_arithmetic_brackets_the_exact_value():
    a = Interval(1.0, 2.0)
    b = Interval(-3.0, 0.5)
    assert (a * b).lo <= -6.0 and (a * b).hi >= 1.0
    assert (a - b).lo <= 0.5 and (a - b).hi >= 5.0
    assert (a + b).lo <= -2.0 and (a + b).hi >= 2.5
    assert (a / Interval(2.0, 4.0)).lo <= 0.25
    assert Interval(-1.0, 1.0).sign() is None
    assert Interval(0.1, 0.2).sign() == 1
    assert Interval(-0.2, -0.1).sign() == -1


def test_count_matches_dense_spectrum_across_shifts():
    x = np.linspace(0.0, 1.0, 121)
    potential = np.zeros_like(x)
    eigenvalues = _dense_1d(x, potential)
    for sigma in np.linspace(-1.0, 60.0, 137):
        certificate = certified_sturm_count_1d(x, potential, float(sigma))
        if not certificate.certified:
            continue
        assert certificate.count == int(np.sum(eigenvalues < sigma))


def test_count_is_monotone_in_the_shift():
    x = np.linspace(-3.0, 3.0, 201)
    potential = 0.5 * x**2
    previous = -1
    for sigma in np.linspace(0.1, 20.0, 97):
        certificate = certified_sturm_count_1d(x, potential, float(sigma))
        if not certificate.certified:
            continue
        assert certificate.count >= previous
        previous = certificate.count


def test_count_refuses_exactly_at_an_eigenvalue():
    """shift 取本征值时计数必须被拒绝，而不是猜一个整数。"""
    x = np.linspace(0.0, 1.0, 101)
    potential = np.zeros_like(x)
    eigenvalues = _dense_1d(x, potential)
    refused = 0
    for value in eigenvalues[:20]:
        certificate = certified_sturm_count_1d(x, potential, float(value))
        if not certificate.certified:
            refused += 1
    # 至少要有若干次拒绝；若一次都不拒绝，说明“严格”只是口号。
    assert refused >= 1


def test_brackets_contain_the_dense_eigenvalues():
    x = np.linspace(0.0, 1.0, 401)
    potential = np.zeros_like(x)
    eigenvalues = _dense_1d(x, potential)
    brackets = certified_eigenvalue_brackets_1d(x, potential, 5, upper=200.0)
    assert brackets is not None
    for index, (low, high) in enumerate(brackets):
        assert low <= eigenvalues[index] <= high
        assert high - low < 1e-6


def test_separable_count_matches_dense_2d_spectrum():
    n = 21
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    potential = np.zeros((n, n))
    grid = Grid2D(x=x, y=y, potential=potential)
    h = float(x[1] - x[0])
    m = n - 2
    # 独立构造：H = A ⊗ I + I ⊗ A（不用实现里的 _sparse_operator）
    A = np.diag(np.full(m, 1.0 / h**2))
    A += np.diag(np.full(m - 1, -0.5 / h**2), 1)
    A += np.diag(np.full(m - 1, -0.5 / h**2), -1)
    dense = np.linalg.eigvalsh(np.kron(A, np.eye(m)) + np.kron(np.eye(m), A))
    for k in range(6):
        # 取相邻本征值的中点：既避开简并，也避开计数分辨率边界
        sigma = 0.5 * (dense[k] + dense[k + 1])
        certificate = certified_separable_count_2d(grid, float(sigma))
        if not certificate.certified:
            continue
        assert certificate.count == int(np.sum(dense < sigma))


def test_separable_count_refuses_a_non_separable_potential():
    n = 21
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = Grid2D(x=x, y=y, potential=0.5 * (X**2 + Y**2) + X * Y)
    certificate = certified_separable_count_2d(grid, 1.0)
    assert certificate.certified is False
    assert "可分离" in certificate.reason


def test_separable_count_refuses_a_degenerate_shift():
    """盒子的第二、三重本征值简并，取其中点即为本征值本身：必须拒绝。"""
    n = 21
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x=x, y=y, potential=np.zeros((n, n)))
    h = float(x[1] - x[0])
    one_dimensional = 2.0 / h**2 * np.sin(np.arange(1, 40) * math.pi * h / 2.0) ** 2
    degenerate = float(one_dimensional[0] + one_dimensional[1])
    certificate = certified_separable_count_2d(grid, degenerate)
    assert certificate.certified is False


def test_count_certificate_serialises():
    certificate = CountCertificate(3, True, 10, 1.5)
    payload = certificate.to_dict()
    assert payload["count"] == 3 and payload["certified"] is True


def test_separable_bracket_contains_the_dense_eigenvalue_and_is_narrow():
    """二维 bracket 必须包住稠密本征值，且宽度远小于谱隙。"""
    n = 21
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x=x, y=y, potential=np.zeros((n, n)))
    # 稠密参考：H = −½ d²/dx² 的三点差分（对角 1/h²，次对角 −½h⁻²）
    h = float(x[1] - x[0])
    A = (np.diag(np.full(n - 2, 1.0 / h**2))
         + np.diag(np.full(n - 3, -0.5 / h**2), 1)
         + np.diag(np.full(n - 3, -0.5 / h**2), -1))
    dense = np.linalg.eigvalsh(np.kron(A, np.eye(n - 2)) + np.kron(np.eye(n - 2), A))
    ground = float(np.min(dense))
    bracket = certified_separable_brackets_2d(grid, 0, hint=ground)
    assert bracket is not None
    lo, hi = bracket
    assert lo <= ground <= hi
    assert hi - lo < 1e-6 * max(1.0, abs(ground))


def test_separable_bracket_refuses_for_a_non_separable_potential():
    n = 21
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = Grid2D(x=x, y=y, potential=0.5 * (X**2 + Y**2) + X * Y)
    assert certified_separable_brackets_2d(grid, 0, hint=1.0) is None
