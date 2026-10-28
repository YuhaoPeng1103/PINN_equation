"""二维证书与二维滤波的测试。

重点锁三件事：

1. ``count_eigenvalues_below_2d`` 的惯性计数必须与**稠密特征分解**一致
   （Sylvester 定律的浮点实现，符号错了整个证书体系就错了）；
2. 精确离散本征向量必须得到有效且极窄的证书，且区间包含该本征值；
3. 简并本征态必须被**拒绝**——二维各向同性谐振子的第一激发态是二重
   简并，任何单态区间都不存在，返回 valid=False 才是正确行为。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nnschrodinger.solver.certify2d import (
    Grid2D,
    _sparse_operator,
    approximate_spectrum_2d,
    certify_eigenpair_2d,
    count_eigenvalues_below_2d,
    discrete_eigenvalue_2d,
    discrete_rayleigh_quotient_2d,
    residual_norm_2d,
)
from nnschrodinger.solver.filtering2d import (
    dst1_2d,
    filter_dirichlet_2d,
    guided_filter_2d,
    transported_angle_bound_2d,
)


def _harmonic_grid(n: int = 21, half: float = 3.0) -> Grid2D:
    x = np.linspace(-half, half, n)
    y = x.copy()
    v = 0.5 * (x[:, None] ** 2 + y[None, :] ** 2)
    return Grid2D(x, y, v)


def _dense_eigenvectors(grid: Grid2D, k: int = 3):
    matrix = _sparse_operator(grid, 0.0).toarray()
    values, vectors = np.linalg.eigh(matrix)
    return values, vectors


def _embed(vec: np.ndarray, grid: Grid2D) -> np.ndarray:
    psi = np.zeros(grid.shape)
    psi[1:-1, 1:-1] = vec.reshape(grid.shape[0] - 2, grid.shape[1] - 2)
    return psi


def test_inertia_count_matches_dense_eigenvalues() -> None:
    grid = _harmonic_grid(21)
    values, _ = _dense_eigenvectors(grid)
    for sigma in (0.5, 0.9, 1.5, 2.5, 3.5, 4.0):
        expected = int(np.sum(values < sigma))
        assert count_eigenvalues_below_2d(grid, sigma) == expected, sigma


def test_approximate_spectrum_matches_dense() -> None:
    grid = _harmonic_grid(21)
    values, _ = _dense_eigenvectors(grid)
    approx = approximate_spectrum_2d(grid, 5)
    assert np.allclose(approx, values[:5], atol=1e-8)


def test_rayleigh_and_residual_on_exact_eigenvector() -> None:
    grid = _harmonic_grid(21)
    values, vectors = _dense_eigenvectors(grid)
    psi = _embed(vectors[:, 0], grid)
    eta = discrete_rayleigh_quotient_2d(psi, grid)
    assert abs(eta - values[0]) < 1e-10
    assert residual_norm_2d(psi, grid, eta) < 1e-10
    # 缩放到任意幅度不改变 Rayleigh 商
    assert abs(discrete_rayleigh_quotient_2d(37.0 * psi, grid) - eta) < 1e-10


def test_exact_eigenvector_certificate_is_valid_and_contains_reference() -> None:
    grid = _harmonic_grid(21)
    values, vectors = _dense_eigenvectors(grid)
    psi = _embed(vectors[:, 0], grid)
    approx = values[:4]
    cert = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
    assert cert.valid
    assert cert.state_index == 0
    assert cert.temple_lower <= values[0] <= cert.temple_upper
    assert cert.width is not None and cert.width < 1e-6
    assert cert.eigenfunction_angle_bound is not None


def test_degenerate_excited_state_is_refused() -> None:
    """二维各向同性谐振子的第一激发态是二重简并，证书必须拒绝。"""
    grid = _harmonic_grid(21)
    values, vectors = _dense_eigenvectors(grid)
    assert abs(values[1] - values[2]) < 1e-10  # 前提：确实简并
    psi = _embed(vectors[:, 1], grid)
    cert = certify_eigenpair_2d(psi, grid, isolate=True, approx=values[:4])
    assert not cert.valid
    assert cert.temple_lower is None and cert.temple_upper is None
    assert "简并" in cert.reason or "单态" in cert.reason


def test_discrete_eigenvalue_2d_matches_dense() -> None:
    grid = _harmonic_grid(21)
    values, _ = _dense_eigenvectors(grid)
    approx = approximate_spectrum_2d(grid, 4)
    assert abs(discrete_eigenvalue_2d(grid, 0, approx=approx) - values[0]) < 1e-7


def test_grid_validation_rejects_bad_input() -> None:
    x = np.linspace(0.0, 1.0, 9)
    with pytest.raises(ValueError):
        Grid2D(x, np.array([0.0, 0.2, 0.5]), np.zeros((9, 3)))
    bad = x.copy()
    bad[4] += 1e-3
    with pytest.raises(ValueError):
        Grid2D(bad, x, np.zeros((9, 9)))
    with pytest.raises(ValueError):
        Grid2D(x, x, np.zeros((9, 8)))


def test_dst1_2d_is_involution() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal((9, 6))
    assert np.allclose(dst1_2d(dst1_2d(values)), values, atol=1e-12)


def test_filter_with_full_cutoff_keeps_interior() -> None:
    grid = _harmonic_grid(15)
    rng = np.random.default_rng(5)
    psi = rng.standard_normal(grid.shape)
    full = filter_dirichlet_2d(psi, grid, grid.shape[0] - 2, grid.shape[1] - 2)
    assert np.allclose(full[1:-1, 1:-1], psi[1:-1, 1:-1], atol=1e-10)
    assert np.allclose(full[0, :], 0.0) and np.allclose(full[:, -1], 0.0)


def test_guided_filter_recovers_exact_well_mode() -> None:
    """方阱基态正是 (1,1) 模式：加噪声后滤波应回到机器精度。"""
    n = 31
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))
    xx, yy = np.meshgrid(x, x, indexing="ij")
    exact = 2.0 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
    rng = np.random.default_rng(0)
    noisy = exact + 2e-3 * np.sin(3 * math.pi * xx) * np.sin(5 * math.pi * yy)
    noisy += 1e-4 * rng.standard_normal((n, n))
    approx = approximate_spectrum_2d(grid, 4)
    result = guided_filter_2d(noisy, grid, approx=approx, refine=False,
                              pairs=[(1, 1), (2, 2), (3, 3)])
    assert result.certificate.valid
    assert result.cutoff == (1, 1)
    assert result.certificate.residual_norm < 1e-9


def test_transported_angle_bound_covers_true_angle_2d() -> None:
    """方阱基态是精确的 (1,1) 模式，加了可分辨污染后搬运界仍成立。"""
    n = 31
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))
    xx, yy = np.meshgrid(x, x, indexing="ij")
    phi = 2.0 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
    psi = phi + 5e-3 * np.sin(3 * math.pi * xx) * np.sin(5 * math.pi * yy)

    result = guided_filter_2d(psi, grid, refine=False)
    assert result.certificate.valid
    assert result.cutoff == (1, 1)
    bound = transported_angle_bound_2d(
        psi, grid, result.psi, result.certificate.eigenfunction_angle_bound
    )

    a = psi[1:-1, 1:-1] / np.sqrt(grid.cell * np.sum(psi[1:-1, 1:-1] ** 2))
    b = phi[1:-1, 1:-1] / np.sqrt(grid.cell * np.sum(phi[1:-1, 1:-1] ** 2))
    true_angle = math.asin(min(1.0, float(np.sqrt(grid.cell * np.sum((a - b) ** 2)))))
    assert bound >= true_angle - 1e-12
    assert bound < math.pi / 2
    with pytest.raises(ValueError):
        transported_angle_bound_2d(psi, grid, result.psi, -0.1)
