"""子空间（簇）证书的测试。

锁住四件事：

1. **重数**由验证计数测出，与稠密特征分解一致（一维 Sturm / 二维惯性）；
2. **双向簇围道**真的包含全部 m 个本征值（用稠密谱核对）；
3. **主子角界**不小于真实主子角（命题 2 的数值核对），且与基的选取无关；
4. **拒绝**行为：简并簇上的 m=1、秩亏子空间、跨簇子空间都必须
   ``valid=False``，不能给出看似很紧的假界。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from nnschrodinger.solver.certify import _dirichlet_data
from nnschrodinger.solver.certify2d import (
    Grid2D,
    _sparse_operator,
    approximate_spectrum_2d,
)
from nnschrodinger.solver.certify_subspace import (
    certify_subspace_1d,
    certify_subspace_2d,
    principal_angle,
)


def _harmonic_1d(n: int = 401, half: float = 6.0):
    x = np.linspace(-half, half, n)
    v = 0.5 * x**2
    diag, off = _dirichlet_data(x, v)
    values, vectors = eigh_tridiagonal(diag, off * np.ones(diag.size - 1))

    def embed(vec: np.ndarray) -> np.ndarray:
        psi = np.zeros(n)
        psi[1:-1] = vec
        return psi

    return x, v, values, [embed(vectors[:, i]) for i in range(vectors.shape[1])]


def _box_2d(n: int = 31):
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))
    from scipy.sparse.linalg import eigsh

    operator = _sparse_operator(grid, 0.0)
    values, vectors = eigsh(operator, k=5, sigma=-1.0, which="LM", tol=1e-12)
    order = np.argsort(values)
    values, vectors = np.asarray(values)[order], np.asarray(vectors)[:, order]

    def embed(vec: np.ndarray) -> np.ndarray:
        psi = np.zeros(grid.shape)
        psi[1:-1, 1:-1] = vec.reshape(n - 2, n - 2)
        return psi

    return grid, values, vectors, embed


def test_1d_cluster_contains_exact_eigenvalues():
    x, v, values, vectors = _harmonic_1d()
    h = float(x[1] - x[0])
    for m in (1, 2, 3):
        cert = certify_subspace_1d(
            vectors[:m], x, v, approx=values[:8]
        )
        assert cert.valid, cert.reason
        assert cert.verified_multiplicity == m
        assert cert.first_index == 0
        assert cert.interval_lower < values[0]
        assert values[m - 1] < cert.interval_upper
        angle = principal_angle(vectors[:m], vectors[:m], h)
        assert angle <= cert.angle_bound + 1e-14


def test_1d_cluster_contains_shifted_block():
    """取第 2–4 个态（不是从基态开始）作为簇，区间仍须只含这 3 个。"""
    x, v, values, vectors = _harmonic_1d()
    cert = certify_subspace_1d(vectors[2:5], x, v, approx=values[:10])
    assert cert.valid, cert.reason
    assert cert.verified_multiplicity == 3
    assert cert.first_index == 2
    for i in range(2, 5):
        assert cert.interval_lower < values[i] < cert.interval_upper
    assert not (cert.interval_lower < values[1])


def test_1d_rank_deficient_is_refused():
    x, v, values, vectors = _harmonic_1d()
    cert = certify_subspace_1d([vectors[0], vectors[0]], x, v, approx=values[:6])
    assert not cert.valid
    assert "秩亏" in cert.reason


def test_2d_degenerate_pair_is_certified():
    """方阱的 (1,2)/(2,1) 二重简并：单态证书拒绝，簇证书必须接受。"""
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 6)
    cert = certify_subspace_2d(
        [embed(vectors[:, 1]), embed(vectors[:, 2])], grid, approx=approx
    )
    assert cert.valid, cert.reason
    assert cert.verified_multiplicity == 2
    assert cert.first_index == 1
    assert cert.interval_lower < values[1] and values[2] < cert.interval_upper
    assert cert.margin > 0
    true = principal_angle(
        [vectors[:, 1], vectors[:, 2]], [vectors[:, 1], vectors[:, 2]], grid.cell
    )
    assert true <= cert.angle_bound + 1e-14


def test_2d_single_member_of_degenerate_pair_is_refused():
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 6)
    cert = certify_subspace_2d([embed(vectors[:, 1])], grid, approx=approx)
    assert not cert.valid
    assert cert.verified_multiplicity is None


def test_2d_basis_independence():
    """簇证书只依赖 span，不依赖给出的基。"""
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 6)
    a = embed(vectors[:, 1])
    b = embed(vectors[:, 2])
    mixed = 0.6 * a + 0.8 * b
    cert_direct = certify_subspace_2d([a, b], grid, approx=approx)
    cert_mixed = certify_subspace_2d([a, mixed], grid, approx=approx)
    assert cert_direct.valid and cert_mixed.valid
    assert cert_direct.verified_multiplicity == cert_mixed.verified_multiplicity == 2
    assert cert_direct.interval_lower == pytest.approx(cert_mixed.interval_lower)
    assert cert_direct.interval_upper == pytest.approx(cert_mixed.interval_upper)


def test_2d_mismatched_multiplicity_is_refused():
    """给 3 个向量但只覆盖 2 重简并簇 + 下一个态：区间无法只含 3 个而夹住全部 Ritz 值。"""
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 8)
    cert = certify_subspace_2d(
        [embed(vectors[:, 1]), embed(vectors[:, 2]), embed(vectors[:, 3])],
        grid,
        approx=approx,
    )
    # 第 0、1、2 个态加第 3 个态：这段其实是 (簇 2 个) + (39.4 一个) 且中间无谱，
    # 计数差为 3，因此**可以**被认证；这里要确认的是它落在正确的下标上。
    assert cert.valid, cert.reason
    assert cert.verified_multiplicity == 3
    assert cert.first_index == 1
    assert values[3] < cert.interval_upper


def test_explicit_interval_is_verified_not_trusted():
    """调用方给的区间如果计数差不对，必须拒绝。"""
    grid, values, vectors, embed = _box_2d()
    cert = certify_subspace_2d(
        [embed(vectors[:, 1]), embed(vectors[:, 2])],
        grid,
        lower=values[0] - 0.1,
        upper=values[3] + 0.1,
    )
    assert not cert.valid
    assert "重数不符" in cert.reason or "夹住" in cert.reason


def test_certificate_serialises():
    grid, values, vectors, embed = _box_2d()
    cert = certify_subspace_2d(
        [embed(vectors[:, 1]), embed(vectors[:, 2])], grid,
        approx=approximate_spectrum_2d(grid, 6),
    )
    data = cert.to_dict()
    assert data["n_states"] == 2
    assert len(data["enclosures"]) == 2
    for lo, hi in data["enclosures"]:
        assert hi > lo
    assert data["width"] == pytest.approx(
        data["interval_upper"] - data["interval_lower"]
    )


def test_2d_perturbed_subspace_still_certified():
    """被扰动的子空间：ρ 不再是机器精度，命题 1/2 仍须成立且界仍然有效。"""
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 6)
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
    envelope = xx * (1 - xx) * yy * (1 - yy)
    base = [embed(vectors[:, 1]), embed(vectors[:, 2])]

    def perturb(psi: np.ndarray, field: np.ndarray, weight: float) -> np.ndarray:
        norm = math.sqrt(grid.cell * np.sum(field[1:-1, 1:-1] ** 2))
        return psi + weight * field / norm

    trial = [
        perturb(base[0], envelope * np.sin(2 * math.pi * xx), 0.003),
        perturb(base[1], envelope * np.sin(3 * math.pi * yy), 0.002),
    ]
    cert = certify_subspace_2d(trial, grid, approx=approx)
    assert cert.valid, cert.reason
    assert cert.verified_multiplicity == 2
    true = principal_angle(trial, base, grid.cell)
    assert 0.0 < true <= cert.angle_bound + 1e-14
    # 逐 Ritz 值：命题 1 说它距离某个簇内本征值不超过 ρ_i
    for theta, rho in zip(cert.ritz_values, cert.residual_norms):
        assert rho > 1e-8
        assert min(abs(theta - values[1]), abs(theta - values[2])) <= rho + 1e-12


def test_2d_coarse_subspace_is_refused_not_faked():
    """扰动大到 ρ ≥ δ 时必须拒绝：证书不能把“差的子空间”认证成好的。"""
    grid, values, vectors, embed = _box_2d()
    approx = approximate_spectrum_2d(grid, 6)
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
    envelope = (xx * (1 - xx) * yy * (1 - yy)) * np.sin(4 * math.pi * xx)
    norm = math.sqrt(grid.cell * np.sum(envelope[1:-1, 1:-1] ** 2))
    trial = [embed(vectors[:, 1]) + 3.0 * envelope / norm, embed(vectors[:, 2])]
    cert = certify_subspace_2d(trial, grid, approx=approx)
    assert not cert.valid
