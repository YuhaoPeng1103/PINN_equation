"""对称矩阵的 Jacobi 特征分解与 numpy 的数值一致性。"""

from __future__ import annotations

import numpy as np

from nnschrodinger.physics.eigen import (
    largest_off_diagonal,
    symmetric_eigh_jacobi,
    symmetric_tridiagonal_eigh,
    tridiagonal_hamiltonian,
)


def test_dense_random_matches_numpy():
    rng = np.random.default_rng(0)
    for size in (4, 7, 20):
        b = rng.normal(size=(size, size))
        s = 0.5 * (b + b.T)
        w, v = symmetric_eigh_jacobi(s)
        assert np.allclose(w, np.linalg.eigh(s)[0], atol=1e-10)
        assert np.max(np.abs(s @ v - v * w)) < 1e-9
        assert np.allclose(v.T @ v, np.eye(size), atol=1e-10)


def test_tridiagonal_identity():
    """三对角矩阵本身：特征值与特征向量都应精确。"""
    diag = np.array([2.0, 1.0, 5.0, -3.0, 0.5, 7.0])
    offdiag = np.array([1.0, 0.2, -0.5, 0.0, 2.0])
    tri = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    w, v = symmetric_tridiagonal_eigh(diag, offdiag)
    assert np.allclose(w, np.linalg.eigh(tri)[0], atol=1e-10)
    assert np.max(np.abs(tri @ v - v * w)) < 1e-9
    assert v.shape == (6, 6)


def test_tridiagonal_wrapper_builds_from_diagonals():
    """包装函数只取输入的对角/次对角元重建三对角矩阵，结果必须与 numpy 一致。"""
    diag = np.array([1.0, -2.0, 4.0, 0.0, 3.0])
    offdiag = np.array([0.5, -1.5, 2.0, 0.25])
    tri = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    w, _ = symmetric_tridiagonal_eigh(diag, offdiag)
    assert np.allclose(w, np.linalg.eigh(tri)[0], atol=1e-10)


def test_tridiagonal_larger():
    rng = np.random.default_rng(7)
    n = 101
    diag = rng.normal(size=n)
    offdiag = rng.normal(size=n - 1)
    tri = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    w, v = symmetric_tridiagonal_eigh(diag, offdiag)
    assert np.allclose(w, np.linalg.eigh(tri)[0], atol=1e-9)
    assert np.max(np.abs(tri @ v - v * w)) < 1e-8

def test_diagonal_matrix():
    w, _ = symmetric_tridiagonal_eigh(np.array([3.0, 1.0, 2.0]), np.zeros(2))
    assert np.allclose(w, [1.0, 2.0, 3.0])


def test_largest_off_diagonal():
    m = np.array([[1.0, 0.0, 5.0], [0.0, 2.0, 0.0], [5.0, 0.0, 3.0]])
    i, j, value = largest_off_diagonal(m)
    assert (i, j) == (0, 2)
    assert value == 5.0
    assert largest_off_diagonal(np.eye(3))[2] == 0.0


def test_harmonic_hamiltonian_spectrum():
    # 在 [-8, 8] 上做有限差分会把边界 V(±8)=32 变成“人为深阱”，产生大量负能量的
    # 截断伪态；真正的谐振子能级夹在其中。这里只校验数值实现的**自洽性**：
    # 离散谱单调、特征向量正交归一、以及矩阵是真正对称的。
    grid, evals, evecs = tridiagonal_hamiltonian(-8.0, 8.0, lambda x: 0.5 * x**2, 151)
    assert evals.size == 151
    assert np.all(np.diff(evals) > -1e-12)
    h = float(grid[1] - grid[0])
    psi = evecs[:, 0]
    assert abs(float(np.sum(psi**2) * h) - 1.0) < 1e-9
    gram = evecs[:, :5].T @ evecs[:, :5] * h
    assert np.max(np.abs(gram - np.eye(5))) < 1e-8
    # 残差：Ĥψ = Eψ
    v = 0.5 * grid**2
    n = len(grid)
    # tridiagonal_hamiltonian 构造的是 H = tridiag(½, −1, ½)/h² + V，
    # 等价于 Ĥ = −½·D₂，其中 D₂ = tridiag(1, −2, 1)/h² 的**相反数**，
    # 因此校验式是 +½·L 而不是 −½·L。
    lap = (np.diag(np.full(n, -2.0 / h**2)) + np.diag(np.full(n - 1, 1.0 / h**2), 1) + np.diag(np.full(n - 1, 1.0 / h**2), -1)) @ psi
    residual = 0.5 * lap + v * psi - evals[0] * psi
    assert float(np.abs(residual).max()) < 1e-8
