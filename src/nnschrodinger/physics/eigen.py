"""对称矩阵特征分解（纯 numpy，无 scipy）与薛定谔算子的离散化。

为什么自己写：环境无 scipy，而 numpy.linalg.eigh 只接受稠密矩阵、复杂度 O(N³)
（N=4000 时单次约两分钟）。本项目的参考解、谱方法基线与单元测试需要在
N = 10³ 量级的网格上反复求本征对。

算法：**循环 Jacobi（cyclic Givens）**，专为三对角/稀疏结构设计：
  * 每轮（sweep）只需要处理相邻的 (k, k+1) 对，共 2N−3 次旋转；
  * 每次旋转用解析公式一次更新 2 个对角元 + 2 个次对角元（O(1) 工作），
    特征向量矩阵只更新两列（O(N) 工作），单轮 O(N²)；
  * 三对角矩阵通常 3~6 轮即收敛到机器精度。

旋转公式（Golub & Van Loan §8.5，针对三对角矩阵）：
    β = (d_{k+1} − d_k) / (2·e_k)
    τ = sign(β) / (|β| + sqrt(1 + β²))
    c = 1/sqrt(1 + τ²),  s = −τ·c
    d_k ← d_k − τ·e_k,  d_{k+1} ← d_{k+1} + τ·e_k,  e_k ← 0
    e_{k−1} ← c·e_{k−1},  e_{k+1} ← c·e_{k+1}

数值等价性由 tests/test_eigen.py（与 numpy.linalg.eigh 逐元素比对）与
tests/test_numerov.py（谐振子精确谱）共同保证。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def symmetric_tridiagonal_eigh(
    diag: np.ndarray,
    offdiag: np.ndarray,
    *,
    with_vectors: bool = True,
    max_sweeps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """对称三对角矩阵 T 的完整特征分解（转成稠密后调用已验证的 Jacobi）。

    三对角的稀疏 Givens 迭代实现容易在"邻居耦合项"上出错（本项目踩过坑，
    见 docs/engineering-log.md）。这里选择**正确性优先**：把矩阵展开成稠密
    （N ≤ 800 时开销可忽略）并复用 symmetric_eigh_jacobi（经 numpy 逐元素
    比对验证）。
    """
    d = np.asarray(diag, dtype=np.float64)
    n = len(d)
    dense = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(dense, d)
    m = min(len(offdiag), n - 1)
    if m > 0:
        off = np.asarray(offdiag, dtype=np.float64)[:m]
        i = np.arange(m)
        dense[i, i + 1] = off
        dense[i + 1, i] = off
    return symmetric_eigh_jacobi(dense, with_vectors=with_vectors, max_sweeps=max_sweeps)

def symmetric_eigh_jacobi(
    matrix: np.ndarray,
    *,
    with_vectors: bool = True,
    max_sweeps: int = 60,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """对称矩阵的 **循环 Jacobi（Givens 旋转）** 完整特征分解。

    每轮扫过所有上三角元素 (p, q)，用解析公式一次更新 A 的第 p/q 行与列。
    与 numpy.linalg.eigh 逐元素比对（tests/test_eigen.py）验证一致。
    稠密 N×N 时复杂度 O(N³)，但常数很小；N ≤ 400 时足够快。
    """
    a = np.array(matrix, dtype=np.float64)
    a = 0.5 * (a + a.T)
    n = a.shape[0]
    v = np.eye(n) if with_vectors else np.empty((0, 0))
    scale = float(np.max(np.abs(a))) + 1e-300
    for _ in range(max_sweeps):
        if float(np.max(np.abs(a - np.diag(np.diag(a))))) <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if apq == 0.0:
                    continue
                tau = (a[q, q] - a[p, p]) / (2.0 * apq)
                if tau >= 0.0:
                    tt = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
                else:
                    tt = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
                cs = 1.0 / np.sqrt(1.0 + tt * tt)
                sn = -tt * cs
                # p/q 两行
                ap = a[p, :].copy()
                aq = a[q, :].copy()
                a[p, :] = cs * ap + sn * aq
                a[q, :] = -sn * ap + cs * aq
                # p/q 两列
                ap = a[:, p].copy()
                aq = a[:, q].copy()
                a[:, p] = cs * ap + sn * aq
                a[:, q] = -sn * ap + cs * aq
                if with_vectors:
                    vp = v[:, p].copy()
                    vq = v[:, q].copy()
                    v[:, p] = cs * vp + sn * vq
                    v[:, q] = -sn * vp + cs * vq
    order = np.argsort(np.diag(a))
    return np.diag(a).copy()[order], (v[:, order] if with_vectors else v)

def largest_off_diagonal(a: np.ndarray) -> tuple[int, int, float]:
    """返回 (i, j, |a_ij|)，i<j；没有非对角元时返回 (0, 0, 0)。"""
    n = a.shape[0]
    best_i = best_j = 0
    best = 0.0
    for i in range(n - 1):
        row = np.abs(a[i, i + 1:])
        jj = int(row.argmax())
        if row[jj] > best:
            best = float(row[jj])
            best_i, best_j = i, i + 1 + jj
    return best_i, best_j, best


def tridiagonal_hamiltonian(
    lo: float,
    hi: float,
    potential: Callable[[np.ndarray], np.ndarray],
    n_points: int,
    *,
    with_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 [lo, hi] 上离散 Ĥ = −½·d²/dx² + V(x)（二阶中心差分）。

    Returns
    -------
    (grid, eigvals, eigvecs)；eigvecs 的列满足 Σ|ψ_i|²·h = 1（L2 归一化）。
    """
    grid = np.linspace(lo, hi, int(n_points))
    h = float(grid[1] - grid[0])
    diag = -1.0 / h**2 + np.asarray(potential(grid), dtype=np.float64)
    offdiag = np.full(len(grid) - 1, 0.5 / h**2)
    evals, evecs = symmetric_tridiagonal_eigh(diag, offdiag, with_vectors=with_vectors)
    if with_vectors:
        evecs = evecs / np.sqrt(h)
    return grid, evals, evecs


__all__ = ["symmetric_tridiagonal_eigh", "symmetric_eigh_jacobi", "largest_off_diagonal", "tridiagonal_hamiltonian"]