"""证书引导谱滤波的测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nnschrodinger.solver.certify import certify_eigenpair
from nnschrodinger.solver.filtering import (
    default_cutoffs,
    dst1,
    filter_dirichlet,
    guided_filter,
)

X = np.linspace(0.0, 1.0, 401)
H = float(X[1] - X[0])
V = np.zeros_like(X)


def _exact(n: int) -> np.ndarray:
    psi = math.sqrt(2.0) * np.sin(n * math.pi * X)
    psi[0] = psi[-1] = 0.0
    return psi


def test_dst1_is_involution_and_matches_matrix() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=9)
    assert np.allclose(dst1(dst1(values)), values, atol=1e-14)
    m = values.size
    sine = math.sqrt(2.0 / (m + 1.0)) * np.sin(
        math.pi * np.outer(np.arange(1, m + 1), np.arange(1, m + 1)) / (m + 1.0)
    )
    assert np.allclose(dst1(values), sine @ values, atol=1e-13)


def test_filter_keeps_low_modes_and_zeroes_high_modes() -> None:
    psi = _exact(1) + 1e-2 * np.sin(120 * math.pi * X)
    psi[0] = psi[-1] = 0.0
    out = filter_dirichlet(psi, X, cutoff=2)
    assert np.allclose(out, _exact(1), atol=1e-12)
    assert out[0] == 0.0 and out[-1] == 0.0


def test_filter_rejects_bad_cutoff_and_shape() -> None:
    with pytest.raises(ValueError):
        filter_dirichlet(np.zeros(10), np.linspace(0, 1, 10), cutoff=0)
    with pytest.raises(ValueError):
        filter_dirichlet(np.zeros(10), np.linspace(0, 1, 11), cutoff=2)


def test_guided_filter_recovers_a_state_hidden_by_high_frequency_noise() -> None:
    """原始 ψ 的 H_h 残差被噪声放大到不可认证；滤波后应恢复认证与紧界。"""
    psi = _exact(1) + 1e-2 * np.sin(100 * math.pi * X)
    psi[0] = psi[-1] = 0.0
    raw = certify_eigenpair(psi, X, V, isolate=True)
    assert not raw.valid

    result = guided_filter(psi, X, V, cutoffs=[1, 2, 3, 5, 10, 20], max_cutoff=20)
    assert result.certificate.valid
    # 噪声是第 100 个正弦模式；任何 <100 的截断都把它去掉（等价的紧界，
    # 取最小宽度的第一个即可）。关键是不要选到 ≥100、把噪声留在解里。
    assert result.cutoff < 100
    assert np.sqrt(H * np.sum((result.psi - _exact(1)) ** 2)) < 1e-12
    # 证书的区间包含同一离散 H_h 的基态本征值
    from nnschrodinger.solver.certify import discrete_eigenvalue

    lam = discrete_eigenvalue(X, V, 0)
    assert result.certificate.temple_lower <= lam <= result.certificate.temple_upper


def test_guided_filter_scan_is_reported() -> None:
    result = guided_filter(_exact(1), X, V, cutoffs=[1, 2, 4])
    assert [entry["cutoff"] for entry in result.scan] == [1, 2, 4]
    assert result.cutoff in (1, 2, 4)
    assert 1 in default_cutoffs(100)


def test_transported_angle_bound_covers_raw_output() -> None:
    """滤波后证书的夹角界可以搬回原始 ψ：2‖u−u_K‖ + sin∠(u_K,φ) 仍是上界。"""
    from nnschrodinger.solver.filtering import transported_angle_bound

    phi = _exact(1)
    noisy = _exact(1) + 1e-2 * np.sin(100 * math.pi * X)
    noisy[0] = noisy[-1] = 0.0
    psi_k = filter_dirichlet(noisy, X, cutoff=1)
    psi_raw = psi_k + 5e-2 * np.sin(37 * math.pi * X)
    psi_raw[0] = psi_raw[-1] = 0.0

    def angle(a: np.ndarray, b: np.ndarray) -> float:
        na = math.sqrt(H * np.sum(a[1:-1] ** 2))
        nb = math.sqrt(H * np.sum(b[1:-1] ** 2))
        ua, ub = a[1:-1] / na, b[1:-1] / nb
        if float(np.dot(ua, ub)) < 0.0:
            ub = -ub
        return math.asin(min(1.0, math.sqrt(H * np.sum((ua - ub) ** 2))))

    true_k = angle(psi_k, phi)
    true_raw = angle(psi_raw, phi)
    bound = transported_angle_bound(psi_raw, X, psi_k, true_k)
    assert bound >= true_raw
    assert bound < math.pi / 2
    with pytest.raises(ValueError):
        transported_angle_bound(psi_raw, X, np.zeros_like(psi_k), 0.0)
    with pytest.raises(ValueError):
        transported_angle_bound(psi_raw, X, psi_k, -0.1)
