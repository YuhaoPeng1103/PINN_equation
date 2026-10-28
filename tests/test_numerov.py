"""Numerov + 打靶求解器的单元测试（与解析谱逐项比对）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nnschrodinger.physics.numerov import numerov_eigenvalues, numerov_spectrum


@pytest.fixture(scope="module")
def harmonic_spectrum():
    return numerov_eigenvalues(lambda x: 0.5 * x**2, -8.0, 8.0, (-0.5, 20.0), n_points=8001)


def test_harmonic_exact_odd_states(harmonic_spectrum):
    # 网格对称时，奇宇称态在离散格式下是**精确**的
    for n in (1, 3, 5):
        assert abs(harmonic_spectrum[n] - (n + 0.5)) < 1e-8


def test_harmonic_all_states_close(harmonic_spectrum):
    reference = np.arange(len(harmonic_spectrum)) + 0.5
    assert np.max(np.abs(harmonic_spectrum - reference)) < 5e-3


def test_even_states_are_second_order_accurate():
    """中点匹配必须用中心差分。

    旧实现用单侧差分估 ψ′/ψ，偶宇称态只有 O(h) 精度：N=2001 时 E₀ 偏 1.7e-3、
    E₂ 偏 4.2e-3；奇宇称态因中点 ψ≈0 恰好避开该误差，看起来“只有奇态是准的”。
    改成中心差分后，同一网格上偶态误差应到 1e-8 以内。
    """
    energies = numerov_eigenvalues(lambda x: 0.5 * x**2, -6.0, 6.0, (-0.5, 10.0), n_points=2001)
    for n in (0, 1, 2, 3):
        assert abs(float(energies[n]) - (n + 0.5)) < 1e-8


def test_infinite_well_against_analytic():
    # 用 V₀ = 2000 的深阱近似无限深阱：误差 ~ E/V₀
    energies = numerov_eigenvalues(lambda x: np.where(np.abs(x - 0.5) < 0.5, 0.0, 2000.0), 0.0, 1.0, (0.0, 2000.0), n_points=8001)
    exact = np.array([n * n * math.pi**2 / 2 for n in range(1, 4)])
    assert np.all(np.abs(energies[:3] - exact) < 1e-1)


def test_finite_well_monotone():
    potential = lambda x: np.where(np.abs(x) < 1.0, -50.0, 0.0)  # noqa: E731
    energies = numerov_eigenvalues(potential, -8.0, 8.0, (-50.0, 0.0), n_points=8001)
    assert np.all(np.diff(energies) > 0)
    assert energies[0] > -50.0
    assert energies[0] < 0.0


def test_wavefunctions_orthonormal_and_shape():
    result = numerov_spectrum(lambda x: 0.5 * x**2, -8.0, 8.0, (-0.5, 20.0), 4, n_points=4001)
    h = float(result.grid[1] - result.grid[0])
    for k in range(4):
        psi = result.vectors[:, k]
        assert abs(float(np.sum(psi**2) * h) - 1.0) < 1e-8
    gram = result.vectors.T @ result.vectors * h
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) < 5e-2


def test_energy_window_too_small():
    with pytest.raises(ValueError):
        numerov_spectrum(lambda x: 0.5 * x**2, -8.0, 8.0, (-0.5, 0.2), 2, n_points=2001)
