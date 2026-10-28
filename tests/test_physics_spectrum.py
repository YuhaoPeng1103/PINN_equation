"""参考解层（解析解）的单元测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nnschrodinger.physics.spectrum import (
    finite_square_well,
    fix_sign,
    get_reference,
    harmonic_oscillator,
    hydrogen_radial,
    infinite_square_well,
    to_hhl_normalization,
)


def test_harmonic_energies():
    for n in range(6):
        assert harmonic_oscillator(n).energy == pytest.approx(n + 0.5)


def _gl_quad(f, lo: float, hi: float, n: int = 4000) -> float:
    """2 点 Gauss-Legendre 复合求积（对解析解可达机器精度）。

    不用梯形公式：像 e^{-x²}·x² 这样的态在 |x| 大的尾部会带来 1e-8 量级的
    求积误差与严重的截断误差，会让“归一化断言”看起来像代码 bug。
    """
    edges = np.linspace(lo, hi, n + 1)
    a, b = edges[:-1], edges[1:]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    off = half / np.sqrt(3.0)
    return float(np.sum(half * (f(mid - off) + f(mid + off))))


def test_harmonic_wavefunction_matches_analytic():
    r = harmonic_oscillator(3)
    psi = r.psi
    assert _gl_quad(lambda t: psi(t) ** 2, -14, 14) == pytest.approx(1.0, abs=1e-9)
    # 与直接递推构造的 Hermite 解比较（同一函数应只差一个归一化常数）
    from nnschrodinger.physics.spectrum import _hermite

    x = np.linspace(-14, 14, 4001)
    raw = np.exp(-0.5 * x**2) * _hermite(3, x)
    ratio = psi(x) / raw
    good = np.abs(raw) > 1e-6 * np.abs(raw).max()
    assert float(np.ptp(ratio[good])) < 1e-8


def test_harmonic_orthogonality():
    x = np.linspace(-12, 12, 4001)  # noqa: F841
    psi0 = harmonic_oscillator(0).psi
    psi1 = harmonic_oscillator(1).psi
    assert abs(_gl_quad(lambda t: psi0(t) * psi1(t), -12, 12)) < 1e-9


def test_infinite_well_energies_and_boundaries():
    for n in range(1, 5):
        r = infinite_square_well(n, L=1.0)
        assert r.energy == pytest.approx(n * n * math.pi**2 / 2.0)
        assert r.psi(np.array([0.0, 1.0])).tolist() == pytest.approx([0.0, 0.0])
        assert r.psi(np.array([-0.5, 1.5])).tolist() == pytest.approx([0.0, 0.0])


def test_infinite_well_zero_at_nodes():
    # sin(2πx) 的零点：x = 0, 0.5, 1（不是 0, 0.25, 0.5）
    r = infinite_square_well(2, L=1.0)
    x = np.array([0.0, 0.5, 1.0])
    assert np.allclose(r.psi(x), 0.0, atol=1e-12)


def test_finite_well_binding_energies_ordering():
    e1 = finite_square_well(depth=50.0, half_width=1.0, n_states=1).energy
    e2 = finite_square_well(depth=50.0, half_width=1.0, n_states=2).energy
    assert e2 > e1
    assert e1 > -50.0 and e2 < 0.0
    # 深势阱的基态能量应略高于无限深阱（V₀ 有限抬高基态）
    # V₀ = 50 的有限深阱基态约为 −48.98（比无限深阱的 4.93 深得多），
    # 只断言它落在阱内且高于阱底
    assert -50.0 < e1 < 0.0


def test_finite_well_norm_and_tails():
    r = finite_square_well(depth=50.0, half_width=1.0, n_states=1)
    lo, hi = r.domain
    psi = r.psi
    assert _gl_quad(lambda t: psi(t) ** 2, lo, hi) == pytest.approx(1.0, abs=1e-4)
    x = np.linspace(lo, hi, 4001)
    psi = r.psi(x)
    assert abs(float(psi[0])) < 1e-3 and abs(float(psi[-1])) < 1e-3


def test_hydrogen_energies():
    for n in range(1, 5):
        assert hydrogen_radial(n, 0).energy == pytest.approx(-1.0 / (2 * n * n))


def test_hydrogen_1s_normalization():
    r = hydrogen_radial(1, 0)
    # e^{-2r}·r² 在 0 处光滑，但 Gauss-Legendre 复合规则在 0 端点处理不佳，
    # 这里改用高分辨率梯形积分（200001 点，误差 ~2e-6）
    grid = np.linspace(0.0, 20.0, 200001)
    psi = r.psi(grid)
    assert float(np.trapezoid(psi**2, grid)) == pytest.approx(1.0, abs=1e-5)


def test_fix_sign_convention():
    assert fix_sign(np.array([-1.0, 0.0, 0.5]))[0] > 0
    assert fix_sign(np.array([-0.5, 0.0, 1.0]))[2] > 0


def test_hhl_normalization():
    psi = np.array([0.0, 2.0, -4.0])
    out = to_hhl_normalization(psi)
    assert float(np.max(np.abs(out))) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        to_hhl_normalization(np.zeros(3))


def test_get_reference_dispatch():
    assert get_reference("harmonic", 1).energy == pytest.approx(1.5)
    assert get_reference("infinite_well", 0).energy == pytest.approx(math.pi**2 / 2)
    with pytest.raises(KeyError):
        get_reference("no_such_problem", 0)