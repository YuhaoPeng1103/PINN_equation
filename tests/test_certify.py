"""后验证书的单元测试：解析本征对上证书必须成立且足够紧。

**离散 vs 连续**：证书是对二阶差分 Hamiltonian H_h 的严格结论，而解析
谐振子能量 0.5、1.5 属于连续算子。N=2001 时两者相差 ~1e-6，因此：

* 与连续值的比较用 1e-5 量级的容差；
* 包含性测试必须对照 H_h 的本征值，而不是连续值（Temple 区间宽度
  可以小于 O(h²) 的离散偏差，拿连续值当目标会误判）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nnschrodinger.physics.spectrum import harmonic_oscillator, infinite_square_well
from nnschrodinger.solver.certify import (
    certify_eigenpair,
    continuous_rayleigh_quotient,
    count_eigenvalues_below,
    discrete_eigenvalue,
    discrete_rayleigh_quotient,
    eigenfunction_angle_bound,
    heuristic_interval,
    isolating_interval,
    normalize,
    residual_norm,
    temple_bounds,
    weinstein_interval,
)

X = np.linspace(-6.0, 6.0, 2001)
V = 0.5 * X**2

# 粗网格：只用于求 H_h 的低阶本征值做“包含性”参照（稠密对角化 O(N³)）。
X_D = np.linspace(-6.0, 6.0, 401)
V_D = 0.5 * X_D**2


def _psi(n: int, x: np.ndarray = X) -> np.ndarray:
    return harmonic_oscillator(n).psi(x)


def _discrete_spectrum(x: np.ndarray, potential: np.ndarray, k: int = 3) -> np.ndarray:
    """Dirichlet 边界下 H_h 的前 k 个本征值（与内部差分格式一致）。"""
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    h_mat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(h_mat)[:k]


def _state_interval(ev: np.ndarray, state: int) -> tuple[float, float, float]:
    """取只包含 ev[state] 的区间，并给出谱隙下界。"""
    if state == 0:
        lower = ev[0] - 0.5 * (ev[1] - ev[0])
    else:
        lower = 0.5 * (ev[state - 1] + ev[state])
    if state + 1 < ev.size:
        upper = 0.5 * (ev[state] + ev[state + 1])
        gap = 0.5 * min(ev[state] - ev[state - 1] if state > 0 else ev[1] - ev[0],
                        ev[state + 1] - ev[state])
    else:
        upper = ev[state] + 0.5 * (ev[state] - ev[state - 1])
        gap = 0.5 * (ev[state] - ev[state - 1])
    return lower, upper, gap


def test_rayleigh_and_residual_on_exact_ground_state() -> None:
    psi = _psi(0)
    eta_cont = continuous_rayleigh_quotient(psi, X, V)
    eta_disc = discrete_rayleigh_quotient(psi, X, V)
    r = residual_norm(psi, X, V, eta_disc)
    # 连续 Rayleigh 商与 H_h 的 Rayleigh 商都应在 O(h²) 内接近 0.5
    assert eta_cont == pytest.approx(0.5, abs=1e-5)
    assert eta_disc == pytest.approx(0.5, abs=1e-5)
    # 二阶中心差分的截断误差是 O(h²)，h≈0.006，残差应在 1e-4 以下
    assert r < 1e-4
    w_lo, w_hi = weinstein_interval(eta_disc, r)
    assert w_lo <= 0.5 <= w_hi


def test_temple_certificate_contains_discrete_ground_energy_and_is_tight() -> None:
    ev = _discrete_spectrum(X_D, V_D)
    lower, upper, gap = _state_interval(ev, 0)
    cert = certify_eigenpair(_psi(0, X_D), X_D, V_D, lower=lower, upper=upper, gap=gap)
    assert cert.valid
    assert cert.temple_lower is not None and cert.temple_upper is not None
    assert cert.temple_lower <= ev[0] <= cert.temple_upper
    assert cert.width is not None and cert.width < 1e-6
    assert cert.eigenfunction_angle_bound is not None
    assert cert.eigenfunction_angle_bound < 1e-3
    # 离散本征值本身也应落在连续值 0.5 的 O(h²) 邻域内
    assert ev[0] == pytest.approx(0.5, abs=1e-4)


def test_temple_certificate_for_first_excited_state() -> None:
    ev = _discrete_spectrum(X_D, V_D)
    lower, upper, gap = _state_interval(ev, 1)
    cert = certify_eigenpair(_psi(1, X_D), X_D, V_D, lower=lower, upper=upper, gap=gap)
    assert cert.valid
    assert cert.temple_lower <= ev[1] <= cert.temple_upper
    assert cert.width is not None and cert.width < 1e-6
    assert ev[1] == pytest.approx(1.5, abs=1e-3)


def test_infinite_well_exact_discrete_state_gets_tight_certificate() -> None:
    """sin(nπx/L) 是 H_h 的**精确**离散本征向量；残差在舍入噪声量级。

    这种情形下区间宽度会小于 η 的 ulp，证书必须显式加浮点裕度，
    否则会得到零宽度区间。
    """
    x = np.linspace(0.0, 1.0, 401)
    potential = np.zeros_like(x)
    h = float(x[1] - x[0])
    psi = infinite_square_well(1).psi(x)
    lam = 2.0 * np.sin(np.pi * h / 2.0) ** 2 / h**2
    cert = certify_eigenpair(psi, x, potential, isolate=True)
    assert cert.valid and cert.state_index == 0
    assert cert.temple_lower <= lam <= cert.temple_upper
    assert cert.width is not None and 0.0 < cert.width < 1e-8


def test_sturm_count_matches_dense_diagonalization() -> None:
    ev = _discrete_spectrum(X_D, V_D, 5)
    for shift in (-0.5, 0.25, 0.5, 0.75, 1.5, 2.5, 3.5):
        assert count_eigenvalues_below(X_D, V_D, shift) == int(np.sum(ev < shift))


def test_discrete_eigenvalue_matches_dense_diagonalization() -> None:
    ev = _discrete_spectrum(X_D, V_D, 4)
    for j in range(4):
        assert discrete_eigenvalue(X_D, V_D, j) == pytest.approx(ev[j], abs=1e-10)


def test_isolating_interval_is_verified_and_automatic() -> None:
    ev = _discrete_spectrum(X_D, V_D)
    psi = _psi(0, X_D)
    eta = discrete_rayleigh_quotient(psi, X_D, V_D)
    iso = isolating_interval(X_D, V_D, eta)
    assert iso.state_index == 0
    assert iso.lower < eta < iso.upper
    n_lo = count_eigenvalues_below(X_D, V_D, iso.lower)
    n_hi = count_eigenvalues_below(X_D, V_D, iso.upper)
    assert n_hi - n_lo == 1
    assert iso.lower <= ev[0] <= iso.upper
    # 谱隙应接近相邻本征值间距，而不是“目标自己的残差尺度”
    assert iso.gap > 0.5
    cert = certify_eigenpair(psi, X_D, V_D, isolate=True)
    assert cert.valid and cert.state_index == 0
    assert cert.temple_lower <= ev[0] <= cert.temple_upper


def test_non_isolating_interval_is_rejected() -> None:
    cert = certify_eigenpair(_psi(0, X_D), X_D, V_D, lower=0.25, upper=1.75)
    assert not cert.valid
    assert "含 2 个本征值" in cert.reason


def test_temple_arithmetic_matches_formula() -> None:
    # Kato–Temple：下界除以上邻居距离 (upper−η)，上界除以下邻居距离 (η−lower)。
    lo, hi = temple_bounds(0.5, 0.1, 0.25, 1.25)
    assert lo == pytest.approx(0.5 - 0.01 / 0.75)
    assert hi == pytest.approx(0.5 + 0.01 / 0.25)


def test_temple_bounds_uses_correct_denominators() -> None:
    """回归测试：分母写反会给出高于真值的“下界”。

    两能级例子：H=diag(0,10)，ψ 取 50/50 混合时 η=5、r²=25。
    正确下界是 5−25/(10−5)=0（恰好等于真值 0）；把分母写成 η−lower
    会得到 5−25/6=0.833>0，直接把真值排除在区间外。
    """
    lo, hi = temple_bounds(5.0, 5.0, -1.0, 10.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(5.0 + 25.0 / 6.0)
    assert lo <= 0.0 <= hi


def test_temple_condition_is_enforced() -> None:
    """r² ≥ (upper−η)(η−lower) 时定理前提不成立，必须报错而不是硬算。"""
    with pytest.raises(ValueError):
        temple_bounds(0.5, 2.0, 0.25, 1.25)


def test_mixture_gets_a_wider_but_still_valid_certificate() -> None:
    """ψ = ψ₀ + 0.2 ψ₁ 的残差更大，Temple 区间应仍包含 λ₀ 且更宽。"""
    ev = _discrete_spectrum(X_D, V_D)
    lower, upper, _ = _state_interval(ev, 0)
    psi = _psi(0, X_D) + 0.2 * _psi(1, X_D)
    psi_unit, _ = normalize(psi, X_D)
    cert = certify_eigenpair(psi_unit, X_D, V_D, lower=lower, upper=upper)
    exact = certify_eigenpair(_psi(0, X_D), X_D, V_D, lower=lower, upper=upper)
    assert cert.valid
    assert cert.temple_lower <= ev[0] <= cert.temple_upper
    assert cert.width > exact.width
    # 区间已被 Sturm 计数验证为单态，因此可以自动给出夹角界
    assert cert.eigenfunction_angle_bound is not None
    assert exact.eigenfunction_angle_bound is not None
    assert cert.eigenfunction_angle_bound > exact.eigenfunction_angle_bound


def test_angle_bound_formula() -> None:
    assert eigenfunction_angle_bound(0.1, 1.0) == pytest.approx(math.asin(0.1))
    # 残差大于谱隙时截断到 π/2，而不是抛 math domain error
    assert eigenfunction_angle_bound(2.0, 1.0) == pytest.approx(math.pi / 2)


def test_invalid_interval_is_reported_not_hidden() -> None:
    cert = certify_eigenpair(_psi(0), X, V, lower=0.6, upper=1.2)
    assert not cert.valid
    assert cert.temple_lower is None and cert.temple_upper is None
    assert "不在区间" in cert.reason


def test_without_interval_only_weinstein() -> None:
    cert = certify_eigenpair(_psi(0), X, V)
    assert not cert.valid
    assert cert.temple_lower is None
    assert cert.weinstein_lower <= 0.5 <= cert.weinstein_upper


def test_heuristic_interval_leaves_room_for_neighbors() -> None:
    spectrum = np.array([0.5, 1.5, 2.5])
    assert heuristic_interval(spectrum, 0, margin=0.1) == (0.4, 0.6)
    assert heuristic_interval(spectrum, 1, margin=0.1) == (1.4, 1.6)


def test_zero_wavefunction_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize(np.zeros_like(X), X)


def test_nonuniform_grid_is_rejected() -> None:
    x_bad = np.linspace(-6.0, 6.0, 100)
    x_bad[50] += 0.01
    with pytest.raises(ValueError):
        discrete_rayleigh_quotient(np.ones_like(x_bad), x_bad, np.zeros_like(x_bad))


def test_sturm_count_survives_shift_exactly_on_diagonal() -> None:
    """回归：shift 恰好落在谱点上（自由粒子 1/h²）时不得抛错。

    旧实现用绝对扰动（1e-13·scale）重试，在 h⁻²=160000 时几次扰动都落在
    同一个浮点间隔里，最终抛错；离散基态参考值因此取不到。

    此时“小于”与“小于等于”在浮点下不可区分，计数本身不适定。约定是
    上移 shift 后重试（返回 ≤ shift 的计数），故结果只能与 dense 的
    ``<`` 或 ``<=`` 计数一致；把边界放进谱隙后必须与 dense 严格一致。
    """
    x = np.linspace(0.0, 1.0, 401)
    potential = np.zeros_like(x)
    h = float(x[1] - x[0])
    shift = 1.0 / h**2
    diag = 1.0 / h**2 + potential[1:-1]
    off = -0.5 / h**2
    m = diag.size
    dense = np.diag(diag) + np.diag(np.full(m - 1, off), 1) + np.diag(np.full(m - 1, off), -1)
    spectrum = np.linalg.eigvalsh(dense)
    below = int(np.sum(spectrum < shift))
    at_or_below = int(np.sum(spectrum <= shift))
    assert count_eigenvalues_below(x, potential, shift) in (below, at_or_below)
    # 谱隙内的 shift 必须与稠密对角化完全一致
    for safe in (shift * (1.0 - 1e-9), shift * (1.0 + 1e-9)):
        assert count_eigenvalues_below(x, potential, safe) == int(np.sum(spectrum < safe))
