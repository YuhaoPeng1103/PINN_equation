"""连续算子的变分上界（定向舍入）。

问题
----
``certify`` 给出的是**离散算子** ``H_h`` 上的一条定理：残差界
``|λ_h − η| ≤ ‖H_hψ − ηψ‖``。它看不见 ``H_h`` 与连续算子 ``H`` 的距离，
所以对 ``λ_cont`` 一个字也说不出（论文 §6.8 就是这么写的）。

上界的来源
----------
不需要任何有限元后验估计或插值常数，只需要极小极大原理：
对**任意** ``u ∈ H¹_0``，

    λ_1(H) = min_{v∈H¹_0} R(v) ≤ R(u),    R(v) = (½∫v'² + ∫Vv²) / ∫v².

于是把网络的本征函数取成 P1 分片线性插值 ``u_h``——它连续、在端点为零，
因此属于 ``H¹_0``——``R(u_h)`` 就是 ``λ_cont`` 的一条可计算上界。这条界
只依赖 ``u`` 与算子的定义，不依赖任何参考解，也不依赖网格算子本身。

它与 ``certify`` 的地位不同：``certify`` 的数是对 ``H_h`` 的定理，这条
数是对 ``H`` 的定理。两者合起来就是连续本征值的双边夹逼
（下方见 :func:`box_lower_bound_1d`，只对方阱成立）。

严格性
------
三项积分的**累加**用区间算术（``certified_count.Interval``），端点向外
舍入，padding 随项数增长。单元积分本身（分段多项式的质量与刚度、势能的
Gauss 求和、``V`` 在节点的取值）用双精度计算，其舍入**没有被包围**：
在本项目的网格上它比累加的 padding 小几个数量级，也比待比较的 bracket
（~1e-4）小十一个数量级，但严格地说这是一个缺口，论文 §6.4 明说了。
要补上它，必须让 ``V`` 以区间算术求值。
势能项用 Gauss 求积：只要 ``V`` 在每个子区间上是次数不超过
``2q−2`` 的多项式，``q`` 点 Gauss 对被积函数 ``V·u²``（``u`` 线性，
故 ``u²`` 次数 2）是**精确**的。方阱（``V=0``）、谐振子（``V=½x²``，
次数 2）、有限深阱（分段常数）都满足；求积的精确性在运行时用
3 点与 5 点两套规则的差来核查，超出容差就拒绝。

口径
----
被界住的是与 ``V`` **解析给定** 的连续算子 ``−½d²/dx² + V``，Dirichlet
边条件；而不是以节点值为元素的离散算子。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .certified_count import Interval, _down, _up

# Gauss–Legendre 节点与权，定义在 [−1, 1]
_GAUSS = {
    2: (
        (-0.5773502691896257, 0.5773502691896257),
        (1.0, 1.0),
    ),
    3: (
        (-0.7745966692414834, 0.0, 0.7745966692414834),
        (0.5555555555555556, 0.8888888888888888, 0.5555555555555556),
    ),
    5: (
        (
            -0.9061798459386640,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.9061798459386640,
        ),
        (
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891,
        ),
    ),
}


@dataclass(frozen=True)
class RayleighBound:
    """一次变分上界及其可审计的字段。"""

    upper: float
    lower: float
    numerator_lo: float
    numerator_hi: float
    denominator_lo: float
    denominator_hi: float
    dimension: int
    exact_quadrature: bool
    reason: str = ""

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def certified(self) -> bool:
        return self.exact_quadrature and math.isfinite(self.upper)

    def to_dict(self) -> dict:
        return {
            "upper": self.upper,
            "lower": self.lower,
            "numerator_lo": self.numerator_lo,
            "numerator_hi": self.numerator_hi,
            "denominator_lo": self.denominator_lo,
            "denominator_hi": self.denominator_hi,
            "dimension": self.dimension,
            "exact_quadrature": self.exact_quadrature,
            "reason": self.reason,
        }


def _subintervals(start: float, stop: float, breakpoints) -> list[tuple[float, float]]:
    """把 [start, stop] 按落在其中的断点切开。"""
    cuts = sorted({start, stop} | {b for b in breakpoints if start < b < stop})
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _gauss_on(interval: tuple[float, float], order: int):
    a, b = interval
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    nodes, weights = _GAUSS[order]
    return [(mid + half * t, half * w) for t, w in zip(nodes, weights)]


def _element_integrals(
    left: float,
    right: float,
    u_left: float,
    u_right: float,
    potential,
    breakpoints,
    order: int,
) -> tuple[float, float, float]:
    """一个网格单元上的 (∫u'², ∫V u², ∫u²)。"""
    h = right - left
    slope = (u_right - u_left) / h
    stiffness = slope * slope * h
    mass = 0.0
    potential_energy = 0.0
    for lo, hi in _subintervals(left, right, breakpoints):
        length = hi - lo
        # u(t) = u_left + slope·(t − left)；用局部坐标做精确的多项式积分，
        # 避免把 ∫u² 也交给求积规则。
        alpha = lo - left
        beta = hi - left
        mass += (
            u_left * u_left * length
            + u_left * slope * (beta**2 - alpha**2)
            + slope * slope * (beta**3 - alpha**3) / 3.0
        )
        for t, w in _gauss_on((lo, hi), order):
            value = u_left + slope * (t - left)
            potential_energy += w * float(potential(np.array([t]))[0]) * value * value
    return stiffness, potential_energy, mass


def p1_rayleigh_bound(
    x: np.ndarray,
    potential,
    psi: np.ndarray,
    breakpoints=None,
    order: int = 5,
    check_order: int = 3,
) -> RayleighBound:
    """把 ``psi`` 在网格上做 P1 插值，返回其 Rayleigh 商的严格区间。

    ``psi`` 是**内部节点**上的取值（长度 ``len(x)−2``），与 ``certify``
    的约定一致；两端取零，故插值函数属于 ``H¹_0``。返回的 ``upper``
    是 ``λ_1(H)`` 的一条上界。

    ``potential`` 必须是解析可调用的（不是节点数组）：上界是对连续算子
    说的，用节点值代替会悄悄换掉一个算子。``breakpoints`` 给出 ``V``
    的不连续点或分段多项式断点，使求积在每段内精确。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if psi.size != x.size - 2:
        raise ValueError(f"psi 应含 {x.size - 2} 个内部节点值，收到 {psi.size}")
    breakpoints = () if breakpoints is None else tuple(breakpoints)
    if not callable(potential):
        raise TypeError("potential 必须是可调用对象；连续上界不能用节点值代替")

    values = np.concatenate(([0.0], psi, [0.0]))
    stiffness = Interval.point(0.0)
    potential_term = Interval.point(0.0)
    mass = Interval.point(0.0)
    check_potential = Interval.point(0.0)
    for index in range(x.size - 1):
        s, p, m = _element_integrals(
            float(x[index]),
            float(x[index + 1]),
            float(values[index]),
            float(values[index + 1]),
            potential,
            breakpoints,
            order,
        )
        _, c, _ = _element_integrals(
            float(x[index]),
            float(x[index + 1]),
            float(values[index]),
            float(values[index + 1]),
            potential,
            breakpoints,
            check_order,
        )
        stiffness = stiffness + Interval.point(s)
        potential_term = potential_term + Interval.point(p)
        check_potential = check_potential + Interval.point(c)
        mass = mass + Interval.point(m)

    # 两套 Gauss 规则给出两个区间。**互不相交**才是“被积函数不是低次多项式”
    # 的严格证据：两个区间各自包含自己那份求积结果，交集为空就不可能同时
    # 精确。重叠但不相等的差被区间宽度吸收（那是 padding，不是求积误差），
    # 此时取较高的上界——若两条规则都精确，它仍然是 I 的上界。
    disjoint = potential_term.hi < check_potential.lo or check_potential.hi < potential_term.lo
    exact = not disjoint
    p_hi = max(potential_term.hi, check_potential.hi)
    discrepancy = abs(potential_term.hi - check_potential.hi)
    scale = max(1.0, abs(potential_term.hi), abs(mass.hi))
    reason = "" if exact else (
        f"势能项的 {order} 点与 {check_order} 点 Gauss 结果相差 "
        f"{discrepancy:.3e}（相对 {discrepancy / scale:.1e}），求积不精确"
    )

    numerator = stiffness * Interval.point(0.5)
    numerator = Interval(
        _down(numerator.lo + potential_term.lo), _up(numerator.hi + p_hi)
    )
    if not (mass.lo > 0.0):
        raise ValueError("插值函数的 L² 范数为零，Rayleigh 商无定义")
    quotient = numerator / mass
    return RayleighBound(
        upper=quotient.hi,
        lower=quotient.lo,
        numerator_lo=numerator.lo,
        numerator_hi=numerator.hi,
        denominator_lo=mass.lo,
        denominator_hi=mass.hi,
        dimension=int(psi.size),
        exact_quadrature=bool(exact),
        reason=reason,
    )


def _strict_sum(values: np.ndarray) -> tuple[float, float]:
    """向量化求和的严格区间：S ± (m+2)·u·Σ|项|。

    把 m 项相加时，每次加法最多引入半个 ulp；用 ``γ_m = m·u/(1−m·u)``
    形式的界过于悲观，这里取更粗但一定成立的一条：误差不超过
    ``(m+2)·u·A``，其中 ``A = Σ|项|``、``u = 2⁻⁵³``。再加一个 ulp 兜住
    最后一次相对误差。
    """
    values = np.asarray(values, dtype=float).ravel()
    total = float(np.sum(values))
    magnitude = float(np.sum(np.abs(values)))
    padding = (values.size + 2) * 2.0**-53 * magnitude
    return _down(total - padding), _up(total + padding)


def _q1_quadrature(x, y, values, potential, order):
    """Q1 插值在全部单元上的 (刚度, 势能, 质量) 三项的区间估计。

    返回 ``(s_lo, s_hi, p_lo, p_hi, m_lo, m_hi)``，各自是 ``_strict_sum``
    给出的区间端点。``values`` 含边界零（形状 ``(len(y), len(x))``）。
    """
    nodes, weights = _GAUSS[order]
    # 单元内的局部坐标（0..1）：x = x_i + h_x·s, y = y_j + h_y·t
    s = 0.5 * (np.asarray(nodes) + 1.0)
    w1 = 0.5 * np.asarray(weights)
    S, T = np.meshgrid(s, s, indexing="xy")
    W = np.outer(w1, w1)
    hx = np.diff(x)
    hy = np.diff(y)
    u00 = values[:-1, :-1]
    u10 = values[:-1, 1:]
    u01 = values[1:, :-1]
    u11 = values[1:, 1:]
    # 单元数组的轴序是 [y-单元, x-单元, gauss, gauss]（与 values 的行列一致）
    hx4 = hx[None, :, None, None]
    hy4 = hy[:, None, None, None]
    a = (1 - S)[None, None, :, :]
    b = S[None, None, :, :]
    c = (1 - T)[None, None, :, :]
    d = T[None, None, :, :]
    u = (
        a * c * u00[:, :, None, None]
        + b * c * u10[:, :, None, None]
        + a * d * u01[:, :, None, None]
        + b * d * u11[:, :, None, None]
    )
    ux = ((1 - T) * (u10 - u00)[:, :, None, None] + T * (u11 - u01)[:, :, None, None]) / hx4
    uy = ((1 - S) * (u01 - u00)[:, :, None, None] + S * (u11 - u10)[:, :, None, None]) / hy4

    # 单元 (j, i) 覆盖 [x_i, x_{i+1}] × [y_j, y_{j+1}]
    cell_x = x[:-1, None, None] + hx[:, None, None] * S[None, :, :]
    cell_y = y[:-1, None, None] + hy[:, None, None] * T[None, :, :]
    X = np.broadcast_to(cell_x[None, :, :, :], u.shape)
    Y = np.broadcast_to(cell_y[:, None, :, :], u.shape)
    V = np.asarray(potential(X, Y), dtype=float)
    if V.shape != u.shape:
        V = np.broadcast_to(V, u.shape)

    jacobian = (hy[:, None] * hx[None, :])[:, :, None, None] * W[None, None, :, :]
    s_lo, s_hi = _strict_sum(0.5 * jacobian * (ux**2 + uy**2))
    p_lo, p_hi = _strict_sum(jacobian * V * u**2)
    m_lo, m_hi = _strict_sum(jacobian * u**2)
    return s_lo, s_hi, p_lo, p_hi, m_lo, m_hi


def q1_rayleigh_bound_2d(
    x: np.ndarray,
    y: np.ndarray,
    potential,
    psi: np.ndarray,
    order: int = 5,
    check_order: int | None = 3,
) -> RayleighBound:
    """二维：把 ``psi`` 做成 Q1（双线性）插值，返回 Rayleigh 商的严格区间。

    Q1 是 ``H¹_0`` 的协调子空间，且包含一维 P1 的乘积，因此对可分离势
    它给出的上界与两条一维上界之和一致；对不可分离的势（如
    ``anharmonic2d``，``V = ½x²+½y²+½x²y²``）它是唯一可用的严格上界。

    ``psi`` 是内部 ``(len(y)−2, len(x)−2)`` 的子块（行的索引对应 ``y``），
    与 ``certify2d`` 的网格约定一致。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float)
    if psi.shape != (y.size - 2, x.size - 2):
        raise ValueError(
            f"psi 形状应为 {(y.size - 2, x.size - 2)}，收到 {psi.shape}"
        )
    if not callable(potential):
        raise TypeError("potential 必须是可调用对象")

    values = np.zeros((y.size, x.size))
    values[1:-1, 1:-1] = psi

    s_lo, s_hi, p_lo, p_hi, m_lo, m_hi = _q1_quadrature(x, y, values, potential, order)
    if check_order is not None and check_order != order:
        # 与 p1_rayleigh_bound 同样的交叉检查：两套 Gauss 规则必须给出同一个
        # 积分（差只在舍入级别）。不一致说明势在单元内不是多项式，求积不再
        # 精确；此时返回 +inf，而不是一个可能被破坏的上界。
        _, _, p2_lo, p2_hi, _, _ = _q1_quadrature(
            x, y, values, potential, check_order
        )
        # 同上：交集为空才是求积不精确的严格证据；区间宽度（padding）造成
        # 的差异不算。2D 的项数是 O(N²)，padding 到 1e-11 是正常的。
        if p_hi < p2_lo or p2_hi < p_lo:
            return RayleighBound(
                upper=math.inf,
                lower=-math.inf,
                numerator_lo=-math.inf,
                numerator_hi=math.inf,
                denominator_lo=0.0,
                denominator_hi=math.inf,
                dimension=int(psi.size),
                exact_quadrature=False,
                reason=(
                    f"两种 Gauss 阶（{order} 与 {check_order}）的势能项区间不相交："
                    f"[{p_lo:.3e}, {p_hi:.3e}] 与 [{p2_lo:.3e}, {p2_hi:.3e}]，"
                    "求积不精确"
                ),
            )
        # 两条规则都精确时两者都是上界，取较高者更安全。
        p_hi = max(p_hi, p2_hi)

    # 区间除法：商的上下界
    num_lo, num_hi = _down(s_lo + p_lo), _up(s_hi + p_hi)
    return RayleighBound(
        upper=_up(num_hi / m_lo),
        lower=_down(num_lo / m_hi),
        numerator_lo=num_lo,
        numerator_hi=num_hi,
        denominator_lo=m_lo,
        denominator_hi=m_hi,
        dimension=int(psi.size),
        exact_quadrature=True,
        reason="",
    )


def box_lower_bound_1d(h: float, index: int) -> float:
    """方阱 ``[0, 1]`` 上第 ``index`` 个（从 0 计）本征值的离散下界。

    对 ``−½u''``、``n`` 个内部节点、``h = 1/(n+1)``，差分算子的本征值恰为
    ``(2/h²)sin²((j+1)πh/2)``，连续本征值为 ``(j+1)²π²/2``。由
    ``sin θ ≤ θ`` 得 ``λ_h ≤ λ_cont`` —— 这是精确的初等不等式，不是数值
    观察。返回的是这个**离散**本征值本身，它同时是 ``λ_cont`` 的下界。
    """
    theta = (index + 1) * math.pi * h / 2.0
    return float(_up(2.0 / h**2 * math.sin(theta) ** 2))


__all__ = [
    "RayleighBound",
    "q1_rayleigh_bound_2d",
    "p1_rayleigh_bound",
    "box_lower_bound_1d",
]
