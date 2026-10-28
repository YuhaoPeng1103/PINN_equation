"""定向舍入（区间算术）的谱计数。

``certify.count_eigenvalues_below`` 与 ``certify2d.count_eigenvalues_below_2d``
用的是浮点主元，其符号在最后一次舍入上并不严格：主元若恰好在舍入边界
附近，符号判定就是猜的。本模块用**向外舍入的区间算术**重做同一件事，
使“小于 σ 的本征值个数”成为一条可证明的陈述，而不是一条在双精度下
几乎总成立的陈述。

区间算术的最小实现
------------------
每个实数用一个 ``Interval``（下界, 上界）表示，四则运算的端点一律用
``math.nextafter`` 向**外**挪一个 ulp。IEEE 754 的四则运算正确舍入，
误差不超过 0.5 ulp，因此外挪一个 ulp 得到的端点一定在真值的外侧。
这不是“高精度计算”：精度仍是双精度，只是每一次舍入都被逼到了对
结论有利的那一侧。

一维：三对角行列式递推
----------------------
对对称三对角 T = H_h − σI，令 d_0 = 1、d_1 = a_1 − σ、
d_k = (a_k − σ)d_{k−1} − b_{k−1}² d_{k−2}，则 T 的小于 σ 的本征值个数
等于 (d_0, …, d_n) 中符号改变的次数（Gantmacher；d_0 = +1 是约定）。
这条递推只用乘、减，不用除法，因此区间算术下不会因除以零而失去严格性。
量级用**只含乘除 2 的幂**的缩放逐项归一：乘以 2^m 在浮点里是精确的，
正因子不改变符号，于是递推既不会上溢也不会损失严格性。

二维：带状 LDLᵀ 的惯性
----------------------
五对角（自然序下带宽为 N_y）矩阵的无主元 LDLᵀ：负主元个数等于小于 σ 的
本征值个数（Sylvester 惯性定律）。与一维的差别是要做除法，但主元若已被
区间判定为严格同号，除法就是安全的。任一主元包含 0 时，计数**拒绝**
（``certified=False``），而不是像浮点版那样扰动 shift 再试。

口径（务必与论文一致）
----------------------
被认证的矩阵是**实现里实际存下来的那个双精度矩阵**：对角元
``1/h² + V_i``、副对角 ``−0.5/h²``（二维为 ``−0.5/h²`` 与
``0.5(2/h_x² + 2/h_y²) + V_ij``），每一个都当作一个精确实数。
因此“计数严格”的含义是：对以这些浮点数为元素的三对角算子严格，
而不是对某个用无误差实数写出的理想算子严格。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

_NEG_INF = float("-inf")
_POS_INF = float("inf")


def _down(value: float) -> float:
    """把浮点数向 −∞ 挪一个 ulp。"""
    return math.nextafter(value, _NEG_INF)


def _up(value: float) -> float:
    """把浮点数向 +∞ 挪一个 ulp。"""
    return math.nextafter(value, _POS_INF)


@dataclass(frozen=True)
class Interval:
    """闭区间 [lo, hi]，四则运算一律向外舍入。"""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (self.lo <= self.hi):
            raise ValueError(f"区间下界大于上界: [{self.lo}, {self.hi}]")

    @staticmethod
    def point(value: float) -> "Interval":
        value = float(value)
        return Interval(value, value)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        if any(math.isnan(p) for p in products):
            return Interval(_NEG_INF, _POS_INF)
        return Interval(_down(min(products)), _up(max(products)))

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.lo <= 0.0 <= other.hi:
            return Interval(_NEG_INF, _POS_INF)
        quotients = (
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        )
        if any(math.isnan(q) for q in quotients):
            return Interval(_NEG_INF, _POS_INF)
        return Interval(_down(min(quotients)), _up(max(quotients)))

    def sign(self) -> int | None:
        """严格符号：−1/0/+1；区间跨零或含非有限端点时返回 ``None``。"""
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            return None
        if self.hi < 0.0:
            return -1
        if self.lo > 0.0:
            return 1
        return None

    @property
    def magnitude(self) -> float:
        return max(abs(self.lo), abs(self.hi))

    def to_pow2_scale(self) -> "Interval":
        """乘以 2 的幂使最大幅值落在 [0.5, 1)。

        只在缩放精确（无上溢、无次正规舍入）时执行；否则原样返回，
        让后续的溢出以非有限端点表现出来，从而被 ``sign`` 判为不确定。
        """
        m = self.magnitude
        if m == 0.0 or not math.isfinite(m):
            return self
        exponent = math.frexp(m)[1]
        lo = math.ldexp(self.lo, -exponent)
        hi = math.ldexp(self.hi, -exponent)
        if math.ldexp(lo, exponent) != self.lo or math.ldexp(hi, exponent) != self.hi:
            return self
        return Interval(lo, hi)

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi}


def _rescale_pair(first: "Interval", second: "Interval") -> tuple["Interval", "Interval"]:
    """用同一个 2 的幂同时缩放两个区间，保持递推的值不变、只改量级。

    缩放因子对两项相同，因此 (a·q_k − b²·q_{k−1}) 的符号序列不受影响。
    只有缩放精确（无上溢、无次正规舍入）时才执行；否则原样返回，让后续的
    溢出以非有限端点出现并被判为不确定。
    """
    magnitude = max(first.magnitude, second.magnitude)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return first, second
    exponent = math.frexp(magnitude)[1]
    scaled = []
    for value in (first, second):
        lo = math.ldexp(value.lo, -exponent)
        hi = math.ldexp(value.hi, -exponent)
        if math.ldexp(lo, exponent) != value.lo or math.ldexp(hi, exponent) != value.hi:
            return first, second
        scaled.append(Interval(lo, hi))
    return scaled[0], scaled[1]


@dataclass(frozen=True)
class CountCertificate:
    """一次定向舍入计数及其可审计的字段。"""

    count: int
    certified: bool
    dimension: int
    shift: float
    reason: str = ""
    min_relative_margin: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 一维
# ---------------------------------------------------------------------------


def _tridiagonal_entries(
    x: np.ndarray,
    potential: np.ndarray,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if x.size < 3:
        raise ValueError("至少需要 3 个网格点")
    if potential.shape != x.shape:
        raise ValueError(f"势形状应为 {x.shape}，收到 {potential.shape}")
    h = float(x[1] - x[0])
    if not np.allclose(np.diff(x), h, rtol=1e-12, atol=0.0):
        raise ValueError("x 必须是等距网格")
    diag = 1.0 / h**2 + potential[1:-1]
    off = -0.5 / h**2
    return diag, off


def certified_sturm_count_1d(
    x: np.ndarray,
    potential: np.ndarray,
    shift: float,
) -> CountCertificate:
    """在定向舍入下数 H_h 小于 ``shift`` 的本征值个数。

    返回的 ``certified`` 为 ``True`` 时，``count`` 对以浮点对角元
    ``1/h² + V_i`` 与副对角 ``−0.5/h²`` 为元素的实对称三对角矩阵是
    **定理**。任何前导主子式行列式的区间跨零（shift 落在谱的
    计数分辨率内）都会使计数被拒绝，而不是被取整成某个整数。
    """
    diag, off = _tridiagonal_entries(x, potential)
    shift = float(shift)
    if not math.isfinite(shift):
        raise ValueError("shift 必须是有限数")
    n = int(diag.size)
    if n == 0:
        return CountCertificate(0, True, 0, shift)

    sigma = Interval.point(shift)
    b_squared = Interval.point(off) * Interval.point(off)

    # 主元递推 q_1 = a_1 − σ，q_k = (a_k − σ) − b²/q_{k−1}；负主元个数即
    # 小于 σ 的本征值个数（Sylvester 惯性）。它等价于行列式递推的相邻比，
    # 但没有后者的相消：后者在 k≈40 处相对宽度就超过 1，完全不能用。
    pivot = Interval.point(float(diag[0])) - sigma
    negative = 0
    margin = 1.0

    for k in range(1, n + 1):
        sign = pivot.sign()
        if sign is None:
            return CountCertificate(
                count=-1,
                certified=False,
                dimension=n,
                shift=shift,
                reason=f"第 {k} 个主元的区间跨零",
                min_relative_margin=0.0,
            )
        if sign < 0:
            negative += 1
        if k < n:
            diagonal = Interval.point(float(diag[k])) - sigma
            product = b_squared / pivot
            # |q_k| / max(|a_k − σ|, |b²/q_{k−1}|)：主元相对于两项的残余量，
            # 它的倒数正是区间宽度被放大的倍数。
            margin = min(
                margin,
                pivot.magnitude / max(diagonal.magnitude, product.magnitude, 1e-300),
            )
            pivot = diagonal - product

    return CountCertificate(
        count=int(negative),
        certified=True,
        dimension=n,
        shift=shift,
        reason="",
        min_relative_margin=float(margin),
    )


# ---------------------------------------------------------------------------
# 二维
# ---------------------------------------------------------------------------


def _interval_product(a_lo, a_hi, b_lo, b_hi):
    """逐元素区间乘法，端点向外舍入。"""
    products = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    lo = np.minimum(
        np.minimum(products[0], products[1]), np.minimum(products[2], products[3])
    )
    hi = np.maximum(
        np.maximum(products[0], products[1]), np.maximum(products[2], products[3])
    )
    return np.nextafter(lo, _NEG_INF), np.nextafter(hi, _POS_INF)


def _separable_parts(grid, tolerance: float = 1e-12):
    """把 V(x, y) 拆成 V_x(x) + V_y(y)；不可分时返回 ``None``。"""
    potential = np.asarray(grid.potential, dtype=float)
    if potential.size == 0:
        return None
    x_part = potential[:, 0].copy()
    y_part = potential[0, :] - potential[0, 0]
    residual = np.max(np.abs(potential - (x_part[:, None] + y_part[None, :])))
    scale = max(1.0, float(np.max(np.abs(potential))))
    if residual > tolerance * scale:
        return None
    return x_part, y_part


def certified_eigenvalue_brackets_1d(
    x: np.ndarray,
    potential: np.ndarray,
    count_needed: int,
    upper: float,
    width: float | None = None,
) -> list[tuple[float, float]] | None:
    """用定向舍入的 Sturm 计数二分出前 ``count_needed`` 个本征值的严格区间。

    下界取 ``min V``：对三对角矩阵，Gershgorin 给出
    ``λ ≥ min_i(1/h² + V_i − 2·0.5/h²) = min V``。任何一次计数被拒绝就整体
    返回 ``None``（上层据此拒绝），不做“扰动 shift 再试”。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if width is None:
        h = float(x[1] - x[0])
        width = 1e-13 * max(1.0, 1.0 / h**2)
    brackets: list[tuple[float, float]] = []
    lower = float(np.min(potential)) - 1.0
    for index in range(count_needed):
        upper_bound = lower + 1.0
        step = 1.0
        while True:
            certificate = certified_sturm_count_1d(x, potential, upper_bound)
            if not certificate.certified:
                return None
            if certificate.count > index:
                break
            if upper_bound > upper:
                return None
            step *= 2.0
            upper_bound = min(upper_bound + step, upper)
        low = lower
        high = upper_bound
        while high - low > width:
            middle = 0.5 * (low + high)
            certificate = certified_sturm_count_1d(x, potential, middle)
            if not certificate.certified:
                # 继续二分会把 shift 推进计数分辨率以内，那里主元的区间跨零、
                # 计数被拒。此时 [low, high] 仍然满足
                # count(low) ≤ index < count(high)，是**已验证**的区间，
                # 只是比要求的宽度宽——收下它，而不是把整个结果丢掉。
                break
            if certificate.count > index:
                high = middle
            else:
                low = middle
        brackets.append((low, high))
        lower = high
    return brackets


def certified_separable_count_2d(
    grid,
    shift: float,
    width: float | None = None,
) -> CountCertificate:
    """可分离势 ``V = V_x(x) + V_y(y)`` 的二维严格计数。

    五对角算子这时恰好是 Kronecker 和 ``H_x ⊗ I + I ⊗ H_y``，因此它的谱是
    一维谱逐对相加：``λ_ij = λ_i^{(x)} + λ_j^{(y)}``。每个一维本征值都由
    ``certified_sturm_count_1d`` 二分到严格区间，二维计数就是数“和对”
    小于 σ 的配对。这条路线完全绕开了二维消去带来的区间过估。

    不可分离的势（如 ``anharmonic2d``）返回 ``certified=False``；那里只能用
    浮点惯性计数，并与稠密参考解对照（论文 §6.9）。
    """
    shift = float(shift)
    if not math.isfinite(shift):
        raise ValueError("shift 必须是有限数")
    parts = _separable_parts(grid)
    if parts is None:
        return CountCertificate(
            count=-1,
            certified=False,
            dimension=grid.n_interior,
            shift=shift,
            reason="势不可分离成 V_x(x) + V_y(y)",
        )
    x_part, y_part = parts
    potential_x = x_part.copy()
    potential_y = y_part.copy()
    # 一维因子只用到内部节点，端点值随意；取全长数组以复用一维例程。
    below_x = shift - float(np.min(potential_y))
    below_y = shift - float(np.min(potential_x))
    if below_x <= float(np.min(potential_x)) or below_y <= float(np.min(potential_y)):
        return CountCertificate(0, True, grid.n_interior, shift)

    def _brackets(axis, potential, ceiling):
        certificate = certified_sturm_count_1d(axis, potential, ceiling)
        if not certificate.certified:
            return None
        if certificate.count == 0:
            return []
        return certified_eigenvalue_brackets_1d(
            axis, potential, certificate.count, ceiling, width
        )

    x_brackets = _brackets(grid.x, potential_x, below_x)
    y_brackets = _brackets(grid.y, potential_y, below_y)
    if x_brackets is None or y_brackets is None:
        return CountCertificate(
            count=-1,
            certified=False,
            dimension=grid.n_interior,
            shift=shift,
            reason="一维本征值二分遇到计数拒绝",
        )
    total = 0
    for low_x, high_x in x_brackets:
        for low_y, high_y in y_brackets:
            if high_x + high_y < shift:
                total += 1
            elif low_x + low_y <= shift:
                return CountCertificate(
                    count=-1,
                    certified=False,
                    dimension=grid.n_interior,
                    shift=shift,
                    reason="某个一维和与 shift 无法分辨",
                )
    return CountCertificate(
        count=int(total),
        certified=True,
        dimension=grid.n_interior,
        shift=shift,
    )


def certified_separable_brackets_2d(
    grid,
    index: int = 0,
    hint: float | None = None,
    width: float | None = None,
) -> tuple[float, float] | None:
    """第 ``index`` 个二维本征值的严格区间 ``[lo, hi]``（仅对可分离势）。

    二分 :func:`certified_separable_count_2d`：计数在穿过本征值时从
    ``index`` 跳到 ``index+1``，所以不变式是 ``N(lo) <= index < N(hi)``，
    即 ``lo <= λ_index < hi``。初始区间由 ``hint``（例如稀疏求解器的值）
    向两侧扩张，直到计数确实在两端不同。任何一次计数拒绝都返回 ``None``
    ——这个函数不猜。简并处 ``lo`` 与 ``hi`` 会同时收敛到该本征值，
    区间宽度趋于零，这仍然是一个正确的包含。
    """
    if hint is None:
        raise ValueError("需要一个起始提示值 hint（例如稀疏求解器的结果）")
    hint = float(hint)
    step = 1e-3 * max(1.0, abs(hint))

    lo = hint - step
    span = step
    for _ in range(60):
        certificate = certified_separable_count_2d(grid, lo, width)
        if not certificate.certified:
            return None
        if certificate.count <= index:
            break
        lo -= span
        span *= 2.0
    else:
        return None

    hi = hint + step
    span = step
    for _ in range(60):
        certificate = certified_separable_count_2d(grid, hi, width)
        if not certificate.certified:
            return None
        if certificate.count > index:
            break
        hi += span
        span *= 2.0
    else:
        return None

    for _ in range(80):
        if hi - lo <= 4.0 * float(np.spacing(max(abs(lo), abs(hi), 1e-300))):
            break
        # 用 0.618 而不是 0.5：初始端点对称地取在 hint 两侧，若中点取 0.5
        # 就**正好**落在 hint 上，而 hint 常常就是双精度里的本征值本身，
        # 计数在那里必然拒绝（那是必须拒绝的情形），区间一步也收不紧。
        middle = lo + 0.6180339887498949 * (hi - lo)
        if middle <= lo or middle >= hi:
            break
        certificate = certified_separable_count_2d(grid, middle, width)
        if not certificate.certified:
            # 二分走到了计数分辨率：再往下走 shift 会被区间宽度吃掉，计数
            # 只能拒绝。此时 [lo, hi] 仍然成立（两端都由已认证的计数固定），
            # 收下它，而不是把整个结果丢掉——与一维 bracket 同一策略。
            break
        if certificate.count <= index:
            lo = middle
        else:
            hi = middle
    return float(lo), float(hi)
