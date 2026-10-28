"""神经本征对的后验误差证书（a posteriori certificates）。

为什么需要它
------------
PINN 论文通常只报告“与参考解的误差”。这有两个问题：

1. **能量准不代表波形对**：与真解正交的污染分量对 Rayleigh 商是一阶不敏感的；
2. **“无监督”可能靠泄漏**：把真实 E 喂进残差项、用解析参考态做 deflation、
   用参考指标做早停，都会让报告误差虚高。

本模块只依赖网络给出的 ψ 和势能 V，用 Sturm 计数构造并**验证**谱隙区间，
给出不依赖真解的双侧误差条。核心是三个经典结果：

* Weinstein 界：若 η = ⟨ψ, Hψ⟩、r = Hψ − ηψ，则谱中至少有一个本征值
  落在 [η − ‖r‖, η + ‖r‖]；
* Temple 界（Kato–Temple）：若区间 [a, b] 内恰有一个本征值 λ、a < η < b，
  且 ‖r‖² < (b−η)(η−a)，则

      η − ‖r‖² / (b − η)  ≤  λ  ≤  η + ‖r‖² / (η − a)

  （下界用上邻居距离、上界用下邻居距离；前提不成立时不给界）

* Davis–Kahan / 谱隙界：若 γ 是 η 到其余谱的距离，则本征函数夹角满足
  sin θ ≤ ‖r‖ / γ。

适用范围（务必写进论文，不要含糊）
----------------------------------
本实现先把 ψ 采样到等距网格上，用二阶中心差分构造离散 Hamiltonian H_h，
因此证书对 **H_h** 是严格的；从 H_h 到连续算子还有 O(h²) 的离散误差，
需要额外的截断误差界。因此证书里的 ``energy`` 与残差都来自同一个 H_h
（``discrete_rayleigh_quotient`` + ``residual_norm``），不能把连续形式的
Rayleigh 商和 H_h 的残差混用。1D 束缚态在截断边界处 ψ≈0，边界影响可忽略；
对无限深阱请把端点设为节点。当前版本不处理简并子空间（需要子空间版
Temple 界），也不处理非自伴算子。

谱隙从哪来：Sturm 计数
-----------------------
Temple 界需要 [a, b] 内“恰有一个本征值”。本模块不靠“近似谱 + 猜 margin”：
对对称三对角 H_h − σI 做 LDLᵀ 分解，负主元个数由 Sylvester 惯性定律给出
小于 σ 的本征值个数。计数是 O(N) 且不需要参考谱，因此可以：

* 用 ``count_eigenvalues_below`` **验证**任意 [a, b] 是否真的只含一个本征值；
  不满足时 ``certify_eigenpair`` 返回 ``valid=False``，而不是给出假界；
* 用 ``isolating_interval`` 二分出相邻本征值的位置，自动构造单态区间，
  并把 γ = min(η−a, b−η) 作为谱隙下界（区间已验证，故这个 γ 是严格的）。

``certify_eigenpair(..., isolate=True)`` 因此是完全自包含的：只给 ψ、x、V
就能出力证书。``heuristic_interval`` 仅保留给实验探索，返回值未经验证，
不能直接当严格证书。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EigenpairCertificate:
    """一个近似本征对的后验证书。

    ``valid=False`` 表示区间假设不成立（η 不在 (a,b) 内、范数为零等），
    此时数值字段可能仍是 None；调用方应把无效证书当作“未认证”，
    而不是当作“误差很小”。
    """

    energy: float
    residual_norm: float
    weinstein_lower: float
    weinstein_upper: float
    interval_lower: float | None
    interval_upper: float | None
    temple_lower: float | None
    temple_upper: float | None
    spectral_gap: float | None
    eigenfunction_angle_bound: float | None
    valid: bool
    reason: str = ""
    state_index: int | None = None

    @property
    def width(self) -> float | None:
        if self.temple_lower is None or self.temple_upper is None:
            return None
        return self.temple_upper - self.temple_lower

    def to_dict(self) -> dict:
        return asdict(self)


def _as_grid(
    x: np.ndarray,
    psi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 5:
        raise ValueError("至少需要 5 个网格点")
    h = float(x[1] - x[0])
    if h <= 0 or not np.allclose(np.diff(x), h, rtol=1e-10, atol=0.0):
        raise ValueError("x 必须是严格递增的等距网格")
    if psi is None:
        return x, None, h
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if x.shape != psi.shape:
        raise ValueError(f"x 与 psi 形状不一致: {x.shape} vs {psi.shape}")
    return x, psi, h


def normalize(psi: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """按 L2 范数归一化，返回 (psi_unit, norm)。"""
    x, psi, _ = _as_grid(x, psi)
    norm = float(np.sqrt(np.trapezoid(psi**2, x)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("波函数范数为零或非有限，无法归一化")
    return psi / norm, norm


def _dirichlet_data(
    x: np.ndarray,
    potential: np.ndarray,
) -> tuple[np.ndarray, float]:
    """H_h 在内部节点上的对称三对角数据（Dirichlet 边界）。"""
    x, _, h = _as_grid(x)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if potential.shape != x.shape:
        raise ValueError(f"potential 形状应为 {x.shape}，收到 {potential.shape}")
    diag = 1.0 / h**2 + potential[1:-1]
    off = -0.5 / h**2
    return diag, off


def _unit_dirichlet(psi: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """内部节点的离散 L2 归一化：‖ψ‖² = h Σ_{i=1}^{N−2} ψ_i²。

    边界值不参与：证书把截断区间端点视为 Dirichlet 节点。对束缚态，
    ψ(端点)≈0，这与 trapezoid 归一化的差别在 O(ψ(端点)²)。
    """
    x, psi, h = _as_grid(x, psi)
    # 证书的 H_h 采用 Dirichlet 边界：先把端点投影为零，避免网络在端点
    # 的非零值通过 i=1 / i=N−2 的差分模板污染证书。
    psi = psi.copy()
    psi[0] = 0.0
    psi[-1] = 0.0
    norm = float(np.sqrt(h * np.sum(psi[1:-1] ** 2)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("波函数范数为零或非有限，无法归一化")
    return psi / norm, norm


def _dirichlet_apply(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
) -> np.ndarray:
    """H_hψ 在内部节点上的值（对称二阶中心差分，Dirichlet 边界）。"""
    x, psi, h = _as_grid(x, psi)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if potential.shape != x.shape:
        raise ValueError(f"potential 形状应为 {x.shape}，收到 {potential.shape}")
    psi = psi.copy()
    psi[0] = 0.0
    psi[-1] = 0.0
    lap = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h**2
    return -0.5 * lap + potential[1:-1] * psi[1:-1]


def continuous_rayleigh_quotient(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
) -> float:
    """连续形式的 Rayleigh 商 E[ψ] = (½∫|ψ′|² + ∫V|ψ|²) / ∫|ψ|²。

    这是“报告给读者看的能量”，用二阶中心差分近似 ψ′，误差 O(h²)。
    它**不**与 ``residual_norm`` 的 H_h 来自同一个离散算子，因此不要拿它
    和 H_h 的残差拼成证书；证书内部请用 ``discrete_rayleigh_quotient``。
    """
    x, psi, _ = _as_grid(x, psi)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if potential.shape != x.shape:
        raise ValueError(f"potential 形状应为 {x.shape}，收到 {potential.shape}")
    psi_unit, _ = normalize(psi, x)
    dpsi = np.gradient(psi_unit, x, edge_order=2)
    kinetic = 0.5 * float(np.trapezoid(dpsi**2, x))
    potential_energy = float(np.trapezoid(potential * psi_unit**2, x))
    return kinetic + potential_energy


def discrete_rayleigh_quotient(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
) -> float:
    """同一离散 Hamiltonian H_h 下的 Rayleigh 商 η_h = h Σ ψ_i (H_h ψ)_i。

    与 ``residual_norm`` 使用完全相同的 H_h（内部节点的对称三对角矩阵，
    Dirichlet 边界），因此 η_h 与 ‖H_hψ − η_hψ‖ 满足 Weinstein/Temple
    定理的前提。在 Dirichlet 边界下 η_h ≥ λ_0(H_h)，这是变分原理的直接
    结果；对连续算子则还有 O(h²) 的离散偏差。
    """
    x, psi, h = _as_grid(x, psi)
    psi_unit, _ = _unit_dirichlet(psi, x)
    h_psi = _dirichlet_apply(psi_unit, x, potential)
    return float(h * np.sum(psi_unit[1:-1] * h_psi))


# 旧名字保留为离散版本：证书的默认口径必须与残差一致。
rayleigh_quotient = discrete_rayleigh_quotient


def residual_norm(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
    energy: float,
) -> float:
    """离散 L2 残差 ‖H_hψ − Eψ‖₂（ψ 先按内部节点归一化）。

    H_h 与 ``discrete_rayleigh_quotient`` 完全一致；残差范数
    r = sqrt(h Σ_i (H_hψ_i − Eψ_i)²)，是 Temple/Weinstein 定理中的 ‖r‖。
    """
    x, psi, h = _as_grid(x, psi)
    psi_unit, _ = _unit_dirichlet(psi, x)
    r = _dirichlet_apply(psi_unit, x, potential) - float(energy) * psi_unit[1:-1]
    return float(np.sqrt(h * np.sum(r**2)))


def count_eigenvalues_below(
    x: np.ndarray,
    potential: np.ndarray,
    shift: float,
) -> int:
    """用 Sturm 序列（LDLᵀ 主元符号）数 H_h 小于 ``shift`` 的本征值个数。

    这是 O(N) 的**严格**计数（在浮点舍入意义下）：对称三对角矩阵
    H_h − shift·I 的 LDLᵀ 分解中，负主元个数等于小于 shift 的本征值个数
    （Sylvester 惯性定律）。它不需要任何参考谱，因此可以用来**验证**
    Temple 区间是否真的只含一个本征值，而不是假设它。

    **恰好在谱点上的 shift**：此时“小于”与“小于等于”无法从浮点数中
    区分，计数本身是不适定的。实现遇到零主元时会把 shift 按相对尺度
    略微**上移**后重试，因此这种情况下返回的是“≤ shift”的计数。用于
    区间验证时请把边界放在谱隙里（``isolating_interval`` 就是这么做的）。
    """
    diag, off = _dirichlet_data(x, potential)
    shift = float(shift)
    if not np.isfinite(shift):
        raise ValueError("shift 必须是有限数")
    n = diag.size
    if n == 0:
        return 0
    scale = max(1.0, abs(shift), float(np.max(np.abs(diag))))
    tol = 1e-13 * scale
    # 若 shift 恰好等于某个前导主子块的本征值，主元会为零。此时按**相对**
    # 尺度扰动 shift 后重试；绝对扰动会在大 h⁻²（如 1/h²=160000）时
    # 落在同一个浮点间隔里，导致 8 次重试都失败（工程记录 16）。
    for attempt in range(6):
        shifted = shift if attempt == 0 else shift + scale * 1e-11 * (10.0 ** (attempt - 1))
        d = diag[0] - shifted
        if abs(d) <= tol:
            continue
        count = int(d < 0.0)
        ok = True
        for i in range(1, n):
            d = (diag[i] - shifted) - off * off / d
            if abs(d) <= tol:
                ok = False
                break
            if d < 0.0:
                count += 1
        if ok:
            return count
    raise ValueError("Sturm 计数遇到退化的主元序列；请扰动 shift 后重试")


@dataclass(frozen=True)
class IsolatingInterval:
    """由 Sturm 计数验证过的单本征值区间。"""

    lower: float
    upper: float
    state_index: int
    gap: float

    def to_dict(self) -> dict:
        return asdict(self)


def _resolution_floor(diag: np.ndarray, off: float) -> float:
    """双精度 Sturm 计数可信的谱隙下限（与二维版同口径）。

    计数在 shift 距谱点约 1e-13·scale 时主元退化，重试扰动可达 1e-11·scale；
    要求验证出的谱隙大于 1e-9·scale，近简并时拒绝认证而不是给出假界。
    """
    scale = max(1.0, float(np.max(np.abs(diag))), 2.0 * abs(float(off)))
    return 1e-9 * scale


def _bisect_transition(
    x: np.ndarray,
    potential: np.ndarray,
    lo: float,
    hi: float,
    target: int,
    iterations: int = 80,
) -> tuple[float, float]:
    """在 count(lo) < target ≤ count(hi) 的前提下二分出计数跃迁点。"""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if count_eigenvalues_below(x, potential, mid) >= target:
            hi = mid
        else:
            lo = mid
    return lo, hi


def _bracket_for_index(
    x: np.ndarray,
    potential: np.ndarray,
    index: int,
    anchor: float,
    width: float,
) -> tuple[float, float]:
    """从 ``anchor`` 出发，二分出第 index 个本征值的内外两侧。

    返回 (lo, hi)，其中 count(lo) ≤ index < count(hi)。
    """
    target = index + 1
    c = count_eigenvalues_below(x, potential, anchor)
    w = width
    if c < target:
        lo = anchor
        hi = anchor + w
        for _ in range(200):
            if count_eigenvalues_below(x, potential, hi) >= target:
                break
            w *= 2.0
            hi = anchor + w
        else:
            raise ValueError(f"向右找不到第 {index} 个本征值")
    else:
        hi = anchor
        lo = anchor - w
        for _ in range(200):
            if count_eigenvalues_below(x, potential, lo) < target:
                break
            w *= 2.0
            lo = anchor - w
        else:
            raise ValueError(f"向左找不到第 {index} 个本征值")
    return _bisect_transition(x, potential, lo, hi, target)


def discrete_eigenvalue(
    x: np.ndarray,
    potential: np.ndarray,
    index: int,
) -> float:
    """用 Sturm 二分求 H_h 的第 ``index`` 个本征值（0 基）。

    先用 Gershgorin 圆盘给出包住整个谱的 [lo, hi]（count(lo)=0，
    count(hi)=n），再二分 count 的跃迁点。复杂度 O(80·N)，N=2001 时
    不到一秒；用于给证书提供**同一离散算子**的参考值。
    """
    diag, off = _dirichlet_data(x, potential)
    n = diag.size
    if not (0 <= index < n):
        raise ValueError(f"index={index} 超出 H_h 的谱范围 (0..{n - 1})")
    radius = 2.0 * abs(off)
    lo = float(np.min(diag) - radius - 1.0)
    hi = float(np.max(diag) + radius + 1.0)
    if count_eigenvalues_below(x, potential, lo) != 0:
        raise ValueError("Gershgorin 下界没有包住谱")
    if count_eigenvalues_below(x, potential, hi) != n:
        raise ValueError("Gershgorin 上界没有包住谱")
    lo_b, hi_b = _bisect_transition(x, potential, lo, hi, index + 1, iterations=100)
    return float(0.5 * (lo_b + hi_b))


def isolating_interval(
    x: np.ndarray,
    potential: np.ndarray,
    energy: float,
    initial_width: float | None = None,
) -> IsolatingInterval:
    """在 η 附近找一个由 Sturm 计数**验证**只含一个本征值的区间。

    做法：令 k = #{λ < η}。分别二分出 η 左右两侧的计数跃迁点
    λ_{k−1}、λ_k，然后选择离 η 更近的那个本征值作为目标，
    把区间端点放在相邻本征值的内侧。返回区间 [a,b]、目标本征值下标
    和谱隙下界 γ = min(η−a, b−η)。

    这不依赖 Numerov/参考谱，也不假设“近似谱就是严格界”。区间验证
    失败时抛出 ``ValueError``，调用方应把它当作“无法认证”，而不是
    “误差很小”。
    """
    x, _, _ = _as_grid(x)
    energy = float(energy)
    if not np.isfinite(energy):
        raise ValueError("energy 必须是有限数")
    diag, _ = _dirichlet_data(x, potential)
    n_states = diag.size
    width = float(initial_width) if initial_width is not None else max(
        1e-6, 1e-3 * max(1.0, abs(energy))
    )
    if width <= 0:
        raise ValueError("initial_width 必须为正")

    k = count_eigenvalues_below(x, potential, energy)

    # 左邻本征值 λ_{k−1} 的内外两侧
    below_lo: float | None = None
    below_hi: float | None = None
    if k > 0:
        w = width
        p = energy - w
        for _ in range(200):
            if count_eigenvalues_below(x, potential, p) < k:
                break
            w *= 2.0
            p = energy - w
        else:
            raise ValueError("向左找不到低于 η 的本征值")
        below_lo, below_hi = _bisect_transition(x, potential, p, energy, k)

    # 右邻本征值 λ_k 的内外两侧
    above_lo: float | None = None
    above_hi: float | None = None
    if k < n_states:
        w = width
        p = energy + w
        for _ in range(200):
            if count_eigenvalues_below(x, potential, p) >= k + 1:
                break
            w *= 2.0
            p = energy + w
        else:
            raise ValueError("向右找不到高于 η 的本征值")
        above_lo, above_hi = _bisect_transition(x, potential, energy, p, k + 1)

    # 选离 η 更近的本征值作为目标，避免 η 略高于 λ_s 时去认证 λ_{s+1}
    if k == 0:
        index = 0
        target_bracket = (above_lo, above_hi)
    elif k == n_states:
        index = n_states - 1
        target_bracket = (below_lo, below_hi)
    else:
        assert below_lo is not None and below_hi is not None
        assert above_lo is not None and above_hi is not None
        lam_below = 0.5 * (below_lo + below_hi)
        lam_above = 0.5 * (above_lo + above_hi)
        if energy - lam_below <= lam_above - energy:
            index = k - 1
            target_bracket = (below_lo, below_hi)
        else:
            index = k
            target_bracket = (above_lo, above_hi)
    assert target_bracket[0] is not None and target_bracket[1] is not None
    t_lo, t_hi = float(target_bracket[0]), float(target_bracket[1])

    # 目标两侧的**邻居**本征值：把端点放得尽量远，谱隙下界才不悲观
    if index == 0:
        a = min(t_lo, energy) - max(width, 1.0)
    else:
        _, prev_hi = _bracket_for_index(x, potential, index - 1, t_lo, width)
        a = prev_hi
    if index == n_states - 1:
        b = max(t_hi, energy) + max(width, 1.0)
    else:
        next_lo, _ = _bracket_for_index(x, potential, index + 1, t_hi, width)
        b = next_lo

    # 端点必须严格夹住 η，并且两个计数只差 1
    if not (a < energy < b):
        raise ValueError(f"构造的区间 ({a:.12g}, {b:.12g}) 没有严格夹住 η={energy:.12g}")
    n_a = count_eigenvalues_below(x, potential, a)
    n_b = count_eigenvalues_below(x, potential, b)
    if n_b - n_a != 1:
        raise ValueError(
            f"区间 ({a:.12g}, {b:.12g}) 含 {n_b - n_a} 个本征值，不是单态区间"
        )
    gap = min(energy - a, b - energy)
    if gap <= 0:
        raise ValueError("谱隙下界非正")
    floor = _resolution_floor(*_dirichlet_data(x, potential))
    if gap <= floor:
        raise ValueError(
            f"验证出的谱隙 {gap:.3e} 低于双精度 Sturm 计数的分辨率 {floor:.3e}"
            "（本征值简并或近简并），拒绝认证"
        )
    return IsolatingInterval(lower=float(a), upper=float(b), state_index=index, gap=float(gap))


def _roundoff_margin(scale: float) -> float:
    """证书的浮点安全裕度（相对 1e-12）。

    Temple/Weinstein 的数学形式是精确的，但 η、r 与区间端点都是双精度
    浮点数。对“精确离散本征态”（r 在舍入噪声量级）区间宽度会小于 η 的
    ulp，直接相加会得到零宽度甚至漏掉真值。这里显式加一个保守裕度，
    并在论文中声明“证书在浮点舍入意义下成立”。对 PINN 的 r≈1e-2，
    这个裕度可以忽略。
    """
    return 1e-12 * max(1.0, abs(float(scale)))


def weinstein_interval(energy: float, residual: float) -> tuple[float, float]:
    """Weinstein 界：谱中至少有一个本征值在 [η−‖r‖, η+‖r‖] 内。"""
    if residual < 0:
        raise ValueError("残差范数不能为负")
    return float(energy - residual), float(energy + residual)


def temple_bounds(
    energy: float,
    residual: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Kato–Temple 双侧界（``lower < η < upper``，区间内恰有一个本征值）。

    若 r = ‖Hψ − ηψ‖ 且 ``r² < (upper−η)(η−lower)``，则

        η − r²/(upper−η) ≤ λ ≤ η + r²/(η−lower)

    **注意分母的配对方向**：下界用**上**邻居的距离，上界用**下**邻居的距离。
    这不是笔误——它来自谱测度的符号分析：η 高于 λ 时污染只能来自上方，
    反之来自下方。把两个分母写反（下界用 η−lower）会给出**高于真值**的
    下界；测试 ``test_temple_bounds_uses_correct_denominators`` 专门锁住这一点。

    这是**条件性**结论：正确性取决于调用方给出的 [lower, upper] 是否
    真的只包含目标本征值。本函数只做算术，不验证谱结构；条件不满足时
    抛 ``ValueError``（此时应报“不可认证”，而不是给出一个看似精确实则无效的区间）。
    """
    if not (lower < energy < upper):
        raise ValueError(f"需要 lower < η < upper，收到 {lower} < {energy} < {upper}")
    if residual < 0:
        raise ValueError("残差范数不能为负")
    r2 = float(residual) ** 2
    below = float(energy) - float(lower)
    above = float(upper) - float(energy)
    if r2 >= below * above:
        raise ValueError(
            f"Temple 条件 r² < (upper−η)(η−lower) 不成立: "
            f"r²={r2:.6g}, 阈值={below * above:.6g}"
        )
    lo = float(energy) - r2 / above
    hi = float(energy) + r2 / below
    return lo, hi


def eigenfunction_angle_bound(residual: float, gap: float) -> float:
    """由 sin θ ≤ ‖r‖/γ 给出的本征函数夹角上界（弧度，∈[0, π/2]）。

    γ 是 η 到目标本征值之外其余谱的距离。简并情形需要子空间版本，
    这里只处理非简并、且目标本征值被谱隙隔离的情形。
    """
    if gap <= 0:
        raise ValueError("谱隙必须为正；简并/近简并请用子空间证书")
    if residual < 0:
        raise ValueError("残差范数不能为负")
    sin_theta = min(1.0, float(residual) / float(gap))
    return float(math.asin(sin_theta))


def certify_eigenpair(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
    gap: float | None = None,
    isolate: bool = False,
) -> EigenpairCertificate:
    """计算一个近似本征对的后验证书。

    参数
    ----
    psi, x, potential:
        等距网格上的波函数与势能采样。
    lower, upper:
        Temple 区间端点。为 None 且 ``isolate=False`` 时只返回 Weinstein
        界与残差范数，此时 ``valid=False``，表示“未认证”，而不是“通过”。
    gap:
        目标本征值到其余谱距离的**下界** γ。若区间已由 Sturm 计数验证
        为单态区间，则 γ 自动取 ``min(η−lower, upper−η)``；若调用方
        另给 gap，则取两者较小值（更保守）。
    isolate:
        为 True 且未显式给区间时，调用 ``isolating_interval`` 自动构造
        一个由 Sturm 计数验证的单态区间。这是完全自包含的证书：不需要
        参考谱，也不把近似本征值当成严格界。

    区间假设会**被验证**而不是被相信：``certify_eigenpair`` 用 Sturm
    计数检查 [lower, upper] 是否恰含一个 H_h 本征值；不满足则返回
    ``valid=False``，绝不静默给出 Temple 界。
    """
    x, psi, _ = _as_grid(x, psi)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    if potential.shape != x.shape:
        raise ValueError(f"potential 形状应为 {x.shape}，收到 {potential.shape}")

    energy = discrete_rayleigh_quotient(psi, x, potential)
    residual = residual_norm(psi, x, potential, energy)
    margin = _roundoff_margin(energy)
    w_lo, w_hi = weinstein_interval(energy, residual)
    w_lo -= margin
    w_hi += margin

    state_index: int | None = None
    if isolate and (lower is None or upper is None):
        try:
            iso = isolating_interval(x, potential, energy)
        except ValueError as exc:
            return EigenpairCertificate(
                energy=energy,
                residual_norm=residual,
                weinstein_lower=w_lo,
                weinstein_upper=w_hi,
                interval_lower=None,
                interval_upper=None,
                temple_lower=None,
                temple_upper=None,
                spectral_gap=None,
                eigenfunction_angle_bound=None,
                valid=False,
                reason=f"无法构造单态区间: {exc}",
            )
        lower, upper = iso.lower, iso.upper
        state_index = iso.state_index
        gap = iso.gap if gap is None else min(float(gap), iso.gap)

    if lower is None or upper is None:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=None,
            interval_upper=None,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=None,
            eigenfunction_angle_bound=None,
            valid=False,
            reason="未提供 Temple 区间 [lower, upper]；只有 Weinstein 界",
        )

    lower = float(lower)
    upper = float(upper)
    if not (lower < energy < upper):
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=None,
            eigenfunction_angle_bound=None,
            valid=False,
            reason=f"η={energy:.6g} 不在区间 ({lower:.6g}, {upper:.6g}) 内",
        )

    # 验证 Temple 的“区间内恰有一个本征值”假设，而不是相信调用方。
    try:
        n_lo = count_eigenvalues_below(x, potential, lower)
        n_hi = count_eigenvalues_below(x, potential, upper)
    except ValueError as exc:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=None,
            eigenfunction_angle_bound=None,
            valid=False,
            reason=f"Sturm 计数失败: {exc}",
        )
    if n_hi - n_lo != 1:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=None,
            eigenfunction_angle_bound=None,
            valid=False,
            reason=(f"区间 ({lower:.6g}, {upper:.6g}) 含 {n_hi - n_lo} 个本征值，"
                    "Temple 界不适用"),
        )
    _floor = _resolution_floor(*_dirichlet_data(x, potential))
    if min(energy - lower, upper - energy) <= _floor:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=None,
            eigenfunction_angle_bound=None,
            valid=False,
            reason=("η 到区间端点的距离低于双精度 Sturm 计数的分辨率 "
                    f"{_floor:.3e}（简并/近简并），拒绝认证"),
        )

    if state_index is None:
        state_index = n_lo
    verified_gap = min(energy - lower, upper - energy) - margin
    gap = verified_gap if gap is None else min(float(gap), verified_gap)
    if gap <= 0:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=gap,
            eigenfunction_angle_bound=None,
            valid=False,
            reason="谱隙下界非正，无法给出 Temple/夹角界",
        )

    try:
        t_lo, t_hi = temple_bounds(energy, residual, lower, upper)
    except ValueError as exc:
        return EigenpairCertificate(
            energy=energy,
            residual_norm=residual,
            weinstein_lower=w_lo,
            weinstein_upper=w_hi,
            interval_lower=lower,
            interval_upper=upper,
            temple_lower=None,
            temple_upper=None,
            spectral_gap=gap,
            eigenfunction_angle_bound=eigenfunction_angle_bound(residual, gap),
            valid=False,
            reason=f"Temple 二次界不可用（残差相对谱隙太大）: {exc}",
            state_index=state_index,
        )
    t_lo -= margin
    t_hi += margin
    angle = eigenfunction_angle_bound(residual, gap)
    return EigenpairCertificate(
        energy=energy,
        residual_norm=residual,
        weinstein_lower=w_lo,
        weinstein_upper=w_hi,
        interval_lower=lower,
        interval_upper=upper,
        temple_lower=t_lo,
        temple_upper=t_hi,
        spectral_gap=gap,
        eigenfunction_angle_bound=angle,
        valid=True,
        reason=("Temple 区间已由 Sturm 计数验证为单态区间"
                + ("；区间由 isolating_interval 自动构造" if isolate else
                   "；区间由调用方提供并验证")),
        state_index=state_index,
    )


def heuristic_interval(
    spectrum: np.ndarray,
    state: int,
    margin: float = 1e-3,
) -> tuple[float, float]:
    """从近似谱构造实验用的 Temple 区间（**不是严格界**）。

    仅用于快速探索：``margin`` 必须覆盖粗谱的误差，否则得到的证书
    只是“看起来很小”的数字，不是严格结论。正式实验请用 Weinstein
    界或区间算术把粗谱转成单侧严格界后再调用 ``certify_eigenpair``。
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=float).reshape(-1))
    if not (0 <= state < spectrum.size):
        raise ValueError(f"state={state} 超出谱范围 (0..{spectrum.size - 1})")
    margin = float(margin)
    if margin <= 0:
        raise ValueError("margin 必须为正")
    lower = spectrum[state] - margin
    upper = spectrum[state] + margin
    # 用相邻态留出间隔；若近似谱本身不单调，这里会显式报错
    if state > 0:
        lower = max(lower, spectrum[state - 1] + margin)
    if state + 1 < spectrum.size:
        upper = min(upper, spectrum[state + 1] - margin)
    if not lower < upper:
        raise ValueError("谱太密，无法用该 margin 构造单态区间；请缩窄 margin 或改子空间证书")
    return lower, upper


__all__ = [
    "EigenpairCertificate",
    "IsolatingInterval",
    "certify_eigenpair",
    "continuous_rayleigh_quotient",
    "count_eigenvalues_below",
    "discrete_eigenvalue",
    "discrete_rayleigh_quotient",
    "eigenfunction_angle_bound",
    "heuristic_interval",
    "isolating_interval",
    "normalize",
    "rayleigh_quotient",
    "residual_norm",
    "temple_bounds",
    "weinstein_interval",
]
