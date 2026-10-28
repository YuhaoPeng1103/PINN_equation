"""子空间（簇）证书：把简并态的“拒绝”变成可验证的正面结论。

为什么需要它
------------
``certify.py`` / ``certify2d.py`` 里的 Temple 界要求“区间 [a,b] 内恰有一个
本征值”。二维各向同性势的第一激发态是**结构性简并**的（方阱 (1,2)/(2,1)、
谐振子 (1,0)/(0,1)），离散谱里的分裂只有 1e-13 量级——不存在任何能通过
惯性计数验证的单态区间，于是单态证书必须返回 ``valid=False``。这个拒绝是
正确的，但它把一大类正常问题挡在门外：网络给出的 ψ 只可能等于简并子空间
里的**某个元素**，而“那个元素”本身并不是良定义的对象。

本模块认证的对象因此不是 ψ，而是 span(ψ₁,…,ψ_m)：

* **验证重数**：用已验证的计数 N(b) − N(a) = m 断言 (a,b) 内恰有 m 个
  本征值（计重数）。m 不是网络的输入，而是由谱计数**测出来**的；
* **双向簇围道**：(a,b) 本身就是这 m 个本征值的双侧围道，其宽度只受
  计数分辨率限制（1e-9·scale），与网络的残差无关；
* **逐 Ritz 值定位**：θ_i 与某个簇内本征值的距离 ≤ ρ_i（子空间 Weinstein）；
* **主子角上界**：sin Θ(span X, V_E) ≤ ρ_F/δ。

记号与结论
----------
设 H 为对称矩阵（这里 H = H_h），X ∈ R^{n×m} 的列按离散内积
⟨u,v⟩_h = h^d Σ u_i v_i 标准正交，压缩矩阵 M = XᵀHX 有特征对 (θ_i, y_i)，
Ritz 向量 x_i = X y_i，残差 r_i = Hx_i − θ_i x_i，ρ_i = ‖r_i‖_h，
ρ_F = (Σ ρ_i²)^{1/2} = ‖HX − XM‖_F。设 (a,b) 已验证 N(b) − N(a) = m，
θ_i ∈ (a,b)，δ_i = min(θ_i − a, b − θ_i)，δ = min_i δ_i > 0。

**命题 1（逐 Ritz 值的簇内定位）** 若 ρ_i < δ_i，则 dist(θ_i, σ(H) ∩ (a,b))
≤ ρ_i。证明：x_i 是单位向量，故 dist(θ_i, σ(H)) ≤ ‖Hx_i − θ_i x_i‖ = ρ_i；
而 (a,b) 外的任何本征值到 θ_i 的距离至少是 δ_i > ρ_i，所以最近的本征值
只能落在 (a,b) 内。∎

**命题 2（主子角）** sin Θ(span X, V_E) ≤ ρ_F/δ，其中 V_E 是 (a,b) 对应的
精确不变子空间。证明：把 x_i 按谱投影分解为簇内分量与簇外分量。由
‖r_i‖² = Σ_j (λ_j − θ_i)² c_j²，对簇外 j 有 |λ_j − θ_i| ≥ δ_i，故簇外分量
的范数 ≤ ρ_i/δ_i。Ritz 基 {x_i} 是一组标准正交基，于是
‖P_E^⊥ X‖_F² = Σ_i ‖P_E^⊥ x_i‖² ≤ Σ_i ρ_i²/δ_i² ≤ ρ_F²/δ²，而主子角的正弦
等于 ‖P_E^⊥ X‖₂ ≤ ‖P_E^⊥ X‖_F。∎

注意两条结论都**不需要**非简并假设：m 个 Ritz 值可以彼此任意接近，只要
它们都落在验证过的簇区间内部。这正是单态 Temple 做不到、而简并问题需要
的东西。

与单态证书的关系
----------------
m = 1 时本模块给出的是“Weinstein 区间 + Davis–Kahan 夹角”，比 ``certify.py``
的 Temple 二次界弱（O(ρ) 对 O(ρ²)），因此单态仍应优先走 ``certify_eigenpair``。
本模块的价值在于 m ≥ 2：那里 Temple 界根本不存在。

一个簇证书也可以覆盖**非简并**的一组相邻本征值（例如 harmonic 的 ψ₀,ψ₁,ψ₂）：
只要它们能被一个验证过的区间与其余谱隔开，命题 1、2 同样成立。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .certify import (
    _dirichlet_apply,
    _dirichlet_data,
    _resolution_floor as _resolution_floor_1d,
    _roundoff_margin,
    count_eigenvalues_below,
)
from .certify2d import (
    Grid2D,
    _resolution_floor as _resolution_floor_2d,
    apply_h as _apply_h_2d,
    approximate_spectrum_2d,
    count_eigenvalues_below_2d,
)


@dataclass(frozen=True)
class SubspaceCertificate:
    """一个 m 维试验子空间的后验证书。

    ``valid=True`` 表示以下四条同时成立：

    1. 子空间秩为 m（列在离散内积下线性无关）；
    2. 区间 (a,b) 的计数差经**验证**恰为 m；
    3. 所有 Ritz 值 θ_i 严格落在 (a,b) 内，且 δ > 分辨率下限；
    4. 每个 ρ_i < δ_i（命题 1 的假设）。

    此时 ``angle_bound`` 是命题 2 的上界（ρ_F ≥ δ 时退化为 π/2，仍然成立
    但无信息量）。``valid=False`` 时数值字段可能是 None，调用方必须把
    它读作“未认证”，而不是“误差很小”。
    """

    n_states: int
    ritz_values: tuple[float, ...]
    residual_norms: tuple[float, ...]
    frobenius_residual: float
    interval_lower: float | None
    interval_upper: float | None
    margin: float | None
    angle_bound: float | None
    verified_multiplicity: int | None
    first_index: int | None
    valid: bool
    reason: str = ""

    @property
    def width(self) -> float | None:
        if self.interval_lower is None or self.interval_upper is None:
            return None
        return self.interval_upper - self.interval_lower

    @property
    def enclosures(self) -> tuple[tuple[float, float], ...]:
        """每个 Ritz 值对应的、**含某个簇内本征值**的双侧区间。"""
        return tuple(
            (t - r, t + r) for t, r in zip(self.ritz_values, self.residual_norms)
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["width"] = self.width
        out["enclosures"] = [list(pair) for pair in self.enclosures]
        return out


# ---------------------------------------------------------------------------
# 两个网格适配器：只暴露“内部向量 → 内部向量”的算子作用、内积权重、
# 验证计数、近似谱与分辨率下限。核心算法对一维/二维完全共用。
# ---------------------------------------------------------------------------


class _Problem:
    """把一维/二维网格的差异收敛到四个方法。"""

    kind: str

    def interior(self, psi: np.ndarray) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def apply(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def weight(self) -> float:  # pragma: no cover - 接口
        raise NotImplementedError

    def count(self, shift: float) -> int:  # pragma: no cover - 接口
        raise NotImplementedError

    def approx(self, k: int) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def floor(self) -> float:  # pragma: no cover - 接口
        raise NotImplementedError


class _Problem1D(_Problem):
    kind = "1d"

    def __init__(self, x: np.ndarray, potential: np.ndarray):
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.potential = np.asarray(potential, dtype=float).reshape(-1)
        if self.x.size < 5:
            raise ValueError("至少需要 5 个网格点")
        self.h = float(self.x[1] - self.x[0])
        if self.h <= 0 or not np.allclose(np.diff(self.x), self.h, rtol=1e-10, atol=0.0):
            raise ValueError("x 必须是严格递增的等距网格")
        if self.potential.shape != self.x.shape:
            raise ValueError("potential 形状必须与 x 一致")

    def interior(self, psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=float).reshape(-1)
        if psi.shape != self.x.shape:
            raise ValueError(f"向量形状 {psi.shape} 与网格 {self.x.shape} 不一致")
        return psi[1:-1]

    def apply(self, u: np.ndarray) -> np.ndarray:
        full = np.zeros(self.x.size)
        full[1:-1] = u
        return _dirichlet_apply(full, self.x, self.potential)

    def weight(self) -> float:
        return self.h

    def count(self, shift: float) -> int:
        return count_eigenvalues_below(self.x, self.potential, shift)

    def approx(self, k: int) -> np.ndarray:
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh

        diag, off = _dirichlet_data(self.x, self.potential)
        n = diag.size
        offs = off * np.ones(max(n - 1, 0))
        matrix = diags([offs, diag, offs], [-1, 0, 1], format="csc")
        k = min(int(k), n - 2)
        if k < 1:
            return np.array([float(np.min(diag))])
        values = eigsh(
            matrix, k=k, sigma=float(np.min(diag)) - 1.0, which="LM", tol=1e-11
        )[0]
        return np.sort(np.asarray(values, dtype=float))

    def floor(self) -> float:
        diag, off = _dirichlet_data(self.x, self.potential)
        return _resolution_floor_1d(diag, off)


class _Problem2D(_Problem):
    kind = "2d"

    def __init__(self, grid: Grid2D):
        if not isinstance(grid, Grid2D):
            raise TypeError("grid 必须是 certify2d.Grid2D")
        self.grid = grid

    def interior(self, psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=float)
        if psi.shape != self.grid.shape:
            raise ValueError(f"向量形状 {psi.shape} 与网格 {self.grid.shape} 不一致")
        return psi[1:-1, 1:-1]

    def apply(self, u: np.ndarray) -> np.ndarray:
        full = np.zeros(self.grid.shape)
        full[1:-1, 1:-1] = np.asarray(u, dtype=float).reshape(
            self.grid.shape[0] - 2, self.grid.shape[1] - 2
        )
        return _apply_h_2d(full, self.grid).reshape(-1)

    def weight(self) -> float:
        return self.grid.cell

    def count(self, shift: float) -> int:
        return count_eigenvalues_below_2d(self.grid, shift)

    def approx(self, k: int) -> np.ndarray:
        return approximate_spectrum_2d(self.grid, k)

    def floor(self) -> float:
        return _resolution_floor_2d(self.grid)


# ---------------------------------------------------------------------------
# 线性代数：离散内积下的正交化与 Rayleigh–Ritz
# ---------------------------------------------------------------------------


def _orthonormalize(problem: _Problem, vectors: list[np.ndarray]) -> np.ndarray:
    """按离散内积做**对称正交化**，返回 n×m 的矩阵 Q（列标准正交）。

    对称正交化（G^{-1/2}）而不是 Gram–Schmidt：它对输入的旋转不敏感，
    证书因此与网络的种子/标号无关。秩亏（子空间维数塌缩）时抛 ValueError，
    因为此时“m 维子空间”这个对象根本不存在。
    """
    m = len(vectors)
    if m < 1:
        raise ValueError("至少需要一个向量")
    cols = [np.asarray(problem.interior(v), dtype=float) for v in vectors]
    shape = cols[0].shape
    if any(c.shape != shape for c in cols):
        raise ValueError("所有向量的网格形状必须一致")
    w = problem.weight()
    C = np.stack([c.reshape(-1) for c in cols], axis=1)
    gram = w * (C.T @ C)
    gram = 0.5 * (gram + gram.T)
    values, vecs = np.linalg.eigh(gram)
    top = float(np.max(np.abs(values)))
    if values[0] <= 1e-13 * max(1.0, top):
        raise ValueError(
            f"试验子空间秩亏（Gram 最小特征值 {values[0]:.3e}，"
            f"最大 {top:.3e}）：m 个向量没有张成 m 维子空间"
        )
    inv_sqrt = 1.0 / np.sqrt(values)
    transform = (vecs * inv_sqrt) @ vecs.T
    return C @ transform


def _rayleigh_ritz(problem: _Problem, Q: np.ndarray):
    """返回 (θ, Ritz 向量 X, ρ_i, ρ_F)。"""
    w = problem.weight()
    m = Q.shape[1]
    HQ = np.stack([problem.apply(Q[:, i]) for i in range(m)], axis=1)
    M = w * (Q.T @ HQ)
    M = 0.5 * (M + M.T)
    theta, Y = np.linalg.eigh(M)
    X = Q @ Y
    R = HQ @ Y - X * theta  # 每列是 Hx_i − θ_i x_i
    rho = np.sqrt(w * np.sum(R**2, axis=0))
    rho_f = float(np.sqrt(w * np.sum(R**2)))
    return theta, X, rho, rho_f


# ---------------------------------------------------------------------------
# 区间构造：先提案（近似谱中点），再用计数验证；失败时退回二分
# ---------------------------------------------------------------------------


def _bisect_count(problem: _Problem, lo: float, hi: float, target: int,
                  iterations: int = 60) -> tuple[float, float]:
    """二分到 N(lo) < target ≤ N(hi) 的相邻浮点数区间。"""
    if problem.count(lo) >= target or problem.count(hi) < target:
        raise ValueError("二分端点没有夹住计数跃迁")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if problem.count(mid) >= target:
            hi = mid
        else:
            lo = mid
    return lo, hi


def _cluster_interval(
    problem: _Problem,
    theta: np.ndarray,
    approx: np.ndarray | None,
    lower: float | None,
    upper: float | None,
) -> tuple[float, float, int]:
    """构造并验证一个恰含 m 个本征值、且夹住所有 θ_i 的区间。

    返回 (a, b, N(b)−N(a))；不能验证时抛 ValueError。
    """
    m = int(theta.size)
    theta_lo = float(theta[0])
    theta_hi = float(theta[-1])

    if lower is not None and upper is not None:
        a, b = float(lower), float(upper)
        if not (a < theta_lo and theta_hi < b):
            raise ValueError(
                f"给定区间 ({a:.6g}, {b:.6g}) 没有严格夹住 Ritz 值 "
                f"[{theta_lo:.6g}, {theta_hi:.6g}]"
            )
        return a, b, problem.count(b) - problem.count(a)

    # 提案：用近似谱把端点放在簇两侧的谱隙中点。近似谱只用于**提案**，
    # 合法性始终由 count 决定。
    spec = (
        np.sort(np.asarray(approx, dtype=float).reshape(-1))
        if approx is not None
        else problem.approx(m + 3)
    )
    centre = 0.5 * (theta_lo + theta_hi)
    i0 = int(np.argmin(np.abs(spec - centre)))
    best_start, best_dist = None, math.inf
    for start in range(max(0, i0 - m + 1), min(spec.size - m + 1, i0 + 1)):
        dist = float(np.max(np.abs(spec[start:start + m] - theta)))
        if dist < best_dist:
            best_start, best_dist = start, dist
    if best_start is None:
        raise ValueError("近似谱太短，无法容纳 m 个候选本征值")
    start = best_start
    lo_out = (
        float(spec[start - 1]) if start > 0
        else float(spec[start]) - max(1.0, abs(float(spec[start])))
    )
    hi_out = (
        float(spec[start + m]) if start + m < spec.size
        else float(spec[-1]) + max(1.0, abs(float(spec[-1])))
    )
    a = 0.5 * (lo_out + float(spec[start]))
    b = 0.5 * (float(spec[start + m - 1]) + hi_out)

    def _verified(a: float, b: float) -> bool:
        return a < theta_lo and theta_hi < b and problem.count(b) - problem.count(a) == m

    if _verified(a, b):
        return a, b, m

    # 回退：直接二分出簇两侧的计数跃迁（较慢，只在提案失败时用）。
    try:
        below = _bisect_count(problem, lo_out, float(spec[start]), problem.count(lo_out) + 1)
        above = _bisect_count(problem, float(spec[start + m - 1]), hi_out,
                              problem.count(float(spec[start + m - 1])) + 1)
        a2, b2 = 0.5 * (below[0] + below[1]), 0.5 * (above[0] + above[1])
        if _verified(a2, b2):
            return a2, b2, m
    except ValueError:
        pass
    # 诊断：区分“重数不符”和“子空间跨了多个能级”这两种真正常见的原因。
    n_a, n_b = problem.count(a), problem.count(b)
    if not (a < theta_lo and theta_hi < b):
        raise ValueError(
            f"Ritz 值 [{theta_lo:.6g}, {theta_hi:.6g}] 超出了近似谱给出的 m={m} "
            f"簇窗口 ({a:.6g}, {b:.6g})：给出的向量跨越了多于 {m} 个能级"
            "（例如漏掉了中间的本征值，或把不同簇的态混在了一起）"
        )
    raise ValueError(
        f"簇窗口 ({a:.6g}, {b:.6g}) 内含 {n_b - n_a} 个本征值，而不是 {m} 个："
        "试验子空间的维数与该能级的重数不符"
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _certify(
    problem: _Problem,
    vectors: list[np.ndarray],
    lower: float | None,
    upper: float | None,
    approx: np.ndarray | None,
) -> SubspaceCertificate:
    m = len(vectors)

    def failed(reason: str, **fields) -> SubspaceCertificate:
        base = dict(
            n_states=m,
            ritz_values=tuple(fields.pop("ritz", ())),
            residual_norms=tuple(fields.pop("rho", ())),
            frobenius_residual=float(fields.pop("rho_f", math.nan)),
            interval_lower=None,
            interval_upper=None,
            margin=None,
            angle_bound=None,
            verified_multiplicity=None,
            first_index=None,
            valid=False,
            reason=reason,
        )
        base.update(fields)
        return SubspaceCertificate(**base)

    try:
        Q = _orthonormalize(problem, vectors)
    except ValueError as exc:
        return failed(str(exc))
    theta, _X, rho, rho_f = _rayleigh_ritz(problem, Q)

    try:
        a, b, count = _cluster_interval(problem, theta, approx, lower, upper)
    except ValueError as exc:
        return failed(str(exc), ritz=theta, rho=rho, rho_f=rho_f)

    margin = _roundoff_margin(0.5 * (abs(a) + abs(b)))
    delta = min(float(theta[0]) - a, b - float(theta[-1])) - margin
    floor = problem.floor()

    if not (a < float(theta[0]) and float(theta[-1]) < b):
        return failed(
            f"Ritz 值 [{theta[0]:.9g}, {theta[-1]:.9g}] 没有被验证区间 "
            f"({a:.9g}, {b:.9g}) 严格夹住；子空间跨越了簇边界（例如把簇内"
            "与簇外的态混在了一起）",
            ritz=theta, rho=rho, rho_f=rho_f, interval_lower=a, interval_upper=b,
            margin=delta,
        )
    if count != m:
        return failed(
            f"验证区间 ({a:.6g}, {b:.6g}) 内含 {count} 个本征值，"
            f"而试验子空间是 {m} 维；重数不符（网络漏掉了簇内某个态，"
            "或给出的向量张成了跨簇的混合空间）",
            ritz=theta, rho=rho, rho_f=rho_f,
        )
    if delta <= floor:
        return failed(
            f"验证出的簇边距 δ={delta:.3e} 低于计数的分辨率下限 {floor:.3e}"
            "（簇与相邻谱无法用双精度惯性/Sturm 计数分离），拒绝认证",
            ritz=theta, rho=rho, rho_f=rho_f,
        )
    too_loose = [
        i for i in range(m)
        if rho[i] >= min(float(theta[i]) - a, b - float(theta[i])) - margin
    ]
    if too_loose:
        return failed(
            "Ritz 残差相对簇边距太大：ρ[{}] ≥ δ_i，无法断言 θ_i 的最近本征值"
            "落在簇内（命题 1 的假设不成立）；请改进子空间或放宽 m".format(
                ",".join(map(str, too_loose))
            ),
            ritz=theta, rho=rho, rho_f=rho_f,
            interval_lower=a, interval_upper=b, margin=delta,
        )

    angle = float(math.asin(min(1.0, rho_f / delta))) if delta > 0 else math.pi / 2
    return SubspaceCertificate(
        n_states=m,
        ritz_values=tuple(float(t) for t in theta),
        residual_norms=tuple(float(r) for r in rho),
        frobenius_residual=float(rho_f),
        interval_lower=a,
        interval_upper=b,
        margin=delta,
        angle_bound=angle,
        verified_multiplicity=count,
        first_index=problem.count(a),
        valid=True,
        reason=(
            f"区间由{'一维 Sturm 序列' if problem.kind == '1d' else '二维 Sylvester 惯性'}"
            f"验证恰含 {count} 个本征值；每个 ρ_i < δ_i；主子角界 sin Θ ≤ ρ_F/δ"
        ),
    )


def certify_subspace_1d(
    vectors,
    x: np.ndarray,
    potential: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
    approx: np.ndarray | None = None,
) -> SubspaceCertificate:
    """一维 m 维试验子空间的簇证书（语义见模块文档）。"""
    return _certify(_Problem1D(x, potential), list(vectors), lower, upper, approx)


def certify_subspace_2d(
    vectors,
    grid: Grid2D,
    lower: float | None = None,
    upper: float | None = None,
    approx: np.ndarray | None = None,
) -> SubspaceCertificate:
    """二维 m 维试验子空间的簇证书（语义见模块文档）。"""
    return _certify(_Problem2D(grid), list(vectors), lower, upper, approx)


def principal_angle(X, exact_vectors, weight: float) -> float:
    """span(X) 与 span(exact) 的主子角正弦（**仅用于实验验证**，不是证书）。

    返回 ‖P_E^⊥ X_unit‖₂，其中 X 先按给定权重正交化、P_E 是精确子空间
    的正交投影。证书的 ``angle_bound`` 必须不小于这个数——测试与实验
    都用它做独立核对。
    """
    w = float(weight)
    s = math.sqrt(w)
    # 按 sqrt(w) 缩放后，离散内积下的正交性变成欧氏正交性；投影算子也
    # 变成欧氏投影，谱范数因此自动 ≤ 1。
    C = s * np.stack([np.asarray(v, dtype=float).reshape(-1) for v in X], axis=1)
    gram = C.T @ C
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    Q = C @ ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T)
    E = s * np.stack(
        [np.asarray(v, dtype=float).reshape(-1) for v in exact_vectors], axis=1
    )
    P_E_Q = E @ np.linalg.solve(E.T @ E, E.T @ Q)
    return float(np.linalg.norm(Q - P_E_Q, 2))


__all__ = [
    "SubspaceCertificate",
    "certify_subspace_1d",
    "certify_subspace_2d",
    "principal_angle",
]
