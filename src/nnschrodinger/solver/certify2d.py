"""二维定态薛定谔方程的离散后验证书（原型）。

与一维 ``certify.py`` 的关系
---------------------------
一维版本用三对角 LDLᵀ 的 Sturm 计数（O(N)）验证 Temple 区间。二维五点
差分 Hamiltonian 不再是三对角，但 Sylvester 惯性定律仍然成立：

    若对称 A = LU 且**没有行列交换**，则 U 的对角元符号即 A 的惯性。

本模块用 SuperLU 关闭部分主元（``diag_pivot_thresh=0``、自然排序）做
稀疏 LU，并在每次分解后检查 ``perm_r/perm_c`` 是否为单位置换、主元是否
退化为零；任一条件不满足就按相对尺度扰动 shift 重试，绝不把不可靠的
惯性当成计数。稀疏 LU 的填充带宽约等于行内点数 p=N−2，成本 O(n p²)
（n=(N−2)²）；N=81 时单次计数约 30 ms，足够做区间验证。

适用范围（与一维版完全一致，必须写进论文）
------------------------------------------
* 证书对**离散** H_h 严格；到连续算子还有 O(h²) 的离散误差。
* 只处理非简并、被谱隙隔离的本征值。二维各向同性势的第一激发态常是
  二重简并（如方阱的 (1,2)/(2,1)），此时任何单态区间都不存在，
  ``certify_eigenpair_2d`` 返回 ``valid=False``——这是**正确行为**，
  不是数值失败。简并子空间需要子空间版 Temple 界，本项目尚未实现。
* 边界固定为 Dirichlet：网格最外圈 ψ=0。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .certify import (
    EigenpairCertificate,
    IsolatingInterval,
    _roundoff_margin,
    eigenfunction_angle_bound,
    temple_bounds,
    weinstein_interval,
)


@dataclass(frozen=True)
class Grid2D:
    """矩形张量网格与势能采样。``potential[i, j] = V(x_i, y_j)``。"""

    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        potential = np.asarray(self.potential, dtype=float)
        if x.size < 3 or y.size < 3:
            raise ValueError("每个方向至少需要 3 个网格点")
        for name, axis in (("x", x), ("y", y)):
            step = float(axis[1] - axis[0])
            if step <= 0.0 or not np.allclose(np.diff(axis), step, rtol=1e-10, atol=0.0):
                raise ValueError(f"{name} 必须是严格递增的等距网格")
        if potential.shape != (x.size, y.size):
            raise ValueError(
                f"potential 形状应为 {(x.size, y.size)}，收到 {potential.shape}"
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "potential", potential)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.x.size, self.y.size)

    @property
    def hx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def hy(self) -> float:
        return float(self.y[1] - self.y[0])

    @property
    def n_interior(self) -> int:
        return (self.x.size - 2) * (self.y.size - 2)

    @property
    def cell(self) -> float:
        return self.hx * self.hy


def _check_psi(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    if psi.shape != grid.shape:
        raise ValueError(f"psi 形状应为 {grid.shape}，收到 {psi.shape}")
    return psi


def _dirichlet(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    """把最外圈投影为零（证书的 H_h 采用 Dirichlet 边界）。"""
    psi = _check_psi(psi, grid).copy()
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0
    return psi


def normalize_2d(psi: np.ndarray, grid: Grid2D) -> tuple[np.ndarray, float]:
    """内部节点的离散 L2 归一化，返回 (psi_unit, norm)。"""
    psi = _dirichlet(psi, grid)
    interior = psi[1:-1, 1:-1]
    norm = float(np.sqrt(grid.cell * np.sum(interior**2)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("波函数范数为零或非有限，无法归一化")
    return psi / norm, norm


def apply_h(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    """H_hψ 在内部节点上的值（五点差分 + Dirichlet 边界）。"""
    psi = _dirichlet(psi, grid)
    interior = psi[1:-1, 1:-1]
    lap = (
        (psi[2:, 1:-1] - 2.0 * interior + psi[:-2, 1:-1]) / grid.hx**2
        + (psi[1:-1, 2:] - 2.0 * interior + psi[1:-1, :-2]) / grid.hy**2
    )
    return -0.5 * lap + grid.potential[1:-1, 1:-1] * interior


def discrete_rayleigh_quotient_2d(psi: np.ndarray, grid: Grid2D) -> float:
    """同一离散 Hamiltonian 下的 Rayleigh 商 η_h = ⟨ψ, H_hψ⟩ / ⟨ψ, ψ⟩。"""
    psi_unit, _ = normalize_2d(psi, grid)
    h_psi = apply_h(psi_unit, grid)
    interior = psi_unit[1:-1, 1:-1]
    return float(grid.cell * np.sum(interior * h_psi))


def residual_norm_2d(psi: np.ndarray, grid: Grid2D, energy: float) -> float:
    """离散 L2 残差 ‖H_hψ − Eψ‖₂（ψ 先按内部节点归一化）。"""
    psi_unit, _ = normalize_2d(psi, grid)
    r = apply_h(psi_unit, grid) - float(energy) * psi_unit[1:-1, 1:-1]
    return float(np.sqrt(grid.cell * np.sum(r**2)))


def _sparse_operator(grid: Grid2D, shift: float):
    """内部未知量上的 H_h − shift·I（稀疏 CSC，行主序编号）。"""
    from scipy import sparse

    nx, ny = grid.shape
    mx, my = nx - 2, ny - 2
    n = mx * my
    if n == 0:
        raise ValueError("网格没有内部节点")
    v = grid.potential[1:-1, 1:-1]
    diag = 0.5 * (2.0 / grid.hx**2 + 2.0 / grid.hy**2) + v - float(shift)
    r = np.arange(n).reshape(mx, my)
    rows = [r.ravel()]
    cols = [r.ravel()]
    vals = [diag.ravel()]

    def _link(src, dst, weight):
        rows.append(src.ravel())
        cols.append(dst.ravel())
        vals.append(np.full(src.size, weight))

    ox = -0.5 / grid.hx**2
    oy = -0.5 / grid.hy**2
    _link(r[:-1, :], r[1:, :], ox)
    _link(r[1:, :], r[:-1, :], ox)
    _link(r[:, :-1], r[:, 1:], oy)
    _link(r[:, 1:], r[:, :-1], oy)
    matrix = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return matrix.tocsc()


def count_eigenvalues_below_2d(grid: Grid2D, shift: float) -> int:
    """数 H_h 小于 ``shift`` 的本征值个数（稀疏 LU 的主元符号）。

    只有在 ``perm_r/perm_c`` 均为单位置换时，U 的对角元才等于 LDLᵀ 的 D；
    发生主元交换或主元退化时扰动 shift 重试。返回的是浮点意义下的严格
    计数；shift 恰好落在谱点上时按“≤ shift”计。
    """
    from scipy.sparse.linalg import splu

    shift = float(shift)
    if not np.isfinite(shift):
        raise ValueError("shift 必须是有限数")
    n = grid.n_interior
    if n == 0:
        return 0
    scale = max(
        1.0,
        abs(shift),
        0.5 * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        + float(np.max(np.abs(grid.potential))),
    )
    tol = 1e-13 * scale
    identity = np.arange(n)
    for attempt in range(6):
        sigma = shift if attempt == 0 else shift + scale * 1e-11 * (10.0 ** (attempt - 1))
        matrix = _sparse_operator(grid, sigma)
        lu = splu(
            matrix,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        no_pivoting = np.array_equal(np.asarray(lu.perm_r), identity) and np.array_equal(
            np.asarray(lu.perm_c), identity
        )
        pivots = lu.U.diagonal()
        if no_pivoting and float(np.min(np.abs(pivots))) > tol:
            return int(np.count_nonzero(pivots < 0.0))
    raise ValueError("二维惯性计数遇到退化主元/主元交换；请扰动 shift 后重试")


def approximate_spectrum_2d(
    grid: Grid2D,
    k: int = 8,
    sigma: float | None = None,
) -> np.ndarray:
    """用 shift-invert Lanczos 取最低 ``k`` 个离散本征值（**仅作提议**）。

    这些值只用来给 Temple 区间挑端点；区间的合法性始终由
    ``count_eigenvalues_below_2d`` 验证。它们**不是**证书的一部分。
    """
    from scipy.sparse.linalg import eigsh

    k = int(k)
    if k < 1:
        raise ValueError("k 必须为正")
    k = min(k, max(1, grid.n_interior - 2))
    matrix = _sparse_operator(grid, 0.0)
    if sigma is None:
        sigma = float(np.min(grid.potential)) - 1.0
    values, _ = eigsh(
        matrix,
        k=k,
        sigma=float(sigma),
        which="LM",
        tol=1e-10,
        maxiter=max(1000, 20 * grid.n_interior),
    )
    return np.sort(np.asarray(values, dtype=float))


def _bisect_transition_2d(
    grid: Grid2D, lo: float, hi: float, target: int, iterations: int = 60
) -> tuple[float, float]:
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if count_eigenvalues_below_2d(grid, mid) >= target:
            hi = mid
        else:
            lo = mid
    return lo, hi


def _bracket_2d(
    grid: Grid2D, index: int, anchor: float, width: float
) -> tuple[float, float]:
    target = index + 1
    c = count_eigenvalues_below_2d(grid, anchor)
    w = width
    if c < target:
        lo, hi = anchor, anchor + w
        for _ in range(60):
            if count_eigenvalues_below_2d(grid, hi) >= target:
                break
            w *= 2.0
            hi = anchor + w
        else:
            raise ValueError(f"向右找不到第 {index} 个本征值")
    else:
        hi, lo = anchor, anchor - w
        for _ in range(60):
            if count_eigenvalues_below_2d(grid, lo) < target:
                break
            w *= 2.0
            lo = anchor - w
        else:
            raise ValueError(f"向左找不到第 {index} 个本征值")
    return _bisect_transition_2d(grid, lo, hi, target)


def discrete_eigenvalue_2d(
    grid: Grid2D, index: int, approx: np.ndarray | None = None
) -> float:
    """用惯性计数二分求 H_h 的第 ``index`` 个本征值（0 基）。

    与一维 ``discrete_eigenvalue`` 同口径；二维用稀疏 LU 代替三对角 LDLᵀ。
    比直接读 ``approximate_spectrum_2d`` 慢（约 60 次分解），但结果由计数
    验证，可用作独立参考值。
    """
    n = grid.n_interior
    if not (0 <= index < n):
        raise ValueError(f"index={index} 超出 H_h 的谱范围 (0..{n - 1})")
    if approx is not None:
        approx = np.sort(np.asarray(approx, dtype=float).reshape(-1))
        anchor = float(approx[min(index, approx.size - 1)])
    else:
        anchor = float(np.mean(grid.potential))
    width = max(1e-3, 1e-3 * max(1.0, abs(anchor)))
    lo, hi = _bracket_2d(grid, index, anchor, width)
    return float(0.5 * (lo + hi))


def _operator_scale(grid: Grid2D) -> float:
    """H_h 对角元的量级；惯性计数的浮点分辨率与之成正比。"""
    return max(
        1.0,
        0.5 * (2.0 / grid.hx**2 + 2.0 / grid.hy**2)
        + float(np.max(np.abs(grid.potential))),
    )


def _resolution_floor(grid: Grid2D) -> float:
    """双精度下可信的谱隙下限。

    惯性计数在 shift 与某个本征值相距约 ``1e-13·scale`` 时主元会退化，
    重试扰动可达 ``1e-11·scale``；因此当两本征值间距小到 ``1e-14`` 量级
    （二维简并）时，计数无法区分它们。要求验证出的谱隙大于
    ``1e-9·scale``，比扰动大两个数量级，简并/近简并一律拒绝认证。
    """
    return 1e-9 * _operator_scale(grid)


def _verified_interval_from_neighbors(
    grid: Grid2D, energy: float, lower: float, upper: float
) -> IsolatingInterval | None:
    if not (lower < energy < upper):
        return None
    try:
        n_a = count_eigenvalues_below_2d(grid, lower)
        n_b = count_eigenvalues_below_2d(grid, upper)
    except ValueError:
        return None
    if n_b - n_a != 1:
        return None
    gap = min(energy - lower, upper - energy)
    if gap <= _resolution_floor(grid):
        return None
    return IsolatingInterval(
        lower=float(lower), upper=float(upper), state_index=int(n_a), gap=float(gap)
    )


def isolating_interval_2d(
    grid: Grid2D,
    energy: float,
    approx: np.ndarray | None = None,
) -> IsolatingInterval:
    """构造并**验证**一个只含一个本征值、且严格夹住 η 的区间。

    先用近似谱给候选端点（两次计数即可验证），失败时退回一维版那套
    二分构造。任何情况下返回的区间都通过了 ``count`` 验证。
    """
    energy = float(energy)
    if not np.isfinite(energy):
        raise ValueError("energy 必须是有限数")
    n = grid.n_interior
    k = count_eigenvalues_below_2d(grid, energy)
    spec = (
        np.sort(np.asarray(approx, dtype=float).reshape(-1))
        if approx is not None
        else approximate_spectrum_2d(grid, k + 3)
    )

    best: IsolatingInterval | None = None
    for state in (k - 1, k):
        if not (0 <= state < spec.size):
            continue
        # 端点尽量贴近相邻本征值（留 10% 余量），使谱隙下界 γ 尽量大。
        # 近似谱只用来**提议**；合法性由下面的 count 验证。
        if state > 0:
            step = spec[state] - spec[state - 1]
            lower = spec[state - 1] + 0.1 * step
        else:
            step = spec[1] - spec[0] if spec.size > 1 else 1.0
            lower = spec[0] - max(1.0, abs(step))
        if state + 1 < spec.size:
            step_up = spec[state + 1] - spec[state]
            upper = spec[state + 1] - 0.1 * step_up
        else:
            continue
        candidate = _verified_interval_from_neighbors(grid, energy, lower, upper)
        if candidate is not None and (best is None or candidate.gap > best.gap):
            best = candidate
    if best is not None:
        return best

    # 回退：与一维 isolating_interval 相同的二分构造（较多计数，仅在前者失败时用）
    width = max(1e-6, 1e-3 * max(1.0, abs(energy)))
    below = above = None
    if k > 0:
        below = _bracket_2d(grid, k - 1, energy, width)
    if k < n:
        above = _bracket_2d(grid, k, energy, width)
    if k == 0:
        index, bracket = 0, above
    elif k == n:
        index, bracket = n - 1, below
    else:
        assert below is not None and above is not None
        lam_below = 0.5 * (below[0] + below[1])
        lam_above = 0.5 * (above[0] + above[1])
        if energy - lam_below <= lam_above - energy:
            index, bracket = k - 1, below
        else:
            index, bracket = k, above
    assert bracket is not None
    t_lo, t_hi = float(bracket[0]), float(bracket[1])

    if index == 0:
        lower = min(t_lo, energy) - max(width, 1.0)
    else:
        _, prev_hi = _bracket_2d(grid, index - 1, t_lo, width)
        lower = prev_hi
    if index == n - 1:
        upper = max(t_hi, energy) + max(width, 1.0)
    else:
        next_lo, _ = _bracket_2d(grid, index + 1, t_hi, width)
        upper = next_lo

    candidate = _verified_interval_from_neighbors(grid, energy, lower, upper)
    if candidate is None:
        if not (lower < energy < upper):
            raise ValueError(
                f"构造的区间 ({lower:.12g}, {upper:.12g}) 没有严格夹住 η={energy:.12g}"
            )
        n_a = count_eigenvalues_below_2d(grid, lower)
        n_b = count_eigenvalues_below_2d(grid, upper)
        if n_b - n_a != 1:
            raise ValueError(
                f"区间 ({lower:.12g}, {upper:.12g}) 含 {n_b - n_a} 个本征值，"
                "不是单态区间（简并或区间过宽）"
            )
        gap = min(energy - lower, upper - energy)
        raise ValueError(
            f"验证出的谱隙 {gap:.3e} 低于双精度惯性计数的分辨率 "
            f"{_resolution_floor(grid):.3e}（本征值简并或近简并），拒绝认证"
        )
    return candidate


def certify_eigenpair_2d(
    psi: np.ndarray,
    grid: Grid2D,
    lower: float | None = None,
    upper: float | None = None,
    gap: float | None = None,
    isolate: bool = False,
    approx: np.ndarray | None = None,
) -> EigenpairCertificate:
    """二维近似本征对的后验证书；语义与 ``certify_eigenpair`` 完全一致。

    ``isolate=True`` 时自动构造单态区间。若目标本征值简并（二维各向同性
    势的常见情形），区间的计数差会大于 1，返回 ``valid=False`` 并在
    ``reason`` 中说明——这不是数值失败，而是证书拒绝回答不适定的问题。
    """
    psi = _check_psi(psi, grid)
    energy = discrete_rayleigh_quotient_2d(psi, grid)
    residual = residual_norm_2d(psi, grid, energy)
    margin = _roundoff_margin(energy)
    w_lo, w_hi = weinstein_interval(energy, residual)
    w_lo -= margin
    w_hi += margin

    state_index: int | None = None
    if isolate and (lower is None or upper is None):
        try:
            iso = isolating_interval_2d(grid, energy, approx=approx)
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

    try:
        n_lo = count_eigenvalues_below_2d(grid, lower)
        n_hi = count_eigenvalues_below_2d(grid, upper)
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
            reason=f"惯性计数失败: {exc}",
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
            reason=(
                f"区间 ({lower:.6g}, {upper:.6g}) 含 {n_hi - n_lo} 个本征值，"
                "Temple 界不适用（简并或区间过宽）"
            ),
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
        reason=(
            "Temple 区间已由二维惯性计数验证为单态区间"
            + ("；区间由 isolating_interval_2d 自动构造" if isolate else
               "；区间由调用方提供并验证")
        ),
        state_index=state_index,
    )


def certificate_dict(cert: EigenpairCertificate) -> dict:
    """带 ``width`` 字段的字典（与一维脚本的输出格式一致）。"""
    out = cert.to_dict()
    out["width"] = cert.width
    return out


__all__ = [
    "Grid2D",
    "apply_h",
    "approximate_spectrum_2d",
    "certificate_dict",
    "certify_eigenpair_2d",
    "count_eigenvalues_below_2d",
    "discrete_eigenvalue_2d",
    "discrete_rayleigh_quotient_2d",
    "isolating_interval_2d",
    "normalize_2d",
    "residual_norm_2d",
]
