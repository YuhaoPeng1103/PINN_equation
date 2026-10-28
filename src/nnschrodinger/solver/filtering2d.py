"""二维证书引导的谱滤波：把神经本征函数投影到低阶二维正弦子空间。

一维版（``filtering.py``）把内部点做 DST-I 截断；二维版在 x、y 两个方向
分别做 DST-I，得到

    ψ_K(x_i, y_j) = Σ_{k≤Kx} Σ_{l≤Ky} c_{kl} sin(πk i/(Mx+1)) sin(πl j/(My+1)),

即矩形区域上 Dirichlet Laplace 算子的低阶张量积本征基。

截断阶数 (Kx, Ky) **不靠参考解选择**：扫描一组候选，取证书宽度最小、
且认证到与原始 ψ 同一个本征态的候选。二维的候选空间是二维的，逐点全扫
代价高，所以先扫对角 (K,K)，再在最优点附近做一次局部细化。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .certify2d import Grid2D, certify_eigenpair_2d
from .certify import EigenpairCertificate
from .filtering import dst1


def dst1_2d(values: np.ndarray) -> np.ndarray:
    """对二维数组先沿 y、再沿 x 做正交 DST-I（变换可分离且自逆）。"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("dst1_2d 需要二维数组")
    along_rows = np.apply_along_axis(dst1, 1, values)
    return np.apply_along_axis(dst1, 0, along_rows)


def filter_dirichlet_2d(
    psi: np.ndarray, grid: Grid2D, cutoff_x: int, cutoff_y: int
) -> np.ndarray:
    """保留 (k ≤ cutoff_x, l ≤ cutoff_y) 的模式，返回端点为零的同形状数组。"""
    psi = np.asarray(psi, dtype=float)
    if psi.shape != grid.shape:
        raise ValueError(f"psi 形状应为 {grid.shape}，收到 {psi.shape}")
    mx, my = grid.shape[0] - 2, grid.shape[1] - 2
    cutoff_x, cutoff_y = int(cutoff_x), int(cutoff_y)
    if not (1 <= cutoff_x <= mx and 1 <= cutoff_y <= my):
        raise ValueError(
            f"cutoff 必须在 [1, {(mx, my)}] 内，收到 {(cutoff_x, cutoff_y)}"
        )
    coefficients = dst1_2d(psi[1:-1, 1:-1])
    coefficients[cutoff_x:, :] = 0.0
    coefficients[:, cutoff_y:] = 0.0
    out = np.zeros_like(psi)
    out[1:-1, 1:-1] = dst1_2d(coefficients)
    out[0, :] = out[-1, :] = 0.0
    out[:, 0] = out[:, -1] = 0.0
    return out


def transported_angle_bound_2d(
    psi: np.ndarray,
    grid: Grid2D,
    psi_filtered: np.ndarray,
    filtered_angle_bound: float,
) -> float:
    """二维版搬运引理：sin∠(u,φ) ≤ 2‖u−u_K‖ + sin∠(u_K,φ)。

    证明与一维完全相同（三角不等式），范数换成 h_x h_y Σ|·|²。
    """
    psi = np.asarray(psi, dtype=float)
    psi_filtered = np.asarray(psi_filtered, dtype=float)
    if psi.shape != grid.shape or psi_filtered.shape != grid.shape:
        raise ValueError("psi / psi_filtered 形状必须与网格一致")
    sin_filtered = math.sin(float(filtered_angle_bound))
    if sin_filtered < 0.0:
        raise ValueError(f"夹角界必须非负，收到 {filtered_angle_bound}")
    a = psi[1:-1, 1:-1]
    b = psi_filtered[1:-1, 1:-1]
    na = float(np.sqrt(grid.cell * np.sum(a**2)))
    nb = float(np.sqrt(grid.cell * np.sum(b**2)))
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("ψ 或 ψ_filtered 的范数为零，夹角无定义")
    distance = float(np.sqrt(grid.cell * np.sum((a / na - b / nb) ** 2)))
    return float(math.asin(min(1.0, 2.0 * distance + sin_filtered)))


BASE_CUTOFFS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80]


def default_cutoff_pairs(
    n_interior: tuple[int, int], max_cutoff: int | None = None
) -> list[tuple[int, int]]:
    """对角候选 (K, K)，去重升序。"""
    mx, my = n_interior
    limit = min(mx, my) if max_cutoff is None else min(mx, my, int(max_cutoff))
    return [(k, k) for k in sorted({k for k in BASE_CUTOFFS if k <= limit})]


def refine_pairs(
    best: tuple[int, int],
    n_interior: tuple[int, int],
    offsets: tuple[int, ...] = (-8, -4, -2, -1, 1, 2, 4, 8),
) -> list[tuple[int, int]]:
    """在最优对角点附近做局部二维细化（各向异性态需要 Kx≠Ky）。"""
    mx, my = n_interior
    pairs: set[tuple[int, int]] = set()
    kx, ky = best
    for dx in offsets:
        for dy in offsets:
            a, b = kx + dx, ky + dy
            if 1 <= a <= mx and 1 <= b <= my and (a, b) != best:
                pairs.add((a, b))
    return sorted(pairs)


@dataclass(frozen=True)
class GuidedFilterResult2D:
    """一次二维证书引导滤波的结果。"""

    cutoff: tuple[int, int]
    psi: np.ndarray
    certificate: EigenpairCertificate
    scan: tuple[dict, ...]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "cutoff": list(self.cutoff),
            "certificate": self.certificate.to_dict(),
            "width": self.certificate.width,
            "scan": list(self.scan),
            "reason": self.reason,
        }


def _evaluate(
    psi: np.ndarray,
    grid: Grid2D,
    cutoff: tuple[int, int],
    target_index: int | None,
    approx: np.ndarray | None,
) -> tuple[dict, np.ndarray, EigenpairCertificate]:
    filtered = filter_dirichlet_2d(psi, grid, cutoff[0], cutoff[1])
    cert = certify_eigenpair_2d(filtered, grid, isolate=True, approx=approx)
    same_state = target_index is None or cert.state_index == target_index
    entry = {
        "cutoff": list(cutoff),
        "energy": cert.energy,
        "residual_norm": cert.residual_norm,
        "width": cert.width,
        "valid": cert.valid,
        "state_index": cert.state_index,
        "same_state": same_state,
        "reason": cert.reason,
    }
    return entry, filtered, cert


def guided_filter_2d(
    psi: np.ndarray,
    grid: Grid2D,
    pairs: list[tuple[int, int]] | None = None,
    approx: np.ndarray | None = None,
    refine: bool = True,
    max_cutoff: int | None = None,
) -> GuidedFilterResult2D:
    """扫描二维截断阶数，返回证书宽度最小的滤波结果。

    选择准则与一维版完全一致：只接受 ``valid`` 且与原始 ψ 认证到同一
    ``state_index`` 的候选；全部无效时返回残差最小者并说明原因。
    整个过程不读取参考本征值或参考波函数。
    """
    psi = np.asarray(psi, dtype=float)
    n_interior = (grid.shape[0] - 2, grid.shape[1] - 2)
    candidates = pairs or default_cutoff_pairs(n_interior, max_cutoff)

    reference = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
    target_index = reference.state_index

    scan: list[dict] = []
    best_valid = None
    best_any = None

    def _run(subset):
        nonlocal best_valid, best_any
        for cutoff in subset:
            entry, filtered, cert = _evaluate(psi, grid, cutoff, target_index, approx)
            scan.append(entry)
            if cert.valid and cert.width is not None and entry["same_state"]:
                if best_valid is None or cert.width < best_valid[0]:
                    best_valid = (cert.width, cutoff, filtered, cert)
            elif entry["same_state"]:
                if best_any is None or cert.residual_norm < best_any[0]:
                    best_any = (cert.residual_norm, cutoff, filtered, cert)

    _run(candidates)
    if refine and best_valid is not None:
        refinement = refine_pairs(best_valid[1], n_interior)
        _run([p for p in refinement if p not in set(candidates)])
    elif refine and best_any is not None:
        refinement = refine_pairs(best_any[1], n_interior)
        _run([p for p in refinement if p not in set(candidates)])

    if best_valid is not None:
        width, cutoff, filtered, cert = best_valid
        return GuidedFilterResult2D(
            cutoff=tuple(cutoff),
            psi=filtered,
            certificate=cert,
            scan=tuple(scan),
            reason="取证书宽度最小的二维截断阶数",
        )
    if best_any is not None:
        _, cutoff, filtered, cert = best_any
        return GuidedFilterResult2D(
            cutoff=tuple(cutoff),
            psi=filtered,
            certificate=cert,
            scan=tuple(scan),
            reason="所有截断阶数均不可认证；返回残差最小者",
        )
    cert = certify_eigenpair_2d(psi, grid, isolate=False)
    return GuidedFilterResult2D(
        cutoff=(0, 0), psi=psi.copy(), certificate=cert, scan=tuple(scan),
        reason="没有可用的截断候选",
    )


__all__ = [
    "GuidedFilterResult2D",
    "default_cutoff_pairs",
    "dst1_2d",
    "filter_dirichlet_2d",
    "guided_filter_2d",
    "refine_pairs",
    "transported_angle_bound_2d",
]
