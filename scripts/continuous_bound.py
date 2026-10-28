"""连续算子的双边夹逼：变分上界 + 离散值。

论文 §6.8 原来只写了“证书看不见 ``H_h − H``”。这个脚本把那一句变成一组
可复核的数：

* **上界**：对任意 ``u ∈ H¹_0``，``λ₁(H) ≤ R(u)``（极小极大原理）。取
  ``u`` 为 ``ψ_h`` 的 P1（二维 Q1）插值——插值函数连续、在边界为零，
  所以属于 ``H¹_0``——``R(u)`` 就是 ``λ_cont`` 的一条可计算上界，
  由 ``continuous_bound.p1_rayleigh_bound`` 用定向舍入给出。
* **下界**：方阱上由 ``sin θ ≤ θ`` 精确成立（``box_lower_bound_1d``）；
  其余势上只能取 ``H_h`` 的本征值，而它与 ``λ_cont`` 的关系需要单独论证。
  有限深阱是反例：``V`` 有跳变，差分误差是 ``O(h)`` 且**符号随网格对齐
  而变**，所以离散本征值不是下界（脚本专门列出这一列）。

用法：
    python scripts/continuous_bound.py
输出 runs/continuous_bound/bound.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nnschrodinger.physics.potentials import get_potential  # noqa: E402
from nnschrodinger.solver.certified_count import (  # noqa: E402
    certified_eigenvalue_brackets_1d,
    certified_separable_brackets_2d,
)
from nnschrodinger.solver.certify2d import Grid2D, discrete_eigenvalue_2d  # noqa: E402
from nnschrodinger.solver.continuous_bound import (  # noqa: E402
    box_lower_bound_1d,
    p1_rayleigh_bound,
    q1_rayleigh_bound_2d,
)

GRIDS_1D = (501, 1001, 2001, 4001)
GRIDS_2D = (41, 61, 81, 101)


def _reference_1d(name: str) -> float | None:
    """一维参考值：方阱用 π²/2，有限深阱用超越方程，谐振子用细网格。"""
    if name == "infinite_well":
        return math.pi**2 / 2.0
    if name == "finite_well":
        depth, half = 50.0, 1.0
        # 偶宇称：k tan(ka) = κ, k = √(2(E+U)), κ = √(−2E)，E ∈ (−U, 0)。
        # 方程在 (−U, 0) 内有多个根，必须逐个扫描取**最小**的那个，
        # 否则 brentq 会收敛到任意一根（第一次写就踩了这个坑）。
        def residual(E: float) -> float:
            k = math.sqrt(2.0 * (E + depth))
            return k * math.tan(k * half) - math.sqrt(-2.0 * E)

        grid = np.linspace(-depth + 1e-6, -1e-6, 20001)
        values = [residual(float(e)) for e in grid]
        for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if flo == 0.0:
                return float(lo)
            if flo * fhi < 0.0:
                return brentq(residual, float(lo), float(hi), xtol=1e-15, rtol=8.9e-16)
        raise RuntimeError("有限深阱基态求根失败")
    if name == "harmonic":
        # 截断到训练域 [−6, 6] 后的真值：细网格求解，误差 ~1e-12
        x = np.linspace(-6.0, 6.0, 200_001)
        h = float(x[1] - x[0])
        diag = 1.0 / h**2 + 0.5 * x[1:-1] ** 2
        off = np.full(diag.size - 1, -0.5 / h**2)
        return float(
            eigh_tridiagonal(diag, off, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
    return None


def _reference_2d(name: str) -> float | None:
    if name == "well2d":
        return math.pi**2
    if name == "harmonic2d":
        # 可分离：截断盒 [−3,3]² 上的真值 = 2 × 一维截断真值
        x = np.linspace(-3.0, 3.0, 200_001)
        h = float(x[1] - x[0])
        diag = 1.0 / h**2 + 0.5 * x[1:-1] ** 2
        off = np.full(diag.size - 1, -0.5 / h**2)
        one = float(
            eigh_tridiagonal(diag, off, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
        return 2.0 * one
    return None


# ---------------------------------------------------------------------------
# 一维
# ---------------------------------------------------------------------------

POTENTIALS_1D = ("harmonic", "infinite_well", "finite_well")


def run_1d(sizes) -> list[dict]:
    rows = []
    for name in POTENTIALS_1D:
        spec = get_potential(name)
        reference = _reference_1d(name)
        for N in sizes:
            x = np.linspace(*spec.domain, N)
            potential = spec(x)
            h = float(x[1] - x[0])
            # 离散侧：定向舍入的 Sturm 计数二分给 λ_h 一个严格区间
            brackets = certified_eigenvalue_brackets_1d(
                x, potential, 1, upper=float(potential.min()) + 10.0 * max(1.0, 1.0 / h**2)
            )
            discrete_lo, discrete_hi = brackets[0] if brackets else (math.nan, math.nan)
            # 上界侧：P1 插值的 Rayleigh 商
            diag = 1.0 / h**2 + potential[1:-1]
            off = np.full(diag.size - 1, -0.5 / h**2)
            _, vectors = eigh_tridiagonal(diag, off, eigvals_only=False)
            bound = p1_rayleigh_bound(
                x, spec, vectors[:, 0], breakpoints=spec.discontinuities
            )
            lower = box_lower_bound_1d(h, 0) if name == "infinite_well" else discrete_lo
            lower_kind = "sine-inequality" if name == "infinite_well" else "discrete-eigenvalue"
            contains = None
            if reference is not None and math.isfinite(bound.upper):
                contains = bool(lower <= reference <= bound.upper)
            rows.append(
                {
                    "potential": name,
                    "N": int(N),
                    "h": h,
                    "discrete_lo": discrete_lo,
                    "discrete_hi": discrete_hi,
                    "discrete_error": (0.5 * (discrete_lo + discrete_hi) - reference)
                    if reference is not None
                    else None,
                    "upper": bound.upper,
                    "upper_lower": bound.lower,
                    "bound_width": bound.width,
                    "bracket": bound.upper - lower,
                    "lower_kind": lower_kind,
                    "lower": lower,
                    "reference": reference,
                    "contains_reference": contains,
                    "quadrature_exact": bound.exact_quadrature,
                    "count_certified": brackets is not None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 二维
# ---------------------------------------------------------------------------

POTENTIALS_2D = ("harmonic2d", "well2d", "anharmonic2d")


def _problem_2d(name: str):
    from exp6_certificate_training_2d import _problem2d

    return _problem2d(name)


def run_2d(sizes, verify_counts: bool = False) -> list[dict]:
    rows = []
    for name in POTENTIALS_2D:
        domain, potential, _, _ = _problem_2d(name)
        reference = _reference_2d(name)
        for N in sizes:
            x = np.linspace(*domain, N)
            y = np.linspace(*domain, N)
            X, Y = np.meshgrid(x, y, indexing="ij")
            grid = Grid2D(x=x, y=y, potential=potential(X, Y))
            lanczos, vectors = _eigenpair_2d(grid, 0)
            # 可分离势走 Kronecker 和的定向舍入计数：给出 λ_h 的严格区间。
            # 不可分离势没有这条路线，λ_h 只能是稀疏分解的本征值（论文里明说）。
            certified = None
            if name in ("harmonic2d", "well2d"):
                certified = certified_separable_brackets_2d(grid, 0, hint=lanczos)
            discrete = (
                0.5 * (certified[0] + certified[1]) if certified is not None else lanczos
            )
            verified = None
            if verify_counts and N == min(sizes):
                verified = discrete_eigenvalue_2d(grid, 0)
            bound = q1_rayleigh_bound_2d(x, y, potential, vectors.T)
            contains = None
            if reference is not None:
                contains = bool(discrete - 1e-12 <= reference <= bound.upper)
            rows.append(
                {
                    "potential": name,
                    "N": int(N),
                    "h": float(x[1] - x[0]),
                    "discrete": discrete,
                    "discrete_lanczos": lanczos,
                    "certified_lo": None if certified is None else certified[0],
                    "certified_hi": None if certified is None else certified[1],
                    "count_certified": certified is not None,
                    "discrete_error": (discrete - reference) if reference is not None else None,
                    "upper": bound.upper,
                    "bound_width": bound.width,
                    "bracket": bound.upper - discrete,
                    "reference": reference,
                    "contains_reference": contains,
                    "verified_count": verified,
                    "verified_agrees": None if verified is None else bool(abs(verified - discrete) < 1e-9),
                    "quadrature_exact": bool(bound.exact_quadrature),
                }
            )
    return rows


def run_alignment(sizes) -> list[dict]:
    """有限深阱：离散本征值误差的**符号**随跳点与网格的对齐而变。

    五点格式在跳点附近只有一阶精度，且误差项的符号由 ``V`` 在节点上的取值
    决定，所以 ``λ_h`` 既可能高于也可能低于 ``λ_cont``。这里把符号、
    ``V`` 在跳点邻域节点上的采样（是否出现“洞”）和误差大小一起落盘，
    供论文 §6.5 引用。
    """
    spec = get_potential("finite_well")
    reference = _reference_1d("finite_well")
    rows = []
    for N in sizes:
        x = np.linspace(*spec.domain, N)
        h = float(x[1] - x[0])
        potential = spec(x)
        diag = 1.0 / h**2 + potential[1:-1]
        off = np.full(diag.size - 1, -0.5 / h**2)
        discrete = float(
            eigh_tridiagonal(diag, off, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
        worst = []
        for jump in spec.discontinuities:
            cells = (jump - spec.domain[0]) / h
            nearest = int(round(cells))
            node = float(spec.domain[0] + nearest * h)
            # “井内”指靠近势中心的一侧；两侧跳点的内测方向相反，不能一律减。
            inward = jump - 1e-9 * np.sign(jump)
            worst.append(
                {
                    "jump": jump,
                    "cells": cells,
                    "offset_h": cells - round(cells),
                    "nearest_node": node,
                    "v_at_node": float(spec(np.array([node]))[0]),
                    "v_inside": float(spec(np.array([inward]))[0]),
                }
            )
        # 同一张网格、只改“恰好落在跳点上的那个节点算井内还是井外”，
        # 用来把误差**归因**于那一个节点的取值，而不是笼统地说 O(h)。
        closed = np.where(np.abs(x) <= 1.0, -50.0, potential)
        diag_closed = 1.0 / h**2 + closed[1:-1]
        discrete_closed = float(
            eigh_tridiagonal(diag_closed, off, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
        error = discrete - reference
        rows.append(
            {
                "N": int(N),
                "h": h,
                "discrete": discrete,
                "reference": reference,
                "error": error,
                "sign": "above" if error > 0 else "below",
                "jumps": worst,
                # “洞”只在节点恰好落在跳点上时才有定义：此时该节点取井外的
                # 值，等于把井削掉一个格点。偏移半格时最近节点并不在跳点上。
                "hole_at_jump": any(
                    abs(j["offset_h"]) < 1e-9 and j["v_at_node"] > j["v_inside"] + 1.0
                    for j in worst
                ),
                "discrete_closed": discrete_closed,
                "error_closed": discrete_closed - reference,
                "convention_shift": discrete_closed - discrete,
            }
        )
    return rows


def _eigenpair_2d(grid: Grid2D, index: int):
    """``H_h`` 的第 ``index`` 个本征对（稀疏 Lanczos）。

    返回的向量形状为 ``(n_x−2, n_y−2)``——与 ``_sparse_operator`` 的行主序
    编号一致。``scripts/continuous_bound.py --verify-counts`` 会用
    ``discrete_eigenvalue_2d``（惯性计数二分）抽查本征值。
    """
    import scipy.sparse.linalg as sla

    from nnschrodinger.solver.certify2d import _sparse_operator

    operator = _sparse_operator(grid, 0.0).tocsc()
    values, vectors = sla.eigsh(operator, k=index + 1, which="SA", tol=0)
    order = np.argsort(values)
    vector = np.asarray(vectors[:, order[index]], dtype=float)
    return float(values[order[index]]), vector.reshape(grid.x.size - 2, grid.y.size - 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/continuous_bound/bound.json")
    parser.add_argument(
        "--verify-counts",
        action="store_true",
        help="在最小二维网格上用惯性计数二分复核 Lanczos 本征值",
    )
    parser.add_argument(
        "--alignment",
        action="store_true",
        help="只跑有限深阱的网格对齐研究，写 runs/continuous_bound/alignment.json",
    )
    args = parser.parse_args()

    if args.alignment:
        sizes = list(range(1000, 1012)) + [4001]
        rows = run_alignment(sizes)
        out = ROOT / "runs/continuous_bound/alignment.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "potential": "finite_well",
            "reference": rows[0]["reference"],
            "grids": sizes,
            "rows": rows,
            # 论文引用的是 N=1000…1011 这十二张网格的符号分布；最后一行
            # N=4001 是“跳点正好落在节点上”的单例，不计入该窗口。
            "window": [1000, 1011],
            "above_in_window": sum(
                1 for r in rows if r["sign"] == "above" and 1000 <= r["N"] <= 1011
            ),
            "below_in_window": sum(
                1 for r in rows if r["sign"] == "below" and 1000 <= r["N"] <= 1011
            ),
            "max_abs_error": max(abs(r["error"]) for r in rows),
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = ["finite_well  N     error        sign   offset/h   hole   err(closed)"]
        for row in rows:
            offset = max(abs(j["offset_h"]) for j in row["jumps"])
            lines.append(
                f"            {row['N']:6d} {row['error']:+.3e}  {row['sign']:5s}"
                f"  {offset:8.4f}  {str(row['hole_at_jump']):5s}  {row['error_closed']:+.3e}"
            )
        (out.parent / "alignment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "wrote runs/continuous_bound/alignment.json"
            f" (N=1000..1011: {payload['above_in_window']} above,"
            f" {payload['below_in_window']} below)"
        )
        return 0

    rows_1d = run_1d(GRIDS_1D)
    rows_2d = run_2d(GRIDS_2D, verify_counts=args.verify_counts)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"grids_1d": list(GRIDS_1D), "grids_2d": list(GRIDS_2D),
               "rows_1d": rows_1d, "rows_2d": rows_2d}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["1D", "potential        N     lambda_h        upper        bracket   contains"]
    for row in rows_1d:
        lines.append(
            f"{row['potential']:15s}{row['N']:6d} {0.5*(row['discrete_lo']+row['discrete_hi']):.10f}"
            f" {row['upper']:.10f} {row['bracket']:.2e} {row['contains_reference']}"
        )
    lines.append("2D")
    for row in rows_2d:
        lines.append(
            f"{row['potential']:15s}{row['N']:6d} {row['discrete']:.10f}"
            f" {row['upper']:.10f} {row['bracket']:.2e} {row['contains_reference']}"
        )
    (out.parent / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.parent.name}/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
