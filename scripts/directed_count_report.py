"""定向舍入计数的统计：一致性、拒绝率、代价。

论文要引用的三个数都在这里：

1. 一维：``certified_sturm_count_1d`` 与浮点计数、稠密特征分解三方一致；
   拒绝只发生在 shift 落在本征值的一个相对宽度以内。
2. 二维可分离：Kronecker 和计数与浮点惯性计数一致；拒绝发生在简并 shift。
3. 二维一般情形：无主元带状区间 LDLᵀ **不可用**——区间过估使主元跨零，
   这里给出拒绝率，作为走可分离路线的证据。

用法：
    python scripts/directed_count_report.py
输出 runs/directed_count/stats.{json,txt}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nnschrodinger.physics.potentials import get_potential  # noqa: E402
from nnschrodinger.solver.certified_count import (  # noqa: E402
    _interval_product,
    certified_separable_count_2d,
    certified_sturm_count_1d,
)
from nnschrodinger.solver.certify import count_eigenvalues_below  # noqa: E402
from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    _sparse_operator,
    count_eigenvalues_below_2d,
)


def report_1d(n_points: int = 2001, n_shifts: int = 121) -> dict:
    rows = []
    for name in ("harmonic", "infinite_well", "finite_well"):
        spec = get_potential(name)
        x = np.linspace(*spec.domain, n_points)
        potential = spec(x)
        h = float(x[1] - x[0])
        diag = 1.0 / h**2 + potential[1:-1]
        off = np.full(diag.size - 1, -0.5 / h**2)
        dense = np.sort(eigh_tridiagonal(diag, off, eigvals_only=True))
        lo = float(dense[0]) - 0.1 * abs(float(dense[0]) + 1.0)
        hi = float(dense[119])
        agree = refused = wrong = 0
        times = []
        refusals = []
        for index, sigma in enumerate(np.linspace(lo, hi, n_shifts)):
            start = time.perf_counter()
            certificate = certified_sturm_count_1d(x, potential, float(sigma))
            times.append(time.perf_counter() - start)
            truth = int(np.sum(dense < sigma))
            if not certificate.certified:
                refused += 1
                # 拒绝必须可审计：记下 shift、它在扫描里的位置、理由，
                # 以及该 shift 与最近本征值的距离（论文声称两者重合）。
                gap = float(np.min(np.abs(dense - float(sigma))))
                refusals.append(
                    {
                        "index": int(index),
                        "of": int(n_shifts),
                        "shift": float(sigma),
                        "reason": certificate.reason,
                        "distance_to_spectrum": gap,
                    }
                )
                continue
            if certificate.count == truth:
                agree += 1
            else:
                wrong += 1
        # 与浮点计数的一致性
        float_agree = sum(
            1
            for sigma in np.linspace(lo, hi, n_shifts)
            if count_eigenvalues_below(x, potential, float(sigma))
            == int(np.sum(dense < float(sigma)))
        )
        rows.append(
            {
                "potential": name,
                "n": int(n_points),
                "shifts": int(n_shifts),
                "certified_agree": int(agree),
                "refused": int(refused),
                "wrong": int(wrong),
                "float_agree": int(float_agree),
                "median_ms": float(np.median(times) * 1e3),
                "max_ms": float(np.max(times) * 1e3),
                "refusals": refusals,
            }
        )
    return {"rows": rows}


def report_determinant(n_points: int = 2001, n_shifts: int = 121) -> dict:
    """区间行列式递推（d_k = (a_k−σ)d_{k−1} − b²d_{k−2}）为什么不能用。

    记录它在每个 shift 上相对宽度首次超过 1 的步号。超过 1 就意味着区间
    跨零，符号不可判定——这条递推在任何一次计数里都跑不完。
    """
    from nnschrodinger.solver.certified_count import Interval, _rescale_pair

    rows = []
    for name in ("harmonic", "infinite_well", "finite_well"):
        spec = get_potential(name)
        x = np.linspace(*spec.domain, n_points)
        potential = spec(x)
        h = float(x[1] - x[0])
        diagonal = 1.0 / h**2 + potential[1:-1]
        off = -0.5 / h**2
        dense = np.sort(
            eigh_tridiagonal(diagonal, np.full(diagonal.size - 1, off), eigvals_only=True)
        )
        lo = float(dense[0]) - 0.1 * abs(float(dense[0]) + 1.0)
        hi = float(dense[min(n_shifts - 2, dense.size - 1)])
        bad = 0
        steps = []
        for sigma in np.linspace(lo, hi, n_shifts):
            previous = Interval.point(1.0)
            current = Interval.point(float(diagonal[0])) - Interval.point(float(sigma))
            squared = Interval.point(off) * Interval.point(off)
            first = None
            for k in range(1, diagonal.size):
                previous, current = _rescale_pair(previous, current)
                nxt = (Interval.point(float(diagonal[k])) - Interval.point(float(sigma))) * current - squared * previous
                width = (nxt.hi - nxt.lo) / max(abs(nxt.lo), abs(nxt.hi), 1e-300)
                if width > 1.0:
                    first = k
                    break
                previous, current = current, nxt
            if first is not None:
                bad += 1
                steps.append(first)
        rows.append(
            {
                "potential": name,
                "n": int(n_points),
                "shifts": int(n_shifts),
                "unusable": int(bad),
                "first_bad_k_min": int(min(steps)) if steps else None,
                "first_bad_k_max": int(max(steps)) if steps else None,
            }
        )
    return {"rows": rows}


def _separable_2d(name: str, n: int):
    from exp6_certificate_training_2d import _problem2d

    domain, potential, energy, _ = _problem2d(name)
    x = np.linspace(*domain, n)
    y = np.linspace(*domain, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = Grid2D(x=x, y=y, potential=potential(X, Y))
    return grid, energy


def report_2d(n: int = 81, n_shifts: int = 25) -> dict:
    rows = []
    for name in ("harmonic2d", "well2d", "anharmonic2d"):
        grid, energy = _separable_2d(name, n)
        if energy is None:
            # 非谐振子没有解析值，扫描范围取基态附近
            center, span = 1.2, 0.4
        else:
            center, span = float(energy), 1.5 * float(energy)
        agree = refused = wrong = 0
        times = []
        for sigma in np.linspace(center - 0.5 * span, center + 0.5 * span, n_shifts):
            start = time.perf_counter()
            certificate = certified_separable_count_2d(grid, float(sigma))
            times.append(time.perf_counter() - start)
            reference = count_eigenvalues_below_2d(grid, float(sigma))
            if not certificate.certified:
                refused += 1
                continue
            if certificate.count == reference:
                agree += 1
            else:
                wrong += 1
        rows.append(
            {
                "potential": name,
                "n": int(n),
                "shifts": int(n_shifts),
                "agree": int(agree),
                "refused": int(refused),
                "wrong": int(wrong),
                "median_ms": float(np.median(times) * 1e3),
                "max_ms": float(np.max(times) * 1e3),
                "certified": bool(refused == 0 or agree > 0),
            }
        )
    return {"rows": rows}


def _dense_interval_ldl(grid: Grid2D, shift: float) -> dict:
    """无主元稠密区间 LDLᵀ：记录第一个跨零的主元。

    这是“一般二维”最直接的区间化路线。与可分离路线不同，它必须在区间
    上做除法并把消去填满整个右下块，区间宽度在每一步都被放大，主元很快
    跨零。这里记录它跑到第几步就失败。
    """
    dense = _sparse_operator(grid, shift).toarray()
    lo = dense.copy()
    hi = dense.copy()
    n = dense.shape[0]
    widths = []
    for k in range(n):
        d_lo, d_hi = float(lo[k, k]), float(hi[k, k])
        if not (d_hi < 0.0 or d_lo > 0.0):
            return {
                "refused_at": k,
                "max_relative_width": max(widths) if widths else 0.0,
                # 拒绝前一步的宽度：论文里引用的就是它（“拒绝时宽度已达 …”）
                "width_at_refusal": widths[-1] if widths else 0.0,
            }
        widths.append((d_hi - d_lo) / max(abs(d_lo), abs(d_hi), 1e-300))
        if k == n - 1:
            break
        col_lo = lo[k + 1:, k].copy()
        col_hi = hi[k + 1:, k].copy()
        quotients = (
            col_lo / d_lo,
            col_lo / d_hi,
            col_hi / d_lo,
            col_hi / d_hi,
        )
        l_lo = np.nextafter(np.minimum.reduce(quotients), -np.inf)
        l_hi = np.nextafter(np.maximum.reduce(quotients), np.inf)
        prod_lo, prod_hi = _interval_product(
            l_lo[:, None], l_hi[:, None], col_lo[None, :], col_hi[None, :]
        )
        lo[k + 1:, k + 1:] -= prod_hi
        hi[k + 1:, k + 1:] -= prod_lo
    return {
        "refused_at": None,
        "max_relative_width": max(widths) if widths else 0.0,
        "width_at_refusal": widths[-1] if widths else 0.0,
    }


def report_general_2d(n: int = 21, n_shifts: int = 10) -> dict:
    grid, _ = _separable_2d("anharmonic2d", n)
    lo = float(np.min(grid.potential))
    refused = 0
    steps = []
    widths = []
    refusal_widths = []
    for sigma in np.linspace(lo + 1.0, lo + 5.0, n_shifts):
        result = _dense_interval_ldl(grid, float(sigma))
        widths.append(result["max_relative_width"])
        if result["refused_at"] is not None:
            refused += 1
            steps.append(result["refused_at"])
            refusal_widths.append((result["refused_at"], result["width_at_refusal"]))
    first = min(refusal_widths) if refusal_widths else None
    return {
        "n": int(n),
        "size": int(grid.n_interior),
        "shifts": int(n_shifts),
        "refused": int(refused),
        "first_refusal_step": int(first[0]) if first else None,
        "median_refusal_step": float(np.median(steps)) if steps else None,
        "first_refusal_width": float(first[1]) if first else None,
        "max_relative_width": float(max(widths)) if widths else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/directed_count")
    args = parser.parse_args()

    payload = {
        "one_dimensional": report_1d(),
        "separable_two_dimensional": report_2d(),
        "determinant_recurrence": report_determinant(),
        "general_two_dimensional": report_general_2d(),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = []
    for row in payload["one_dimensional"]["rows"]:
        lines.append(
            f"1D {row['potential']:14s} agree={row['certified_agree']}/{row['shifts']} "
            f"refused={row['refused']} wrong={row['wrong']} "
            f"median={row['median_ms']:.1f}ms"
        )
    for row in payload["separable_two_dimensional"]["rows"]:
        lines.append(
            f"2D {row['potential']:14s} agree={row['agree']}/{row['shifts']} "
            f"refused={row['refused']} wrong={row['wrong']} "
            f"median={row['median_ms']:.0f}ms"
        )
    for row in payload["determinant_recurrence"]["rows"]:
        lines.append(
            f"determinant recurrence {row['potential']:14s} unusable on "
            f"{row['unusable']}/{row['shifts']} shifts, first bad k="
            f"{row['first_bad_k_min']}..{row['first_bad_k_max']}"
        )
    general = payload["general_two_dimensional"]
    lines.append(
        f"2D general (n={general['n']}, {general['size']} unknowns): "
        f"refused {general['refused']}/{general['shifts']} shifts, "
        f"first refusal at step {general['first_refusal_step']}, "
        f"max relative pivot width {general['max_relative_width']:.1e}"
    )
    (out / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.name}/stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
