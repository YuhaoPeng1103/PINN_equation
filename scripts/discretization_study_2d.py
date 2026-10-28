"""二维离散化误差研究：证书认证的是 H_h，不是连续算子。

对每个 N，用**惯性计数二分**（不是 ARPACK）求 H_h 的基态能量 λ₀(h)，
与解析值比较，并拟合误差的 h 阶数。这是论文里必须交代的“证书边界”：
二维原型在 N=81 时的离散偏差（~1e-3）比网络误差（~1e-5）大两个数量级，
所以报告精度时必须分开写。

用法：
    python scripts/discretization_study_2d.py --sizes 41 61 81 121
输出 runs/discretization2d/summary.{json,md}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    approximate_spectrum_2d,
    discrete_eigenvalue_2d,
)

PROBLEMS = {
    "harmonic2d": {
        "domain": (-3.0, 3.0),
        "potential": lambda x, y: 0.5 * (x**2 + y**2),
        "energy": 1.0,
    },
    "well2d": {
        "domain": (0.0, 1.0),
        "potential": lambda x, y: np.zeros_like(x + y),
        "energy": math.pi**2,
    },
}


def _fit_order(errors: list[float], steps: list[float]) -> float | None:
    """log|error| = p·log h + c 的最小二乘斜率。"""
    e = np.asarray(errors, dtype=float)
    h = np.asarray(steps, dtype=float)
    mask = e > 0
    if mask.sum() < 2:
        return None
    p, _ = np.polyfit(np.log(h[mask]), np.log(e[mask]), 1)
    return float(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="二维离散化误差研究")
    parser.add_argument("--sizes", nargs="+", type=int, default=[41, 61, 81, 121])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "discretization2d")
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}
    for name, spec in PROBLEMS.items():
        lo, hi = spec["domain"]
        rows = []
        for n in args.sizes:
            x = np.linspace(lo, hi, n)
            xx, yy = np.meshgrid(x, x, indexing="ij")
            grid = Grid2D(x, x.copy(), np.asarray(spec["potential"](xx, yy), dtype=float))
            approx = approximate_spectrum_2d(grid, 3)
            lam = discrete_eigenvalue_2d(grid, 0, approx=approx)
            rows.append({
                "n": int(n),
                "h": float(grid.hx),
                "lambda0": float(lam),
                "analytic": float(spec["energy"]),
                "error": float(abs(lam - spec["energy"])),
                "arpack_lambda0": float(approx[0]),
            })
            print(f"{name} N={n} h={grid.hx:.4f} λ0={lam:.10f} "
                  f"err={abs(lam - spec['energy']):.3e}")
        rows.sort(key=lambda r: r["n"])
        # λ(h) = λ* + c·h² 的最小二乘拟合：把“纯离散化误差”与“区域截断
        # 带来的常数偏移”分开。谐振子的解析值 1.0 是**无穷域**的结果，
        # 有限盒 [-3,3]² 上的 Dirichlet 问题本身就有 ~6e-4 的偏移。
        h = np.asarray([r["h"] for r in rows])
        lam = np.asarray([r["lambda0"] for r in rows])
        design = np.stack([np.ones_like(h), h**2], axis=1)
        coeff, *_ = np.linalg.lstsq(design, lam, rcond=None)
        lambda_star = float(coeff[0])
        for r in rows:
            r["lambda_star"] = lambda_star
            r["error_vs_lambda_star"] = float(abs(r["lambda0"] - lambda_star))
        results[name] = rows

    orders = {
        name: _fit_order([r["error_vs_lambda_star"] for r in rows], [r["h"] for r in rows])
        for name, rows in results.items()
    }
    stars = {name: rows[0]["lambda_star"] for name, rows in results.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"problems": results, "fit_order": orders, "lambda_star": stars},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# 二维离散化误差（惯性计数二分，非 ARPACK）", ""]
    for name, rows in results.items():
        lines += [f"## {name}（纯离散误差拟合阶数 p ≈ {orders[name]:.2f}）" if orders[name] else f"## {name}",
                  "", f"λ(h) = λ* + c·h² 拟合得 λ* = {stars[name]:.10f}（解析值 {rows[0]['analytic']:.10f}）。",
                  "", "| N | h | λ₀(h) | 误差 vs 解析 | 误差 vs λ* |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['n']} | {r['h']:.4f} | {r['lambda0']:.10f} | "
                f"{r['error']:.3e} | {r['error_vs_lambda_star']:.3e} |"
            )
        lines.append("")
    text = "\n".join(lines)
    (args.out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
