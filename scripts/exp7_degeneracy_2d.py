"""实验七：二维简并态的证书拒绝行为。

二维各向同性势的第一激发态通常是**结构性简并**的：
方阱 (1,2)/(2,1)、各向同性谐振子 (1,0)/(0,1)。
这时：

* 残差证书（Weinstein）仍然给出一个极窄的区间——残差确实很小；
* Temple/Davis–Kahan 需要“区间内恰有一个本征值”，而简并意味着**不存在**
  这样的区间，证书必须返回 ``valid=False``。

本脚本用**精确的离散本征向量**（ARPACK）演示这一点，排除“训练不好”
这一解释：即使 ψ 的残差是 1e-13，证书也拒绝认证简并态。同时给出一个
随机混合态 a·ψ₁ + b·ψ₂（同样是本征态）的结果。

用法：
    python scripts/exp7_degeneracy_2d.py --n-grid 81
输出 runs/degeneracy2d/summary.{json,md}
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
    _sparse_operator,
    approximate_spectrum_2d,
    certify_eigenpair_2d,
)


def _discrete_vectors(grid: Grid2D, k: int):
    from scipy.sparse.linalg import eigsh

    operator = _sparse_operator(grid, 0.0)
    sigma = float(np.min(grid.potential)) - 1.0
    values, vectors = eigsh(operator, k=k, sigma=sigma, which="LM", tol=1e-12)
    order = np.argsort(values)
    return np.asarray(values)[order], np.asarray(vectors)[:, order]


def _embed(vec: np.ndarray, grid: Grid2D) -> np.ndarray:
    psi = np.zeros(grid.shape)
    psi[1:-1, 1:-1] = vec.reshape(grid.shape[0] - 2, grid.shape[1] - 2)
    return psi


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="二维简并态的证书拒绝")
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "degeneracy2d")
    args = parser.parse_args()

    n = int(args.n_grid)
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))
    values, vectors = _discrete_vectors(grid, 4)
    approx = approximate_spectrum_2d(grid, 5)
    split = abs(float(values[1] - values[2]))

    rng = np.random.default_rng(0)
    theta = 0.43
    mix = math.cos(theta) * vectors[:, 1] + math.sin(theta) * vectors[:, 2]
    mix /= np.linalg.norm(mix)

    cases = {
        "ground": _embed(vectors[:, 0], grid),
        "excited_a": _embed(vectors[:, 1], grid),
        "excited_b": _embed(vectors[:, 2], grid),
        "excited_mix": _embed(mix, grid),
    }
    rows = []
    for name, psi in cases.items():
        cert = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
        rows.append({
            "case": name,
            "energy": cert.energy,
            "residual_norm": cert.residual_norm,
            "weinstein_width": cert.weinstein_upper - cert.weinstein_lower,
            "valid": cert.valid,
            "state_index": cert.state_index,
            "temple_width": cert.width,
            "spectral_gap": cert.spectral_gap,
            "reason": cert.reason,
        })

    summary = {
        "n_grid": n,
        "degenerate_pair": [float(values[1]), float(values[2])],
        "degenerate_split": split,
        "ground_reference": float(values[0]),
        "cases": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 二维简并态的证书行为（方阱，{}×{}）".format(n, n),
        "",
        f"离散谱：E₀={values[0]:.9f}，E₁={values[1]:.9f}，E₂={values[2]:.9f}，"
        f"|E₁−E₂|={split:.3e}。",
        "",
        "| 态 | E | 残差 r | Weinstein 宽 2r | 有效 | 证书态 | Temple 宽 | 谱隙 | 说明 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {E} | {r} | {wv} | {ok} | {state} | {wt} | {gap} | {reason} |".format(
                case=row["case"], E=_fmt(row["energy"], 9), r=_fmt(row["residual_norm"]),
                wv=_fmt(row["weinstein_width"]), ok="✓" if row["valid"] else "✗",
                state=_fmt(row["state_index"]), wt=_fmt(row["temple_width"]),
                gap=_fmt(row["spectral_gap"]), reason=row["reason"][:70],
            )
        )
    lines += [
        "",
        "> 简并态的残差可以低到 1e-13：**残差小 ≠ 可认证**。Weinstein 界仍然",
        "> 成立（“存在某个本征值在区间内”），但它不能回答“是哪个态”，也因此",
        "> 无法给出本征函数夹角。Temple 需要单态区间，简并时该区间不存在，",
        "> 证书返回 valid=False——这是正确行为，不是数值失败。",
        "",
    ]
    md = "\n".join(lines)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
