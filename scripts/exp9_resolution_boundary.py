"""实验九：单态证书的“分辨率边界”——简并何时变成可认证的近简并。

实验七/八 说明：在**精确简并**的能级上单态证书必须拒绝、簇证书可以认证。
本实验把这条界线量化：在方阱里加一个各向异性微扰

    V_λ(x, y) = λ (x² − y²),

它把 (1,2)/(2,1) 这一对拆开，裂距随 λ 线性增长（网格上真实裂距由
ARPACK 读出的离散谱给出，不依赖微扰论）。对每个 λ 同时跑

* 单态证书（``certify_eigenpair_2d``，isolate=True）——预测：当验证出的
  谱隙 min(η−a, b−η) 低于计数分辨率下限 floor = 1e-9·scale 时拒绝，
  即“可认证”的门槛是 **裂距 > floor**，而不是“是否简并”。谱隙本身由
  区间构造决定：常规提议把端点向相邻本征值内缩 10%（谱隙 = 0.9·裂距），
  在贴近 floor 时退化为二分兜底（端点落在相邻本征值上，谱隙 ≈ 裂距），
  因此门槛在 裂距/floor ≈ 1 处，1.11 以上由常规提议接管；
* 簇证书（``certify_subspace_2d``，m=2）——预测：始终接受，因为它的
  边距 δ 由**到相邻能级**的距离决定（~7.4），与簇内裂距无关。

若两个预测都被数据证实，那么“简并只能拒绝”的说法可以被替换成更精确
的一句：**单态证书的障碍是计数分辨率，簇证书把这个障碍完全移除了。**

用法：
    python scripts/exp9_resolution_boundary.py --n-grid 81
输出 runs/resolution2d/boundary.{json,md}
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
    _resolution_floor,
    _sparse_operator,
    approximate_spectrum_2d,
    certify_eigenpair_2d,
)
from nnschrodinger.solver.certify_subspace import (  # noqa: E402
    certify_subspace_2d,
)

LAMBDAS = (
    0.0, 1e-5, 3e-5, 1e-4, 1.5e-4, 1.6e-4, 1.7e-4, 1.9e-4,
    2e-4, 3e-4, 4e-4, 1e-3, 1e-2,
)


def _grid(n: int, lam: float) -> Grid2D:
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return Grid2D(x, x.copy(), lam * (xx**2 - yy**2))


def _low_states(grid: Grid2D, k: int = 4):
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


def _cert_row(cert) -> dict:
    return {
        "valid": bool(cert.valid),
        "width": cert.width,
        "spectral_gap": cert.spectral_gap,
        "state_index": cert.state_index,
        "reason": cert.reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="单态证书的分辨率边界")
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "resolution2d")
    args = parser.parse_args()

    n = int(args.n_grid)
    rows: list[dict] = []

    for lam in LAMBDAS:
        grid = _grid(n, lam)
        approx = approximate_spectrum_2d(grid, 6)
        values, vectors = _low_states(grid, 4)
        pair = [_embed(vectors[:, 1], grid), _embed(vectors[:, 2], grid)]
        split = float(abs(values[1] - values[2]))
        floor = _resolution_floor(grid)

        singles = [
            _cert_row(certify_eigenpair_2d(psi, grid, isolate=True, approx=approx))
            for psi in pair
        ]
        cluster = certify_subspace_2d(pair, grid, approx=approx)
        rows.append({
            "lambda": lam,
            "energy_pair": [float(values[1]), float(values[2])],
            "split": split,
            "floor": floor,
            "single": singles,
            "single_accepted": all(s["valid"] for s in singles),
            "cluster": {
                "valid": bool(cluster.valid),
                "multiplicity": cluster.verified_multiplicity,
                "frobenius_residual": cluster.frobenius_residual,
                "margin": cluster.margin,
                "angle_bound": cluster.angle_bound,
                "reason": cluster.reason,
            },
        })
        print(
            f"lambda={lam:8.1e}  split={split:9.3e}  floor={floor:9.3e}  "
            f"split/floor={split / floor:6.2f}  "
            f"single={'ACCEPT' if rows[-1]['single_accepted'] else 'refuse'}  "
            f"cluster={'accept' if cluster.valid else 'refuse'}"
        )

    summary = {
        "n_grid": n,
        "prediction": "single-state accepts iff verified gap > floor; "
                      "gap = 0.9*split above split/floor ~ 1.11, gap ~ split "
                      "at the crossing (bisection fallback)",
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "boundary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 单态证书的分辨率边界（方阱 + λ(x²−y²)，{}×{}）".format(n, n),
        "",
        "| λ | 裂距 | floor | 裂距/floor | 单态(下) | 单态(上) | 簇证书 | δ | sinΘ 界 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["single"]
        lines.append(
            "| {lam:.0e} | {split:.2e} | {floor:.2e} | {ratio:.1f} | {a} | {b} | {c} | "
            "{d} | {e} |".format(
                lam=r["lambda"], split=r["split"], floor=r["floor"],
                ratio=r["split"] / r["floor"],
                a="✓" if lo["valid"] else "✗", b="✓" if hi["valid"] else "✗",
                c="✓" if r["cluster"]["valid"] else "✗",
                d=("—" if r["cluster"]["margin"] is None
                   else f"{r['cluster']['margin']:.4f}"),
                e=("—" if r["cluster"]["angle_bound"] is None
                   else f"{r['cluster']['angle_bound']:.2e}"),
            )
        )
    lines += [
        "",
        "预测：单态证书在验证出的谱隙 min(η−a, b−η) 低于 floor 时拒绝，",
        "即 **裂距 > floor** 才可认证。常规提议内缩 10%，谱隙 = 0.9·裂距，",
        "因此 裂距/floor ≳ 1.11 时谱隙/floor = 0.9·裂距/floor；贴近 floor 时",
        "退回二分兜底，端点落在相邻本征值上，谱隙 ≈ 裂距，门槛因此落在",
        "裂距/floor ≈ 1 处。簇证书的 δ 由到相邻能级的距离决定，与簇内裂距",
        "无关，因此在整个扫描上都能认证。",
        "",
    ]
    (args.out_dir / "boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
