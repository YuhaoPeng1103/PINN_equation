"""残差与真实误差的差距：从已保存的场重算，不重新训练。

附录 A 的 ``unnormalized`` 失败说明：损失对尺度不变时，网络可以靠**缩小
振幅**把残差做小，而场本身离本征函数仍然很远。论文因此需要两个落盘数字：
场按原样（未归一化）的残差范数，以及它到离散本征函数的 L2 距离。

本脚本读取 ``runs/appendix_failures/<variant>/wavefunction2d.npz``，
在同一个离散算子 H_h（Dirichlet 边界、五点差分）上计算

* ``eta``：ψ 的离散 Rayleigh 商（对尺度不变）；
* ``residual_norm_raw`` = ‖H_hψ − ηψ‖，ψ 取原样；
* ``distance_l2``：ψ/‖ψ‖ 到离散基态的 L2 距离，基态由 LAPACK 稠密分解
  独立算出（与被检验的场无关）；
* ``ratio`` = distance_l2 / residual_norm_raw。

输出与 npz 同目录的 ``residual_vs_error.json``。

用法：
    python scripts/residual_vs_error.py runs/appendix_failures/unnormalized
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
    discrete_rayleigh_quotient_2d,
    normalize_2d,
    residual_norm_2d,
)


def reference_ground_state(grid: Grid2D) -> np.ndarray:
    """用稠密 LAPACK 求离散基态（单位 Dirichlet 范数，边界为零）。"""
    import scipy.linalg as sla

    operator = _sparse_operator(grid, 0.0).toarray()
    values, vectors = sla.eigh(operator, subset_by_index=[0, 0])
    psi = np.zeros(grid.shape)
    psi[1:-1, 1:-1] = vectors[:, 0].reshape(grid.shape[0] - 2, grid.shape[1] - 2)
    unit, _ = normalize_2d(psi, grid)
    return unit, float(values[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="残差 vs 真实 L2 误差")
    parser.add_argument("run_dir", type=Path,
                        help="含 wavefunction2d.npz 的 run 目录")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.run_dir / "wavefunction2d.npz")
    x, y, raw = data["x"], data["y"], data["psi_raw"]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid = Grid2D(x, y, np.zeros_like(xx))

    eta = discrete_rayleigh_quotient_2d(raw, grid)
    unit, norm = normalize_2d(raw, grid)
    residual_unit = residual_norm_2d(raw, grid, eta)
    residual_raw = norm * residual_unit

    ref, lambda0 = reference_ground_state(grid)
    # 到单位本征函数的距离按**原样的场**算（这正是损失没有约束的那个对象）：
    # ‖ψ − φ‖² = ‖ψ‖² + 1 − 2⟨ψ, φ⟩，边界节点取零（Dirichlet 口径）。
    field = raw.copy()
    field[0, :] = field[-1, :] = 0.0
    field[:, 0] = field[:, -1] = 0.0
    inner = grid.cell * float(np.sum(field[1:-1, 1:-1] * ref[1:-1, 1:-1]))
    if inner < 0.0:  # 本征向量的整体符号任意
        inner = -inner
    distance = math.sqrt(max(0.0, norm**2 + 1.0 - 2.0 * inner))

    payload = {
        "run": str(args.run_dir.as_posix()),
        "norm_raw": norm,
        "eta": eta,
        "lambda0_dense": lambda0,
        "residual_norm_unit_field": residual_unit,
        "residual_norm_raw": residual_raw,
        "distance_l2_to_ground_state": distance,
        "ratio_distance_over_residual": distance / residual_raw,
    }
    out = args.out or (args.run_dir / "residual_vs_error.json")
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"eta={eta:.6f} lambda0={lambda0:.6f}")
    print(f"||psi||={norm:.3e} residual(raw)={residual_raw:.3e} "
          f"L2 distance={distance:.4f} ratio={distance / residual_raw:.0f}")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
