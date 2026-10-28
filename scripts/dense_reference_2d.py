"""独立的二维参考谱：稠密对称特征分解（LAPACK，MRRR）。

二维实验里，``filter_certificates2d.py`` 用 ARPACK 的近似谱同时**提议区间**
和充当“参考值”，这有循环论证的嫌疑：两者来自同一次求解。本脚本用完全
独立的方法给出参考——把 ``H_h``（81x81 网格，内部 6241 个未知量）显式稠密
化，用 LAPACK 的 ``eigh(..., subset_by_index=[0,2])`` 求前三个。H_h 只依赖
势能、不依赖网络，所以每个势只需一次分解，之后三个种子的证书一起比对。

输出 ``runs/reference2d/dense_ref.json``：每个势的稠密本征值、ARPACK 值、
两者之差，以及每条已记录的证书区间是否包含稠密值。

用法：
    python scripts/dense_reference_2d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    _sparse_operator,
    approximate_spectrum_2d,
)

POTENTIALS = ("harmonic2d", "well2d", "anharmonic2d")


def _load_grid(potential: str) -> Grid2D:
    path = ROOT / "runs" / "seeds2d" / potential / "seed0" / "wavefunction2d.npz"
    data = np.load(path)
    return Grid2D(data["x"], data["y"], data["potential"])


def main() -> int:
    from scipy.linalg import eigh

    summary = json.loads(
        (ROOT / "runs" / "certificates_filtered2d" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    by_potential: dict[str, list[dict]] = {}
    for row in summary:
        by_potential.setdefault(row["potential"], []).append(row)

    report = {"method": "scipy.linalg.eigh, subset_by_index=[0,2], driver=evr",
              "grid": "81x81", "problems": {}}

    for potential in POTENTIALS:
        grid = _load_grid(potential)
        matrix = _sparse_operator(grid, 0.0).toarray()
        values, vectors = eigh(matrix, subset_by_index=[0, 2], driver="evr")
        # LAPACK 本征值的绝对误差是 eps*||H|| ~ 6e-12，比要验证的宽度还大；
        # 本征向量的角度误差是 eps*||H||/gap，Rayleigh 商的误差是这个量的
        # 平方再乘 gap，实测达到 1e-15。因此参考值取 Rayleigh 商。
        dense = np.array([
            float(v @ (matrix @ v) / (v @ v)) for v in vectors.T
        ])
        del matrix, vectors
        approx = approximate_spectrum_2d(grid, 6)

        rows = []
        for row in sorted(by_potential[potential], key=lambda r: r["path"]):
            cert = row["filtered_certificate"]
            index = cert["state_index"]
            dense_value = float(dense[index]) if index is not None else None
            arpack_value = float(approx[index]) if index is not None else None
            lo, hi = cert["temple_lower"], cert["temple_upper"]
            rows.append({
                "path": row["path"],
                "state_index": index,
                "dense": dense_value,
                "arpack": arpack_value,
                "arpack_minus_dense": (
                    None if dense_value is None or arpack_value is None
                    else arpack_value - dense_value
                ),
                "temple_lower": lo,
                "temple_upper": hi,
                "contains_dense": (
                    None if None in (lo, hi, dense_value)
                    else bool(lo <= dense_value <= hi)
                ),
            })
        report["problems"][potential] = {
            "dense_eigenvalues": [float(v) for v in values],
            "dense_rayleigh": [float(v) for v in dense],
            "arpack": [float(v) for v in approx[:3]],
            "runs": rows,
        }
        print(potential, "dense", np.array2string(dense, precision=10),
              "arpack", np.array2string(approx[:3], precision=10))

    out = ROOT / "runs" / "reference2d" / "dense_ref.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    total = sum(len(v["runs"]) for v in report["problems"].values())
    inside = sum(
        1 for v in report["problems"].values() for r in v["runs"] if r["contains_dense"]
    )
    print(f"containment against dense reference: {inside}/{total}")
    print(f"wrote {out.name}")
    return 0 if inside == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
