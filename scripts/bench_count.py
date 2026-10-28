"""基准：一次“计数”的耗时（1D Sturm 主元序列 / 2D 稀疏 LU 惯性）。

论文里报告了“一次计数 0.03--0.13 s”这样的数字，它必须是可复现的，
所以由本脚本测量并写入 ``runs/_bench/count_timing.json``，正文引用
该文件而不是凭印象给数。

测量对象：

* 一维：``count_eigenvalues_below``（O(N) 三对角 LDLᵀ 主元符号），
  N=2001（与论文一维实验网格一致）；
* 二维：``count_eigenvalues_below_2d``（``splu`` 无主元交换的 U 对角），
  81x81（与论文二维实验网格一致）。

每个 shift 重复 ``--repeat`` 次取中位数，并记录最小/最大值。shift 取
若干互不相同的值，避免同一个符号模式被缓存/分支预测占便宜。

另测三项正文引用的数字：一次**完整证书**（``certify_eigenpair`` /
``certify_eigenpair_2d``，含区间提议与验证）的耗时，证书引导滤波
扫描（截断阶数由证书自己选）的耗时，以及同一矩阵上一次 shift-invert
Lanczos 求解（k=1）的耗时——§7.3 用它回答“为什么不直接解 H_h”。

用法：
    python scripts/bench_count.py
输出 runs/_bench/count_timing.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify import (  # noqa: E402
    _dirichlet_data,
    certify_eigenpair,
    count_eigenvalues_below,
)
from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    _sparse_operator,
    certify_eigenpair_2d,
    count_eigenvalues_below_2d,
)
from nnschrodinger.solver.filtering import guided_filter  # noqa: E402
from nnschrodinger.solver.filtering2d import guided_filter_2d  # noqa: E402


def _harmonic_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-5.0, 5.0, n)
    return x, 0.5 * x**2


def _harmonic_2d(n: int) -> Grid2D:
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return Grid2D(x, x.copy(), 0.5 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2))


def _timings(fn, shifts: list[float], repeat: int) -> dict:
    fn(shifts[0])  # 预热：把 scipy 首次导入/缓存的开销排除在外
    samples: list[float] = []
    for shift in shifts:
        for _ in range(repeat):
            start = time.perf_counter()
            fn(shift)
            samples.append(time.perf_counter() - start)
    return {
        "n_shifts": len(shifts),
        "repeat": repeat,
        "median_s": float(np.median(samples)),
        "min_s": float(np.min(samples)),
        "max_s": float(np.max(samples)),
    }


def _ground_state_1d(
    x: np.ndarray, potential: np.ndarray
) -> np.ndarray:
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    diag, off = _dirichlet_data(x, potential)
    matrix = diags([off * np.ones(diag.size - 1), diag,
                    off * np.ones(diag.size - 1)], [-1, 0, 1]).tocsc()
    sigma = float(np.min(potential)) - 1.0
    _, vectors = eigsh(matrix, k=1, sigma=sigma, which="LM", tol=1e-12)
    psi = np.zeros(x.size)
    psi[1:-1] = vectors[:, 0]
    return psi


def _ground_state_2d(grid: Grid2D) -> np.ndarray:
    from scipy.sparse.linalg import eigsh

    operator = _sparse_operator(grid, 0.0)
    sigma = float(np.min(grid.potential)) - 1.0
    _, vectors = eigsh(operator, k=1, sigma=sigma, which="LM", tol=1e-12)
    psi = np.zeros(grid.shape)
    psi[1:-1, 1:-1] = vectors[:, 0].reshape(grid.shape[0] - 2, grid.shape[1] - 2)
    return psi


def main() -> int:
    parser = argparse.ArgumentParser(description="一次计数的耗时基准")
    parser.add_argument("--n1d", type=int, default=2001)
    parser.add_argument("--n2d", type=int, default=81)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "_bench" / "count_timing.json")
    args = parser.parse_args()

    x, potential = _harmonic_1d(int(args.n1d))
    one_d = _timings(
        lambda s: count_eigenvalues_below(x, potential, s),
        [0.6, 1.6, 2.6, 3.6],
        int(args.repeat),
    )
    one_d["grid_points"] = int(args.n1d)

    grid = _harmonic_2d(int(args.n2d))
    two_d = _timings(
        lambda s: count_eigenvalues_below_2d(grid, s),
        [2.2, 2.6, 3.0, 3.4],
        int(args.repeat),
    )
    two_d["grid"] = f"{args.n2d}x{args.n2d}"

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "count_1d": one_d,
        "count_2d": two_d,
    }

    # 计数耗时随网格的变化（2D，shift 取在基态与第二态之间，计数 = 1）。
    scaling = []
    for n in (41, 61, 81, 101, 121, 161, 201):
        grid_n = _harmonic_2d(n)
        count_eigenvalues_below_2d(grid_n, 25.0)  # 预热
        start = time.perf_counter()
        count = count_eigenvalues_below_2d(grid_n, 25.0)
        scaling.append({"n": n, "count": count,
                        "seconds": time.perf_counter() - start})
    payload["scaling_2d"] = scaling

    # 一次完整的证书引导扫描（截断阶数由证书自己选），1D 与 2D 各一次。
    psi_1d = _ground_state_1d(x, potential)
    start = time.perf_counter()
    scan_1d = guided_filter(psi_1d, x, potential)
    payload["scan_1d"] = {
        "grid_points": int(args.n1d),
        "candidates": len(scan_1d.scan),
        "seconds": time.perf_counter() - start,
    }

    psi_2d = _ground_state_2d(grid)
    start = time.perf_counter()
    scan_2d = guided_filter_2d(psi_2d, grid)
    payload["scan_2d"] = {
        "grid": f"{args.n2d}x{args.n2d}",
        "candidates": len(scan_2d.scan),
        "seconds": time.perf_counter() - start,
    }
    # 一次**完整证书**（含区间提议与验证）的耗时：正文 Program summary 引用。
    def _median(fn, repeat: int) -> dict:
        fn()  # 预热
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn()
            samples.append(time.perf_counter() - start)
        return {
            "repeat": repeat,
            "median_s": float(np.median(samples)),
            "min_s": float(np.min(samples)),
            "max_s": float(np.max(samples)),
            "valid": bool(result.valid),
            "width": float(result.width),
        }

    payload["certificate_1d"] = _median(
        lambda: certify_eigenpair(psi_1d, x, potential, isolate=True),
        int(args.repeat),
    )
    payload["certificate_1d"]["grid_points"] = int(args.n1d)
    payload["certificate_2d"] = _median(
        lambda: certify_eigenpair_2d(psi_2d, grid, isolate=True),
        int(args.repeat),
    )
    payload["certificate_2d"]["grid"] = f"{args.n2d}x{args.n2d}"

    # 同一矩阵上一次 shift-invert Lanczos 求解（k=1）的耗时：§7.3 的速度比较。
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    def _time_s(fn, repeat: int) -> dict:
        fn()  # 预热
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return {
            "repeat": repeat,
            "median_s": float(np.median(samples)),
            "min_s": float(np.min(samples)),
            "max_s": float(np.max(samples)),
        }

    diag_1d, off_1d = _dirichlet_data(x, potential)
    matrix_1d = diags(
        [off_1d * np.ones(diag_1d.size - 1), diag_1d,
         off_1d * np.ones(diag_1d.size - 1)], [-1, 0, 1]).tocsc()
    sigma_1d = float(np.min(potential)) - 1.0
    payload["solve_1d"] = _time_s(
        lambda: eigsh(matrix_1d, k=1, sigma=sigma_1d, which="LM", tol=1e-12),
        int(args.repeat),
    )
    payload["solve_1d"]["grid_points"] = int(args.n1d)

    operator_2d = _sparse_operator(grid, 0.0)
    sigma_2d = float(np.min(grid.potential)) - 1.0
    payload["solve_2d"] = _time_s(
        lambda: eigsh(operator_2d, k=1, sigma=sigma_2d, which="LM", tol=1e-12),
        int(args.repeat),
    )
    payload["solve_2d"]["grid"] = f"{args.n2d}x{args.n2d}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"1D N={one_d['grid_points']}: median {one_d['median_s']*1e3:.1f} ms "
          f"({one_d['min_s']*1e3:.1f}--{one_d['max_s']*1e3:.1f})")
    print(f"2D {two_d['grid']}: median {two_d['median_s']*1e3:.1f} ms "
          f"({two_d['min_s']*1e3:.1f}--{two_d['max_s']*1e3:.1f})")
    print("scaling 2D: " + ", ".join(
        f"N={row['n']}: {row['seconds']:.3f} s" for row in scaling))
    print(f"scan 1D: {payload['scan_1d']['candidates']} candidates in "
          f"{payload['scan_1d']['seconds']:.2f} s")
    print(f"scan 2D: {payload['scan_2d']['candidates']} candidates in "
          f"{payload['scan_2d']['seconds']:.2f} s")
    print(f"certificate 1D: median {payload['certificate_1d']['median_s']*1e3:.1f} ms")
    print(f"certificate 2D: median {payload['certificate_2d']['median_s']*1e3:.1f} ms")
    print(f"solve 1D: median {payload['solve_1d']['median_s']*1e3:.1f} ms")
    print(f"solve 2D: median {payload['solve_2d']['median_s']*1e3:.1f} ms")
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
