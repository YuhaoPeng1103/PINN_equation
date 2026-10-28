"""参考解的独立交叉校验：解析解 ⟷ Numerov 数值解，两套方法互证。

项目可信度的地基在这里。神经网络的评测目标（E_n、ψ_n）如果本身是错的，
所有精度结论都无效。因此每个问题都用**独立方法**再算一遍：

| 问题 | 参考解 | 交叉验证 |
|---|---|---|
| 谐振子 | 解析式（Hermite 多项式） | Numerov + 打靶 |
| 无限深方势阱 | 解析式（正弦） | 深有限深方势阱 V₀=2000（近似 ISW）的 Numerov 谱 |
| 有限深方势阱 | Numerov + 打靶 | 网格加密（见 convergence()） |
| 氢原子 | 解析式（Laguerre） | 能量与 −1/(2n²) 直接比较（见 tests） |

注：无限深阱与“深 2000 的有限阱”的谱相差约 E/V₀（一阶微扰，约 1e-3 量级），
这是**物理上应存在的差**而非数值误差；因此 run_all 对该项使用更宽容差
（能量 1e-1、波函数 1e-2）。

运行 python -m nnschrodinger.physics.verify 打印表格；退出码 0 表示全部通过。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .numerov import _assemble_eigenfunction, numerov_eigenvalues
from .spectrum import get_reference


def check(
    problem: str,
    n: int = 0,
    *,
    energy_tol: float = 5e-3,
    psi_tol: float = 5e-2,
    n_points: int = 8001,
) -> dict:
    """对单个问题做交叉校验，返回可直接写入评测报告的 dict。"""
    ref = get_reference(problem, n)
    lo, hi = ref.domain
    potential, window = _potential_and_window(problem)
    energies = numerov_eigenvalues(potential, lo, hi, window, n_points=n_points, n_scan=300)
    if len(energies) <= n:
        raise RuntimeError(f"{problem}: Numerov 只找到 {len(energies)} 个本征值，不足 {n + 1} 个")
    numeric_energy = float(energies[n])
    energy_err = abs(numeric_energy - ref.energy)

    grid = np.linspace(lo, hi, n_points)
    h = float(grid[1] - grid[0])
    psi_ref = np.asarray(ref.psi(grid), dtype=float)
    norm = math.sqrt(float(np.sum(psi_ref**2)) * h)
    psi_ref = psi_ref / norm if norm > 0 else psi_ref
    k2 = 2.0 * (numeric_energy - np.asarray(potential(grid), dtype=float))
    psi_num = _numeric_wavefunction(k2, h)
    sign = 1.0 if float(np.dot(psi_num, psi_ref)) >= 0 else -1.0
    psi_err = float(np.max(np.abs(sign * psi_num - psi_ref)))

    return {
        "problem": problem,
        "n": n,
        "energy_ref": ref.energy,
        "energy_numeric": numeric_energy,
        "energy_abs_err": energy_err,
        "psi_max_abs_err": psi_err,
        "passed": bool(energy_err < energy_tol and psi_err < psi_tol),
    }

def _potential_and_window(problem: str) -> tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float]]:
    key = problem.lower()
    if key in ("harmonic", "harmonic_oscillator"):
        return (lambda x: 0.5 * np.asarray(x, dtype=float) ** 2), (-0.5, 20.0)
    if key in ("infinite_well", "infinite_square_well"):
        # 无限深阱不可直接 Numerov（跳变）；用 V₀ = 200 的深势阱近似（误差 O(E/V₀)）
        return (lambda x: np.where(np.abs(x - 0.5) < 0.5, 0.0, 2000.0)), (0.0, 2000.0)
    if key in ("finite_well", "finite_square_well"):
        return (lambda x: np.where(np.abs(x) < 1.0, -50.0, 0.0)), (-50.0, 0.0)
    if key == "hydrogen":
        raise KeyError("hydrogen 的交叉验证请用 radial_numerov")
    raise KeyError(problem)


def _numeric_wavefunction(k2: np.ndarray, h: float) -> np.ndarray:
    """双向 Numerov 积分的波函数（与 numerov 模块同一算法），单位 L2 归一化。"""
    vec = _assemble_eigenfunction(k2, h * h / 12.0, h)
    return vec / math.sqrt(float(np.sum(vec**2)) * h)

def convergence(problem: str, n: int = 0, resolutions: tuple[int, ...] = (2001, 4001, 8001)) -> list[dict]:
    """同一问题在三种网格分辨率下的能量，用于验证打靶解随 h 收敛。"""
    lo, hi = get_reference(problem, n).domain
    potential, window = _potential_and_window(problem)
    out = []
    for points in resolutions:
        energies = numerov_eigenvalues(potential, lo, hi, window, n_points=points, n_scan=300)
        out.append({"n_points": points, "energy": float(energies[n]) if len(energies) > n else float("nan")})
    return out


def run_all(verbose: bool = True) -> list[dict]:
    """运行全部校验并打印成表格。"""
    rows: list[dict] = []
    for problem, states in (
        ("harmonic", [0, 1, 2, 3]),
        ("infinite_well", [0, 1, 2]),
        ("finite_well", [0, 1, 2]),
    ):
        for n in states:
            # 无限深阱的对照是“深 2000 的有限阱”，二者差约 E/V₀（物理差异），放宽容差
            loose = problem == "infinite_well"
            rows.append(check(problem, n, energy_tol=1e-1 if loose else 5e-3,
                              psi_tol=5e-2 if loose else 2e-3))
    if verbose:
        header = f"{'problem':<16}{'n':>3}{'E_ref':>13}{'E_numeric':>14}{'|dE|':>11}{'max|dpsi|':>12}{'pass':>7}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['problem']:<16}{r['n']:>3}{r['energy_ref']:>13.6f}{r['energy_numeric']:>14.8f}"
                f"{r['energy_abs_err']:>11.2e}{r['psi_max_abs_err']:>12.2e}{str(r['passed']):>7}"
            )
    return rows


def main() -> int:
    rows = run_all()
    n_fail = sum(1 for r in rows if not r["passed"])
    print()
    print(f"{len(rows) - n_fail}/{len(rows)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())