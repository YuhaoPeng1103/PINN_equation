"""经典数值基线：在与实验相同的网格上跑 Numerov，输出与网络可比的指标。

HANDOFF 的 P0 要求"与经典数值方法的同表对比"。参考解本身对谐振子用的是解析式，
所以 Numerov 在这里是**独立第二方法**，不是自证。指标与 solver.evaluate 完全一致：

    energy_abs_err / overlap / inf_norm / residual_rms

这样 PINN 与 Numerov 的数字才可以直接并排。

注意：
  * 无限深阱对 Numerov 是势跳变，不能直接积分；沿用 verify.py 的做法，
    用 V₀ = 2000 的深阱近似，误差是物理差异 O(E/V₀)，表里会注明。
  * 有限深阱的"参考解"本身就是 Numerov，不能拿它当独立基线，故不列入。

用法：python scripts/classical_baseline.py
输出：runs/classical_baseline.json + 终端 Markdown 表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.physics.numerov import numerov_spectrum  # noqa: E402
from nnschrodinger.physics.spectrum import harmonic_oscillator, infinite_square_well  # noqa: E402
from nnschrodinger.solver.evaluate import evaluate_model  # noqa: E402

N_POINTS = 2001


def harmonic_baseline(n_states: int = 4, domain: tuple[float, float] = (-6.0, 6.0)) -> list[dict]:
    potential = lambda x: 0.5 * np.asarray(x, dtype=float) ** 2  # noqa: E731
    num = numerov_spectrum(potential, domain[0], domain[1], (-0.5, 10.0), n_states, n_points=N_POINTS)
    grid = np.linspace(*domain, N_POINTS)
    rows = []
    for n in range(n_states):
        ref = harmonic_oscillator(n, domain=domain)
        psi = num.psi(grid, n)
        metrics = evaluate_model(psi, grid, ref.psi, ref.energy, potential,
                                 energy_model=float(num.energies[n])).to_dict()
        rows.append({"problem": "harmonic", "n": n, "method": "numerov", **metrics})
    return rows


def infinite_well_baseline(n_states: int = 1, L: float = 1.0) -> list[dict]:
    # 与 verify.py 相同的深阱近似：内部 V=0，端点 V=2000（不可直接用硬壁）
    v0 = 2000.0
    potential = lambda x: np.where((np.asarray(x) > 0.0) & (np.asarray(x) < L), 0.0, v0)  # noqa: E731
    num = numerov_spectrum(potential, 0.0, L, (0.0, 30.0), n_states, n_points=N_POINTS)
    grid = np.linspace(0.0, L, N_POINTS)
    rows = []
    for n in range(n_states):
        ref = infinite_square_well(n + 1, L)  # 1 基
        psi = num.psi(grid, n)
        metrics = evaluate_model(psi, grid, ref.psi, ref.energy, potential,
                                 energy_model=float(num.energies[n])).to_dict()
        rows.append({"problem": "infinite_well", "n": n, "method": "numerov", **metrics,
                     "note": "V0=2000 深阱近似；误差含 O(E/V0) 物理偏差"})
    return rows


def main() -> int:
    rows = harmonic_baseline() + infinite_well_baseline()
    out = ROOT / "runs" / "classical_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("| 问题 | n | 方法 | 能量 | |ΔE| | overlap | inf_norm |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['problem']} | {r['n']} | Numerov | {r['energy_model']:.6f} | "
              f"{r['energy_abs_err']:.3e} | {r['overlap']:.9f} | {r['inf_norm']:.3e} |")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
