"""评测：把“神经网络解”与参考解逐项比较（纯 numpy）。

所有对外结论都由本模块的指标给出，定义如下：

| 指标 | 定义 | 含义 |
|---|---|---|
| energy_abs_err | |E_model − E_ref| | 能量绝对误差（Hartree） |
| energy_rel_err | 上式 / |E_ref| | 能量相对误差 |
| overlap | |⟨ψ_model|ψ_ref⟩| / (‖ψ_model‖‖ψ_ref‖) | 波形相似度，1 为完全一致 |
| inf_norm | max|ψ_model − ψ_ref|（均已 L2 归一） | 最坏点偏差 |
| residual_rms | 网格上 ‖Ĥψ − Eψ‖ 的 RMS | 是否真的满足方程 |

注意：overlap 与能量反映同一件事的两个侧面。PINN 的典型失败模式是“能量很准、
波形却是杂波”（与真解正交的污染分量会把 Rayleigh 商拉向 E，却不改变归一化），
因此必须同时报告 overlap 与 inf_norm。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class Metrics:
    energy_model: float
    energy_ref: float
    energy_abs_err: float
    energy_rel_err: float
    overlap: float
    inf_norm: float
    residual_rms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def residual_rms(psi: np.ndarray, x: np.ndarray, energy: float, potential: np.ndarray) -> float:
    """Ĥψ − Eψ 在网格上的 RMS（ψ 先按 L2 归一）。"""
    psi = np.asarray(psi, dtype=float)
    h = float(x[1] - x[0])
    norm = float(np.sqrt(np.sum(psi**2) * h))
    psi = psi / norm if norm > 0 else psi
    d2 = np.zeros_like(psi)
    d2[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h**2
    d2[0], d2[-1] = d2[1], d2[-2]
    r = -0.5 * d2 + (np.asarray(potential, dtype=float) - energy) * psi
    return float(np.sqrt(np.mean(r**2)))


def evaluate_model(
    psi_model: np.ndarray,
    x: np.ndarray,
    reference: Callable[[np.ndarray], np.ndarray],
    energy_ref: float,
    potential: Callable[[np.ndarray], np.ndarray],
    energy_model: float | None = None,
) -> Metrics:
    """比较模型波函数与参考解。energy_model 为 None 时用 Rayleigh 商估计能量。"""
    x = np.asarray(x, dtype=float)
    h = float(x[1] - x[0])
    psi = np.asarray(psi_model, dtype=float)
    norm = float(np.sqrt(np.sum(psi**2) * h))
    psi_unit = psi / norm if norm > 0 else psi
    ref = np.asarray(reference(x), dtype=float)
    ref_unit = ref / np.sqrt(np.sum(ref**2) * h)
    v = np.asarray(potential(x), dtype=float)

    if energy_model is None:
        d = np.gradient(psi_unit, x)
        energy_model = 0.5 * float(np.trapezoid(d**2, x)) + float(np.trapezoid(v * psi_unit**2, x))

    inner = float(np.trapezoid(psi_unit * ref_unit, x))
    overlap = abs(inner)
    sign = 1.0 if inner >= 0 else -1.0
    inf_norm = float(np.max(np.abs(sign * psi_unit - ref_unit)))
    res = residual_rms(psi_unit, x, float(energy_model), v)
    abs_err = abs(float(energy_model) - float(energy_ref))
    return Metrics(
        energy_model=float(energy_model),
        energy_ref=float(energy_ref),
        energy_abs_err=abs_err,
        energy_rel_err=abs_err / abs(float(energy_ref)) if energy_ref != 0 else float("inf"),
        overlap=overlap,
        inf_norm=inf_norm,
        residual_rms=res,
    )


def format_table(rows: list[dict]) -> str:
    """把若干指标 dict 排版成 Markdown 表格。"""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = []
        for c in cols:
            val = row[c]
            cells.append(f"{val:.4e}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = ["Metrics", "evaluate_model", "residual_rms", "format_table"]
