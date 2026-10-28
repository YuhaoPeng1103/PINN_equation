"""目标函数（label）与相似度度量的约定。

本项目同时服务两类方法，它们对"label"的定义不同：

1. **PINN（无监督）**：没有 label，只有"物理残差 + 边界 + 归一化 + 正交"约束。
   对应的输入就是**配点**（collocation points），见 ``solver.inputs``。

2. **HHL 回归（有监督）**：先给定一个目标波函数 ψ_target(x)（或 ψ_target(x, E)），
   网络用 ``|ψ_model − ψ_target|`` 训练。HHL 论文的 10 个目标函数清单见
   ``HHL_FUNCTIONS``，它们都是**连续、非线性、在 [-H, H] 内取值的函数**（H≈100）。

相似度度量（论文中评估"网络是否等价于求值了特征向量"的核心指标）：
    sim(ψ, φ) = |⟨ψ|φ⟩| / (||ψ||·||φ||)，where ⟨ψ|φ⟩ = ∫ ψφ dx
值为 1 表示两条曲线完全一致（差一个无关紧要的符号）。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# HHL 论文的训练区间与点数
HHL_RANGE: tuple[float, float] = (-5.0, 5.0)
HHL_GRID_POINTS: int = 2500
HHL_H: float = 100.0

#: 论文 Table 1 风格的目标函数清单（名称 -> 解析式）。
#: 每个目标都必须在 [-H, H] 内且满足 f(0)=0 或 f 逐渐偏离恒等映射的要求。
HHL_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "quadratic": lambda x: 0.7 * x**2,
    "cubic": lambda x: 0.2 * x**3,
    "tanh": lambda x: 3.0 * np.tanh(x),
    "gaussian": lambda x: 2.0 * np.exp(-(x**2) / 3.0) - 0.3,
    "quartic": lambda x: 0.08 * x**4,
    "sin": lambda x: 1.0 * np.sin(1.2 * x),
    "scaled_sin": lambda x: 3.5 * np.sin(1.3 * x),
    "sigmoid": lambda x: 2.0 / (1.0 + np.exp(-x)) - 1.0,
    "sinc": lambda x: np.sinc(x / np.pi),
    "double_well": lambda x: 0.05 * (x**2 - 4.0) ** 2,
}

#: 输入模式：只有 x，或 (x, E)
TARGET_MODES = ("none", "target")
TARGET_FUNCTIONS = tuple(HHL_FUNCTIONS)


def make_grid(lo: float = HHL_RANGE[0], hi: float = HHL_RANGE[1], n: int = HHL_GRID_POINTS) -> np.ndarray:
    """默认 2500 点的等距网格（与 HHL 论文的实验设置一致）。"""
    return np.linspace(lo, hi, n)


def make_target(name: str, x: np.ndarray) -> np.ndarray:
    """按名字生成目标函数取值。"""
    if name not in HHL_FUNCTIONS:
        raise KeyError(f"unknown target {name!r}; choose from {sorted(HHL_FUNCTIONS)}")
    return np.asarray(HHL_FUNCTIONS[name](np.asarray(x, dtype=float)), dtype=float)


def similarity(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """两个波函数/向量在 L2 内积下的相似度 |⟨a|b⟩| / (||a||·||b||)，返回 [0, 1]。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.trapezoid(a * b, x))
    den = float(np.sqrt(np.trapezoid(a * a, x) * np.trapezoid(b * b, x)))
    return abs(num / den) if den > 0 else 0.0


__all__ = [
    "HHL_RANGE",
    "HHL_GRID_POINTS",
    "HHL_H",
    "HHL_FUNCTIONS",
    "TARGET_FUNCTIONS",
    "TARGET_MODES",
    "make_grid",
    "make_target",
    "similarity",
]
