"""能量泛函的离散化：梯度与拉普拉斯。

深度学习框架按需导入（PyTorch）。为了让 `losses.py` 不绑定具体框架，
这里定义 `EnergyOps` 协议：`grad(u)`、`laplacian(u)` 之外再加 `energy_expectation()`，
默认实现即论文中的

    E = ∫ [½|ψ'|² + V(x)|ψ|²] dx / ∫ |ψ|² dx        （真实 ψ 上等价于 ⟨Ĥ⟩）

**为什么用这个形式而不是直接 ∂²ψ/∂x²？** 因为 `½∫|ψ'|²` 对 ψ 的导数只到一阶，
二阶导数通过自动微分嵌套也能算，但 `|ψ'|²` 形式对噪声更稳健，且天然对应
变分下界（Rayleigh quotient），这是 `losses.py` 的变分损失的基础。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EnergyOps(Protocol):
    """一维函数上所需的最小算子集合。"""

    def grad(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def laplacian(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def trapezoid_scalar(u: np.ndarray, x: np.ndarray) -> float:
    """∫u dx 的梯形近似（numpy 后端，用于测试与阈值校验）。"""
    return float(np.trapezoid(u, x))


def gradient_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """等距网格上的二阶中心差分（边界用一阶）。"""
    return np.gradient(y, x, edge_order=1)


def laplacian_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """等距网格上的二阶差分 Δψ = (ψ_{i+1} − 2ψ_i + ψ_{i−1})/h²。"""
    h = float(x[1] - x[0])
    lap = np.zeros_like(y)
    lap[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / h**2
    lap[0] = lap[1]
    lap[-1] = lap[-2]
    return lap


def hamiltonian_apply_numpy(x: np.ndarray, psi: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Ĥψ = −½ψ'' + Vψ（numpy 后端，供单元测试做“残差必须为零”的断言）。"""
    return -0.5 * laplacian_numpy(x, psi) + V * psi


__all__ = [
    "EnergyOps",
    "trapezoid_scalar",
    "gradient_numpy",
    "laplacian_numpy",
    "hamiltonian_apply_numpy",
]
