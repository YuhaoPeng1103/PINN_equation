"""势函数库：所有 V(x) 都是一维函数，供训练与参考解共用。

物理约定 Ĥ = −½d²/dx² + V(x)。这里额外提供 **训练/采样域** 与 **奇点位置**
两元信息：无穷深势阱和有限深势阱在阱壁处 V 不连续，PIDL 论文需要用
"势阱内外分别采样"（见 inputs.py）来避免残差在跳变点爆炸。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PotentialSpec:
    """一个问题的完整描述：势函数、训练域、跳变位置、深阱标记。"""

    name: str
    V: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float]
    discontinuities: tuple[float, ...] = ()
    deep_well: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.V(np.asarray(x, dtype=float))


def harmonic_oscillator() -> PotentialSpec:
    """V(x) = ½ω²x²，ω = 1。深阱：训练域取 ±6。"""
    return PotentialSpec("harmonic_oscillator", lambda x: 0.5 * x**2, (-6.0, 6.0), (), deep_well=True)


def infinite_well(L: float = 1.0) -> PotentialSpec:
    """无限深方势阱的"有限深近似"：阱内 V=0，阱外 V=1e6（足够大，可被网络学到）。

    跳变点 0 与 L 需要特殊采样；网络在阱外应当学到 ψ ≈ 0。
    """
    barrier = 1.0e6

    def V(x: np.ndarray) -> np.ndarray:
        inside = (x > 0.0) & (x < L)
        return np.where(inside, 0.0, barrier)

    return PotentialSpec("infinite_well", V, (0.0, float(L)), (0.0, float(L)), deep_well=True)


def finite_well(depth: float = 50.0, half_width: float = 1.0) -> PotentialSpec:
    """有限深方势阱 V(x) = −depth (|x| < a)，0 (else)。跳变点 ±a。"""
    a = float(half_width)

    def V(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(x) < a, -float(depth), 0.0)

    return PotentialSpec("finite_well", V, (-8.0 * a, 8.0 * a), (-a, a))


def poschl_teller(alpha: float = 1.0) -> PotentialSpec:
    """Pöschl–Teller 势 V(x) = −α(α+1)/2 · sech²(x)，有精确解：
    E_n = −(α−n)²/2, n = 0..floor(α)−1，用于变分方法之外的另一种解析基准。"""
    a = float(alpha)

    def V(x: np.ndarray) -> np.ndarray:
        ch = np.cosh(np.clip(np.asarray(x, dtype=float), -30.0, 30.0))
        return -0.5 * a * (a + 1.0) / ch**2

    return PotentialSpec("poschl_teller", V, (-12.0, 12.0))


POTENTIALS: dict[str, Callable[..., PotentialSpec]] = {
    "harmonic": harmonic_oscillator,
    "harmonic_oscillator": harmonic_oscillator,
    "infinite_well": infinite_well,
    "finite_well": finite_well,
    "poschl_teller": poschl_teller,
}


def get_potential(name: str, **kwargs) -> PotentialSpec:
    key = name.lower()
    if key not in POTENTIALS:
        raise KeyError(f"unknown potential {name!r}; choose from {sorted(POTENTIALS)}")
    return POTENTIALS[key](**kwargs)


__all__ = [
    "PotentialSpec",
    "POTENTIALS",
    "get_potential",
    "harmonic_oscillator",
    "infinite_well",
    "finite_well",
    "poschl_teller",
]
