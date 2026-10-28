"""训练与评测所需的采样器。纯 numpy。

三种采样策略对应三类研究问题：

1. ``uniform_collocation``  —— PINN 的配点：每次迭代在 [lo, hi] 内重新均匀采样，
   避免网络只在固定网格上"记住"答案。
2. ``energy_blocks``       —— HHL 论文的 ``(x, E)`` 双输入：同一个 x 会与不同 E 配对，
   网络学到的必须是"给定能量时对应的本征函数"这一映射本身。
3. ``well_aware_collocation`` —— 针对含跳变势（无限深/有限深方势阱）的配点：
   在 V 的跳变点两侧加采样，避免残差在跳变处爆炸（PIDL 论文的处理）。

此外 ``hhl_input_map`` 实现论文中"把 x 映射到远离中心的区间"这一技巧：
    x -> (x − c)·M，使网络在小 x 区域有足够分辨率、大 x 区域自动饱和。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Samples:
    """一批样本。

    x : (N,) 或 (N, 1) 的空间坐标
    E : 标量能量（PINN 时是"要解的那个本征值"，HHL 时是该样本对应的能量）
    """

    x: np.ndarray
    E: float

    def as_inputs(self, energy_augmented: bool = False) -> np.ndarray:
        """转为网络输入矩阵。``energy_augmented`` 为 True 时输出 (N, 2) 的 (x, E)。"""
        x = np.asarray(self.x, dtype=float).reshape(-1, 1)
        if not energy_augmented:
            return x
        return np.concatenate([x, np.full_like(x, float(self.E))], axis=1)


def uniform_collocation(lo: float, hi: float, n: int, rng: np.random.Generator, energy: float = 0.0) -> Samples:
    """在 [lo, hi] 内均匀采样 n 个配点。"""
    if n <= 0:
        raise ValueError("n must be positive")
    return Samples(np.sort(rng.uniform(lo, hi, int(n))), float(energy))


def energy_blocks(lo: float, hi: float, energies: list[float], n_per_energy: int,
                  rng: np.random.Generator) -> list[Samples]:
    """为每个能量各采一组配点，返回 ``list[Samples]``（每组共享一个 E）。"""
    return [uniform_collocation(lo, hi, n_per_energy, rng, energy=e) for e in energies]


def well_aware_collocation(lo: float, hi: float, edges: tuple[float, ...], n: int,
                           rng: np.random.Generator, energy: float = 0.0,
                           n_near_edge: int = 8) -> Samples:
    """按跳变点 ``edges`` 把区间切段，每段按长度比例采样；并在每个跳变点左右
    各 ``n_near_edge`` 个点加密（固定小偏移），让残差能约束跳变两侧。"""
    breaks = sorted({float(lo), *map(float, edges), float(hi)})
    xs: list[np.ndarray] = []
    lengths = np.diff(breaks)
    weights = lengths / lengths.sum()
    alloc = np.maximum(2, np.round(weights * n).astype(int))
    for (a, b), m in zip(zip(breaks[:-1], breaks[1:]), alloc):
        xs.append(rng.uniform(a, b, int(m)))
    eps = 1e-3 * (hi - lo)
    for e in breaks:
        xs.append(np.linspace(e - eps, e + eps, 2 * n_near_edge))
    x = np.clip(np.concatenate(xs), lo, hi)
    return Samples(np.sort(x), float(energy))


def hhl_input_map(x: np.ndarray, c: float, M: float) -> np.ndarray:
    """HHL 论文的输入映射 ``x -> (x − c)·M``。

    论文用一个大 M 把训练区间压缩到中心附近，使网络在小 x 区间有高分辨率；
    这里保持线性（论文的精确形式也几乎线性），只在参数说明中给出 M 的取法。
    """
    return (np.asarray(x, dtype=float) - float(c)) * float(M)


def hhl_pairs(lo: float, hi: float, energies: list[float], n: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """生成 (N, 2) 的 (x, E) 输入对：x 与 E 独立采样。"""
    xs = rng.uniform(lo, hi, n)
    es = rng.choice(np.asarray(energies, dtype=float), size=n)
    return np.stack([xs, es], axis=1), es


def dirichlet_points(lo: float, hi: float) -> np.ndarray:
    """Dirichlet 边界点（两端各一个）。"""
    return np.array([float(lo), float(hi)], dtype=float).reshape(-1, 1)


__all__ = [
    "Samples",
    "uniform_collocation",
    "energy_blocks",
    "well_aware_collocation",
    "hhl_input_map",
    "hhl_pairs",
    "dirichlet_points",
]
