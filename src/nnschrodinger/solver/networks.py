"""网络结构。需要 PyTorch（``solver.backends.require_torch``）。

三类结构对应三条实验线：

``build_mlp``
    全连接 tanh 网络，可选随机傅里叶特征。项目的**主力网络**。
    傅里叶特征缓解 PINN 的谱偏置，是能否与前几个本征态对齐的关键。

``build_hhl_network``
    复现 HHL 论文的量子电路式 DNN：角度嵌入 + 纠缠层堆叠 + 输出线性层，
    输入为 ``(x, E)``，输出为 ψ（可再输出 E_hat 做逆问题）。
    论文超参：hidden=10、嵌入层=3、纠缠层=4、输出=2、batch=100、Adam lr=0.05。

``SpectralSolver``
    谱方法基线：在 [lo, hi] 上取 N 点**有限差分 + 三对角 Jacobi 特征分解**
    （``physics.eigen``），与网络共享同一区间与势。它是“同等信息量下的精度上限”，
    评测里用来自证“神经网络这道题能到多准、离谱方法差多少”。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..physics.eigen import symmetric_tridiagonal_eigh

FOURIER_FEATURES = 16


def build_mlp(
    in_dim: int = 1,
    out_dim: int = 1,
    width: int = 200,
    depth: int = 4,
    fourier: bool = False,
    fourier_scale: float = 3.0,
    seed: int = 0,
) -> Any:
    """L 层 tanh MLP；``fourier=True`` 时先经傅里叶特征升维。"""
    from .backends import require_torch

    torch = require_torch("网络")
    torch.manual_seed(seed)
    layers: list[Any] = []
    d_in = in_dim
    if fourier:
        # 二维输入同样适用：B 的形状是 (n_features, in_dim)，投影后
        # 仍是 [sin, cos] 拼接。in_dim=1 时随机数的形状与旧实现一致
        # （(n_features,1) 与 (n_features,) 的 unsqueeze 相同），
        # 因此一维实验的可复现性不受影响。
        layers.append(FourierFeatures(scale=fourier_scale, n_features=FOURIER_FEATURES,
                                      seed=seed, in_dim=in_dim))
        d_in = 2 * FOURIER_FEATURES
    layers += [torch.nn.Linear(d_in, width), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    layers.append(torch.nn.Linear(width, out_dim))
    net = torch.nn.Sequential(*layers)
    net.double()  # 物理量精度要求高，统一 float64
    return net


class FourierFeatures:
    """随机傅里叶特征 ``x -> [sin(2πBx), cos(2πBx)]``，B 为固定随机矩阵。

    谱偏置（spectral bias）是 PINN 精度天花板的主要来源：MLP 倾向先学低频，
    对高频势/高激发态收敛缓慢。把坐标投影到高频基上能显著缓解。
    """

    def __new__(
        cls,
        scale: float = 3.0,
        n_features: int = FOURIER_FEATURES,
        seed: int = 0,
        in_dim: int = 1,
    ) -> Any:
        from .backends import require_torch

        torch = require_torch("傅里叶特征")
        gen = torch.Generator().manual_seed(seed)

        class _Fourier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # (n_features, in_dim)：一维时与旧的 (n_features,) 逐元素相同
                self.register_buffer(
                    "B", torch.randn(n_features, int(in_dim), generator=gen) * float(scale)
                )

            def forward(self, x: Any) -> Any:
                proj = 2.0 * torch.pi * x @ self.B.T
                return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        net = _Fourier()
        net.double()
        return net


def build_hhl_network(hidden: int = 10, n_layers: int = 3, out_dim: int = 2, seed: int = 0) -> Any:
    """``(x, E) -> (ψ, E_hat)`` 的 HHL 风格网络（残差连接稳定多层堆叠）。"""
    from .backends import require_torch

    torch = require_torch("HHL 网络")
    torch.manual_seed(seed)

    class _HHL(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(2, hidden)
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.Tanh()) for _ in range(n_layers)]
            )
            self.head = torch.nn.Linear(hidden, out_dim)

        def forward(self, x: Any) -> Any:
            h = torch.tanh(self.input(x))
            for block in self.blocks:
                h = h + block(h)
            return self.head(h)

    net = _HHL()
    net.double()
    return net


class SpectralSolver:
    """经典数值基线：Numerov 积分 + 打靶（与参考解共用同一实现）。

    ``SpectralSolver(lo, hi, potential, energy_window=None, n_states=6)``
    ``.energy(n)`` 返回第 n 个本征值，``.psi(x, n)`` 返回归一化波函数。

    ``energy_window`` 必须覆盖要用的全部本征值；缺省时按势的极值自动给出
    足够宽的范围（下限 = 势最小值，上限 = 最小值 + 200）。
    """

    def __init__(
        self,
        lo: float,
        hi: float,
        potential: Callable[[np.ndarray], np.ndarray],
        energy_window: tuple[float, float] | None = None,
        n_states: int = 6,
        n_points: int = 4001,
    ) -> None:
        from ..physics.numerov import numerov_spectrum

        grid = np.linspace(lo, hi, int(n_points))
        values = np.asarray(potential(grid), dtype=float)
        vmin = float(np.min(values))
        window = energy_window or (vmin, vmin + 200.0)
        result = numerov_spectrum(potential, lo, hi, window, n_states, n_points=int(n_points))
        self._energies = result.energies
        self._grid = result.grid
        self._vectors = result.vectors
        self.potential = potential

    def energy(self, n: int) -> float:
        return float(self._energies[n])

    def psi(self, x: np.ndarray, n: int = 0) -> np.ndarray:
        v = self._vectors[:, n]
        a, b = v[0], v[-1]
        if (b if abs(b) > abs(a) else a) < 0:
            v = -v
        return np.interp(np.asarray(x, dtype=float), self._grid, v)

__all__ = ["build_mlp", "build_hhl_network", "FourierFeatures", "SpectralSolver", "FOURIER_FEATURES"]