"""损失函数：无监督（PINN）、数据同化、HHL 回归三类，按“可独立测试的项”组织。

A. 物理项（无标签）
   * residual            ‖Ĥψ − Eψ‖² 的逐点均值；
   * energy              Rayleigh 商 E[ψ] = ⟨ψ|Ĥ|ψ⟩/⟨ψ|ψ⟩（无需预先知道本征值）；
   * normalization       (∫|ψ|² dx − 1)²；
   * boundary            Dirichlet 边界点上的 ψ² 之和；
   * orthogonality       Gram 矩阵非对角元平方和（求激发态时用）。

B. 数据项（可选）
   * supervised          ‖ψ − ψ_target‖²，HHL 回归任务的唯一损失。

C. 正则化项
   * l2                  参数 L2（论文里以 weight decay 出现）；
   * curvature            ∫|ψ″|² 的惩罚，压制非物理的高频振荡。

数值要点（论文里分散在各处的做法，这里统一到 Losses.compute）：
* 能量泛函里先除以归一化因子，避免 |ψ|² 的量级差异导致梯度失衡；
* 二阶导用 autograd.grad(create_graph=True) 嵌套求；
* 各项量纲不同，给出 grad_balance 做梯度 RMS 均衡。

本模块按需导入 PyTorch：函数体内部才调用 require_torch，所以顶层导入是安全的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import math


@dataclass
class LossWeights:
    """各损失项权重（默认值取自 Markham 2020 / HHL 2021 的常见设定）。"""

    residual: float = 1.0
    energy: float = 1.0
    normalization: float = 100.0
    boundary: float = 100.0
    orthogonality: float = 1.0
    supervised: float = 1.0
    l2: float = 1e-6
    curvature: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class LossBreakdown:
    """逐项损失，便于打印与调试。"""

    total: float
    terms: dict[str, float]

    def __str__(self) -> str:
        parts = " ".join(f"{k}={v:.3e}" for k, v in self.terms.items())
        return f"total={self.total:.4e} {parts}"


class Losses:
    """在 PyTorch 上实现所有损失项；compute(terms) 按权重组合。"""

    def __init__(self, weights: LossWeights | None = None, dx: float = 1.0 / 2500.0) -> None:
        from .backends import require_torch

        self.torch = require_torch("损失函数")
        self.weights = weights or LossWeights()
        self.dx = float(dx)

    # -- 单点算子 --------------------------------------------------------
    def _grad(self, y: Any, x: Any) -> Any:
        """对 x 求一阶导。

        两件事必须在这里处理，否则调用方会踩坑：
        1. x 往往直接来自 torch.linspace/tensor（没有 requires_grad），而残差/能量项
           需要对它求二阶导——这里统一打开，避免每个调用点都记得 set；
        2. x 可能并未真正参与 y 的计算（例如模型返回常数），此时 autograd 会抛
           'not used in the graph'；用 allow_unused 转成可读的错误信息。
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        grad, = self.torch.autograd.grad(
            y, x, grad_outputs=self.torch.ones_like(y), create_graph=True, allow_unused=True
        )
        if grad is None:
            raise RuntimeError(
                '模型输出与输入 x 无计算图连接：请确认 forward 真的用到了 x'
            )
        return grad

    def grad(self, y: Any, x: Any) -> Any:
        return self._grad(y, x)

    def laplacian(self, y: Any, x: Any) -> Any:
        return self._grad(self._grad(y, x), x)
    def integral(self, y: Any) -> Any:
        return self.torch.sum(y) * self.dx

    # -- 物理项 ----------------------------------------------------------
    def _prepare(self, x: Any) -> Any:
        """确保 x 可以作为求导起点。

        只在 x 是叶子时才需要处理：
          * 需要梯度 -> 直接用；
          * 不需要梯度 -> 打开；
          * 非叶子（已参与上游计算）-> 原样返回，否则 detach 会切断计算图。

        最后一种情况曾经导致一个隐蔽 bug：在 no_grad 下算出的 net(g) 和常量张量
        都会被 detach，autograd 拿不到 dψ/dx，Rayleigh 商变成错误值（实测 6.38 vs 1.5）。
        """
        if x.is_leaf:
            return x if x.requires_grad else x.detach().requires_grad_(True)
        return x

    def residual(self, model: Callable[[Any], Any], x: Any, energy: float | Any,
                 potential: Callable[[Any], Any]) -> Any:
        """稳态残差的均方：−½ψ″ + (V − E)ψ。

        注意：x 必须先变成 requires_grad 的叶子**再喂给 model**，否则 model(x)
        的输出与用于求导的 x 不是同一个张量，autograd 会报“does not require grad”。
        """
        x = self._prepare(x)
        psi = model(x)
        psi_xx = self.laplacian(psi, x)
        r = -0.5 * psi_xx + (potential(x) - energy) * psi
        return (r**2).mean()

    def energy_projected(
        self,
        model: Callable[[Any], Any],
        x: Any,
        potential: Callable[[Any], Any],
        projector: Any,
    ) -> tuple[Any, Any]:
        """带正交约束的 Rayleigh 商（求激发态用）。

        约束是 ψ ⟂ {φ_k}（k < n），即 minimize R(ψ) s.t. ⟨ψ|φ_k⟩ = 0。

        两个必须注意的点：
        1. **只投影 ψ，不投影 ψ′**。约束空间是 "与低阶态正交的函数"，其切空间
           的基是 φ_k′ 而不是 φ_k。曾经用 φ_k 去投影 ψ′，等于减掉了一个本不属于
           该空间的分量——实测把 dψ/dx 在中点从 1.062 削到 0.531（正好一半），
           Rayleigh 商因此变成 1.25 而不是 1.5。
        2. autograd 必须能对 ψ 求导，所以要在投影**之前**取导数。
        """
        x = self._prepare(x)
        psi = model(x)
        dpsi = self.grad(psi, x)
        psi_p = projector(psi)  # 只投影波函数本身
        norm = self.integral(psi_p**2)
        kinetic = 0.5 * self.integral(dpsi**2)
        potential_energy = self.integral(potential(x) * psi_p**2)
        e = (kinetic + potential_energy) / (norm + 1e-8)
        return e, norm

    def residual_projected(
        self,
        model: Callable[[Any], Any],
        x: Any,
        energy: float | Any,
        potential: Callable[[Any], Any],
        projector: Any,
    ) -> Any:
        """带正交约束的残差项。

        Ĥ 与投影算子 P 可交换（P 作用在空间坐标上、Ĥ 作用在同一个网格上，
        且 P 是常数矩阵），所以 (ĤPψ − EPψ) = P(Ĥψ − Eψ)：
        先作用 Ĥ 再投影即可，不必对导数做任何额外处理。
        """
        x = self._prepare(x)
        psi = model(x)
        psi_xx = self.laplacian(psi, x)
        r = projector(-0.5 * psi_xx + (potential(x) - energy) * psi)
        return (r**2).mean()

    def energy(self, model: Callable[[Any], Any], x: Any, potential: Callable[[Any], Any]) -> tuple[Any, Any]:
        """Rayleigh 商 E = (½∫|ψ′|² + ∫V|ψ|²) / ∫|ψ|²，返回 (E, 归一化因子)。

        ⚠️ 调用方必须传入**确定性等距网格**，不要传每步重采样的随机点。

        实测（谐振子基态，见 docs/results.md）：用 2500 个均匀随机点估计 Rayleigh
        商时，估计量本身的采样标准差是 **1.9%**——这与训练是否收敛无关，纯粹是
        用 mean 当积分带来的蒙特卡洛噪声，会让能量曲线始终在 ±1% 抖动。
        换成同数量的等距网格后误差降到 1e-8。残差损失可以随机采样（它本质是
        逐点均方），但**能量项必须是确定性的**。
        """
        x = self._prepare(x)
        psi = model(x)
        dpsi = self.grad(psi, x)
        norm = self.integral(psi**2)
        kinetic = 0.5 * self.integral(dpsi**2)
        potential_energy = self.integral(potential(x) * psi**2)
        e = (kinetic + potential_energy) / (norm + 1e-8)
        return e, norm
    def normalization(self, model: Callable[[Any], Any], x: Any) -> Any:
        psi = model(x)
        return (self.integral(psi**2) - 1.0) ** 2

    def boundary(self, model: Callable[[Any], Any], points: Any) -> Any:
        return (model(points) ** 2).sum()

    def orthogonality(self, models: Sequence[Callable[[Any], Any]], x: Any) -> Any:
        """Gram 矩阵非对角元平方和（用于依次训练激发态）。"""
        if len(models) < 2:
            return self.torch.zeros((), dtype=self.torch.float64)
        vals = self.torch.cat([m(x) for m in models], dim=1)
        gram = vals.T @ vals * self.dx
        off = gram - self.torch.diag(self.torch.diag(gram))
        return (off**2).sum()

    def l2(self, parameters: Any) -> Any:
        total = self.torch.zeros((), dtype=self.torch.float64)
        for p in parameters:
            total = total + (p**2).sum()
        return total

    def supervised(self, pred: Any, target: Any) -> Any:
        return ((pred - target) ** 2).mean()

    # -- 组合 ------------------------------------------------------------
    def compute(self, **terms: Any) -> LossBreakdown:
        """按权重加权求和；值为 None 或权重为 0 的项忽略。"""
        total = self.torch.zeros((), dtype=self.torch.float64)
        named: dict[str, float] = {}
        for name, value in terms.items():
            if value is None:
                continue
            w = getattr(self.weights, name, 0.0)
            if w == 0.0:
                continue
            named[name] = float(value.detach())
            total = total + w * value
        return LossBreakdown(total=float(total.detach()), terms=named)


def total_loss(losses: Losses, **terms: Any) -> Any:
    """只返回可反向传播的标量（训练循环用）。"""
    total = None
    for name, value in terms.items():
        if value is None:
            continue
        w = getattr(losses.weights, name, 0.0)
        if w == 0.0:
            continue
        contribution = w * value
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("没有任何有效损失项")
    return total


def discrete_dirichlet_terms(
    psi: Any,
    potential: Any,
    h: float,
    energy: Any,
    *,
    use_energy: bool = True,
) -> dict[str, Any]:
    """固定网格上的 H_h 残差/归一化/边界/能量项（与后验证书同口径）。

    参数
    ----
    psi : (N,) 或 (N,1) 的 torch 张量，定义在含端点的整张网格上
    potential : 同形状的网格势能
    h : 网格步长
    energy : 残差 ``H_h ψ − Eψ`` 中的 E（标量或可学习张量）

    实现要点
    --------
    **残差**用真实端点（它就是“把网络输出当成区间上的光滑函数”的 PDE
    残差）；**能量项**必须用 Dirichlet 延拓（端点置零）后的二次型。
    若能量项也直接用非零端点，二次型会丢边界项、从而**没有下界**——V=0
    的无限深阱也能算出负能量，优化器会靠放大端点把损失推到负无穷
    （见工程记录 18）。端点值另由 ``boundary`` 项约束。

    也不要反过来把残差里的端点也置零：那会在边界处制造人为跳变，
    残差永远压不下去（实测 3000 epoch 后 overlap 只有 0.64）。

    返回的字典与 ``Trainer(term_weights=...)`` 的键名一致。
    """
    from .backends import require_torch

    torch = require_torch("离散残差项")
    psi = psi.reshape(-1)
    potential = potential.reshape(-1)
    psi_i = psi[1:-1]
    # 残差：真实端点（区间上的光滑函数）
    lap = (psi[2:] - 2.0 * psi_i + psi[:-2]) / h**2
    h_psi = -0.5 * lap + potential[1:-1] * psi_i
    r = h_psi - energy * psi_i
    norm2 = h * torch.sum(psi_i**2)
    terms = {
        "residual": torch.mean(r**2),
        "normalization": (norm2 - 1.0) ** 2,
        "boundary": psi[0] ** 2 + psi[-1] ** 2,
    }
    if use_energy:
        # 能量：Dirichlet 延拓（端点置零），保证二次型 ≥ λ₀‖ψ‖²
        zeros = psi.new_zeros(1)
        psi_b = torch.cat((zeros, psi_i, zeros))
        lap_b = (psi_b[2:] - 2.0 * psi_i + psi_b[:-2]) / h**2
        h_psi_b = -0.5 * lap_b + potential[1:-1] * psi_i
        terms["energy"] = h * torch.sum(psi_i * h_psi_b) / norm2
    return terms


def grad_balance_norm(scalars: Sequence[Any], parameters: Any) -> list[float]:
    """按梯度 RMS 给出各项的均衡系数（需先各自 backward 取到梯度）。"""
    from .backends import require_torch

    torch = require_torch("梯度均衡")
    rms: list[float] = []
    for scalar in scalars:
        parameters.zero_grad(set_to_none=True)
        scalar.backward(retain_graph=True)
        sq = [float((p.grad**2).sum()) for p in parameters.parameters() if p.grad is not None]
        rms.append(math.sqrt(sum(sq)) if sq else 0.0)
    reference = max(rms) if rms else 1.0
    return [reference / (r + 1e-12) for r in rms]


__all__ = ["LossWeights", "LossBreakdown", "Losses", "total_loss", "grad_balance_norm",
           "discrete_dirichlet_terms"]