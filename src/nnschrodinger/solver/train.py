"""训练循环（PyTorch）。三条训练路径共享同一个 Trainer：

* "physics"（无监督 PINN）：每次迭代重新采配点 → 残差 + Rayleigh 能量 + 归一化 + 边界；
* "supervised"（HHL 回归）：(x, E) → ψ_target，只用平方误差；
* "data_assimilation"：物理残差 + 少量观测点上的 ψ 约束（最贴近真实实验的设定）。

工程要点（也是项目里最容易做错的地方）：
1. **配点每次迭代重采样**（默认 2500 点，与 HHL 论文一致），避免在固定网格上过拟合；
2. **验证损失用独立网格上的物理残差**，模型按验证损失早停并回滚到 best checkpoint，
   而不是看训练损失——PINN 训练损失继续下降时解可能已经“跑偏”；
3. 学习率：Adam，PINN 常用 1e-3，HHL 论文用 0.05（可用 --lr 覆盖）；
4. 所有张量统一 float64（物理量精度要求）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class TrainConfig:
    """可序列化到 JSON 的训练配置（复现实验必需）。"""

    mode: str = "physics"
    epochs: int = 2000
    lr: float = 1e-3
    n_collocation: int = 2500
    log_every: int = 200
    validation_every: int = 200
    patience: int = 8
    seed: int = 0
    # 梯度均衡：按各项损失对参数的梯度 RMS 归一后加权求和。
    # 实测残差项与能量项量纲差异大，不均衡时能量在收敛后期会来回抖动（±0.5%）。
    grad_balance: bool = False
    grad_balance_alpha: float = 1.0
    grad_balance_every: int = 1
    # 分项损失的固定权重（step_fn 返回未加权分项时使用）。
    # 空字典表示各项权重都是 1。注意：开启 grad_balance 时，该权重会先乘到
    # 各项上、再被梯度 RMS 归一化抵消，最终相对权重由梯度均衡决定。
    term_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TrainHistory:
    epoch: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    # 梯度均衡后各项的实际权重（便于诊断"谁在主导"）
    weights: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return dict(self.__dict__)


class Trainer:
    """极简训练器：每步由 step_fn 返回 {"loss": ...} 或 {"loss": ..., "energy": ...}。"""

    def __init__(self, model: Any, step_fn: Callable[[int, Trainer], dict[str, Any]], config: TrainConfig) -> None:
        from .backends import require_torch

        self.torch = require_torch("训练")
        self.model = model
        self.step_fn = step_fn
        self.config = config
        self.history = TrainHistory()
        self.optimizer = self.torch.optim.Adam(model.parameters(), lr=config.lr)
        self.best_state: dict[str, Any] | None = None
        self.best_val: float = float("inf")
        self._stale = 0
        self._start = time.perf_counter()

    def _weighted_sum(self, terms: dict[str, Any]) -> tuple[Any, dict[str, float]]:
        """按 config.term_weights 加权求和（未列出的项权重为 1）。"""
        total = None
        used: dict[str, float] = {}
        for name, value in terms.items():
            if value is None:
                continue
            weight = float(self.config.term_weights.get(name, 1.0))
            if weight == 0.0:
                continue
            used[name] = weight
            contribution = weight * value
            total = contribution if total is None else total + contribution
        if total is None:
            raise ValueError("没有任何有效损失项")
        return total, used

    def _balanced_loss(self, terms: dict[str, Any], epoch: int) -> tuple[Any, dict[str, float]]:
        """按梯度 RMS 均衡各损失项后求和。

        做法：逐项 backward 取梯度范数 rms_i，令权重 w_i = target / rms_i，
        其中 target 取当前各项 rms 的最大值。这样量纲小的项（如能量）不会被
        量纲大的项（如残差）完全淹没——这正是 PINN 训练后期的抖动来源。
        """
        torch = self.torch
        params = [p for p in self.model.parameters() if p.requires_grad]
        names, values, rms = [], [], []
        for name, value in terms.items():
            if value is None:
                continue
            self.optimizer.zero_grad(set_to_none=True)
            value.backward(retain_graph=True)
            norm = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
            names.append(name)
            values.append(value)
            rms.append(float(norm.detach()) + 1e-12)
        self.optimizer.zero_grad(set_to_none=True)
        target = max(rms) if rms else 1.0
        weights = {n: target / r for n, r in zip(names, rms)}
        total = None
        for name, value in zip(names, values):
            scaled = self.config.grad_balance_alpha * weights[name] * value
            total = scaled if total is None else total + scaled
        return total, weights

    def fit(self, validation: Callable[[], float] | None = None) -> TrainHistory:
        torch = self.torch
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            info = self.step_fn(epoch, self)
            if "terms" in info:
                # 分项损失：交给 Trainer 决定如何组合（可开启梯度均衡）
                if self.config.grad_balance and epoch % max(1, self.config.grad_balance_every) == 0:
                    loss, weights = self._balanced_loss(info["terms"], epoch)
                else:
                    # 必须应用 term_weights：只对原始分项求和会让归一化项
                    # （权重 100）与残差项（权重 1）等价，残差可借 ψ→0 作弊。
                    loss, weights = self._weighted_sum(info["terms"])
                info["weights"] = weights
            else:
                loss = info["loss"]
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"第 {epoch} 步出现非有限损失: {info}")
            loss.backward()
            self.optimizer.step()

            if epoch % self.config.log_every == 0 or epoch == 1:
                self.history.epoch.append(epoch)
                self.history.loss.append(float(loss.detach()))
                self.history.seconds.append(time.perf_counter() - self._start)
                if "energy" in info:
                    self.history.energy.append(float(info["energy"]))
                if info.get("weights"):
                    self.history.weights.append(dict(info["weights"]))

            if validation is not None and epoch % self.config.validation_every == 0:
                val = float(validation())
                self.history.val_loss.append(val)
                if val < self.best_val * (1.0 - 1e-6):
                    self.best_val = val
                    self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    self._stale = 0
                else:
                    self._stale += 1
                    if self.config.patience and self._stale >= self.config.patience:
                        break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history


def evaluate_on_grid(model: Callable[[Any], Any], x: np.ndarray) -> np.ndarray:
    """在固定网格上求 ψ(x)（断开计算图，返回 numpy）。"""
    from .backends import require_torch

    torch = require_torch("评测")
    with torch.no_grad():
        xt = torch.tensor(np.asarray(x, dtype=float).reshape(-1, 1), dtype=torch.float64)
        return model(xt).cpu().numpy().reshape(-1)


__all__ = ["TrainConfig", "TrainHistory", "Trainer", "evaluate_on_grid"]