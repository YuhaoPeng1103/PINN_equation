"""训练循环的冒烟测试：在谐波基态上跑极少的步数，确认不崩、损失下降。"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch

torch = pytest.importorskip("torch")

from nnschrodinger.solver.losses import LossWeights, Losses, total_loss  # noqa: E402
from nnschrodinger.solver.networks import build_mlp  # noqa: E402
from nnschrodinger.solver.train import TrainConfig, Trainer, evaluate_on_grid  # noqa: E402


@requires_torch
def test_trainer_reduces_loss():
    torch.manual_seed(0)
    net = build_mlp(width=32, depth=3, seed=0)
    losses = Losses(LossWeights(residual=1.0, normalization=10.0, boundary=0.0))
    config = TrainConfig(epochs=80, lr=1e-2, n_collocation=256, log_every=10, validation_every=10)

    def step(epoch, trainer):
        # 配点必须在 [−6, 6] 内且带梯度（Losses 内部也会处理，这里显式更清楚）
        x = torch.rand(256, 1, dtype=torch.float64) * 12.0 - 6.0
        return {
            "loss": total_loss(
                losses,
                normalization=losses.normalization(net, x),
                residual=losses.residual(net, x, 0.5, lambda t: 0.5 * t**2),
            )
        }

    trainer = Trainer(net, step, config)
    history = trainer.fit()
    assert len(history.loss) >= 2
    assert history.loss[-1] < history.loss[0]

def test_evaluate_on_grid_shape():
    net = build_mlp(width=8, depth=2, seed=1)
    x = np.linspace(-1, 1, 33)
    out = evaluate_on_grid(net, x)
    assert out.shape == (33,)
    assert np.all(np.isfinite(out))


@requires_torch
def test_trainer_detects_non_finite():
    net = build_mlp(width=4, depth=1, seed=0)
    losses = Losses()
    config = TrainConfig(epochs=3, lr=1e-2, log_every=1)

    def step(epoch, trainer):
        return {"loss": torch.tensor(float("nan"), dtype=torch.float64, requires_grad=True)}

    trainer = Trainer(net, step, config)
    with pytest.raises(FloatingPointError):
        trainer.fit()

def test_grad_balance_produces_weights():
    """开启梯度均衡后，Trainer 应记录每项的实际权重，且量纲小的项权重更大。"""
    torch.manual_seed(0)
    net = build_mlp(width=16, depth=2, seed=0)
    losses = Losses()
    config = TrainConfig(epochs=5, lr=1e-2, log_every=1, grad_balance=True)

    def step(epoch, trainer):
        x = torch.rand(32, 1, dtype=torch.float64) * 12.0 - 6.0
        return {
            "terms": {
                # 残差项量纲大（含 1/h² 级因子），能量项量纲小
                "residual": losses.residual(net, x, 0.5, lambda t: 0.5 * t**2),
                "energy": losses.energy(net, x, lambda t: 0.5 * t**2)[0],
            }
        }

    trainer = Trainer(net, step, config)
    history = trainer.fit()
    assert history.weights, "未记录梯度均衡权重"
    first = history.weights[0]
    assert set(first) == {"residual", "energy"}
    assert all(w > 0 for w in first.values())
    # 均衡后目标是最小 rms 的项被放大到与最大者同量级 -> 其权重应为 1，
    # 最大 rms 的项权重应 < 1
    # 权重 = target / (rms + 1e-12)，因此最小 rms 的项权重略大于或等于 1，
    # 所有项权重都应落在 [1, 1e3] 这个量级内（不会出现 1e8 级的失衡）
    assert 1.0 - 1e-6 <= min(first.values()) <= 1.0 + 1e-6
    assert max(first.values()) >= 1.0
    # 学习过程仍应有效
    assert history.loss[-1] < history.loss[0]


def test_grad_balance_off_uses_plain_sum():
    """关闭均衡且未给 term_weights 时，总损失应是各分项的直接相加。"""
    torch.manual_seed(0)
    net = build_mlp(width=8, depth=2, seed=0)
    config = TrainConfig(epochs=1, lr=1e-2, log_every=1, grad_balance=False)

    def step(epoch, trainer):
        w = next(net.parameters())
        # 两项都可训练，其和 = 3·w0 对齐形状后的标量
        a = (w.sum()) ** 2
        b = 2.0 * (w.sum()) ** 2
        return {"terms": {"a": a, "b": b}}

    history = Trainer(net, step, config).fit()
    assert history.weights[0] == {"a": 1.0, "b": 1.0}


def test_term_weights_are_applied():
    """回归：Trainer 必须应用 term_weights，而不是把原始分项直接相加。

    exp2 曾把 normalization=100 传给 LossWeights，但 Trainer 只做 sum(terms)，
    结果归一化项按 1 生效，残差项靠 ψ→0 就能把损失压到最低。
    """
    torch.manual_seed(0)
    net = build_mlp(width=8, depth=2, seed=0)
    config = TrainConfig(
        epochs=1, lr=0.0, log_every=1, grad_balance=False,
        term_weights={"a": 10.0, "b": 1.0},
    )

    def step(epoch, trainer):
        zero = sum(p.sum() for p in net.parameters()) * 0.0
        return {"terms": {"a": zero + 1.0, "b": zero + 2.0}}

    history = Trainer(net, step, config).fit()
    # 10·1 + 1·2 = 12（未加权时是 3）
    assert history.loss[0] == pytest.approx(12.0)
    assert history.weights[0] == {"a": 10.0, "b": 1.0}