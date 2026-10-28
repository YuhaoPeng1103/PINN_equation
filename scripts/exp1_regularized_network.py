"""实验一：正则化（有监督回归）网络求解本征态 —— 复现 HHL 论文“正则化网络”分支。

设定（与论文一致的部分）：
  * 输入 (x, E)：能量作为第二输入通道，网络学的是“给定能量时的本征函数”；
  * 训练集只覆盖 E = {0.5, 1.5, 2.5, 3.5} 对应的精确本征函数；
  * 在能量**取中间值**（如 E = 2.0）时，网络应能外推出对应本征函数——
    这正是论文“用 (x, E) 做插值”的核心主张；
  * 归一化约定与论文一致：max|ψ| = 1。

本脚本额外做一件论文没有明说、但对结论很关键的事：把网络在
“训练能量”和“插值能量”上的误差与经典线性插值阈值一并报告，
给出“神经网络 vs 经典插值”的精度对照（论文结论：二者等价）。

依赖 PyTorch；当前沙箱环境未安装 torch，脚本会打印环境信息后以退出码 2 结束。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.backends import describe_environment, has_torch  # noqa: E402

TRAIN_ENERGIES = (0.5, 1.5, 2.5, 3.5)
INTERP_ENERGIES = (1.0, 2.0, 3.0)
DOMAIN = (-6.0, 6.0)
N_POINTS = 2500


# max|ψ| = 1 归一化下的谐振子本征函数峰值（预先解析计算，避免每个样本都调用 get_reference）
_PEAK: dict[int, float] = {}


def _peak(n: int) -> float:
    """raw(x) = e^{-x²/2}·H_n(x) 的全局峰值 max_x |raw(x)|。

    要使 ψ = raw / scale 满足 max|ψ| = 1，必须取 scale = max|raw|。
    （先后两次写错过：用 ‖raw‖ 归一得到的是 L2 归一化；用"传入点的峰值"会让
    不同批次的缩放不一致——两者都会让训练目标自相矛盾。）
    """
    if n not in _PEAK:
        from nnschrodinger.physics.spectrum import _hermite

        grid = np.linspace(-12.0, 12.0, 80001)
        raw = np.exp(-0.5 * grid**2) * _hermite(n, grid)
        _PEAK[n] = float(np.max(np.abs(raw)))
    return _PEAK[n]


def harmonic(n: int, x: np.ndarray) -> np.ndarray:
    """max|ψ| = 1 的谐振子本征函数（与 HHL 论文的归一化一致）。

    向量化实现：H_n 用三项递推、一次性作用在整个数组上；峰值预先解析算好，
    避免每批 2500 个样本都去构造 ReferenceSolution（那会让每批数据慢 100 倍）。

    注意：峰值必须是**整条曲线**上的最大值，而不是"传入这批点里的最大值"，
    否则同一个波函数在不同批次会得到不同缩放，训练目标自相矛盾。
    """
    from nnschrodinger.physics.spectrum import _hermite

    xs = np.asarray(x, dtype=float)
    return np.exp(-0.5 * xs**2) * _hermite(n, xs) / _peak(n)


def energy_to_index(e: float) -> int:
    """能量 E 对应的谐振子量子数 n：E_n = n + ½，所以 n = E − ½。"""
    return int(round(e - 0.5))


def reference_for(e: float, grid: np.ndarray) -> np.ndarray:
    """精确本征能量取解析解；插值能量取相邻两能级的线性插值。"""
    n = energy_to_index(e)
    # 判断"是否恰好是本征能量"必须用 n + 0.5 == e，而不是 n == e
    # （曾经写成 abs(n - e) < 1e-9，导致 E=0.5 与 E=1.5 都被判为插值点）
    if abs((n + 0.5) - e) < 1e-9:
        return harmonic(n, grid)
    # 相邻两个本征态线性插值（经典基线）：E 介于 n+½ 与 (n+1)+½ 之间
    lo_n = energy_to_index(e)
    hi_n = lo_n + 1
    w = e - (lo_n + 0.5)
    return (1.0 - w) * harmonic(lo_n, grid) + w * harmonic(hi_n, grid)


def run(seed: int = 0, epochs: int = 4000, width: int = 64, depth: int = 3, out_dir: Path | None = None) -> dict:
    import torch

    from nnschrodinger.solver.losses import LossWeights, Losses, total_loss
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_mlp(in_dim=2, width=width, depth=depth, seed=seed)
    losses = Losses(LossWeights(supervised=1.0, l2=1e-6))
    config = TrainConfig(mode="supervised", epochs=epochs, lr=1e-3, n_collocation=N_POINTS, log_every=500)

    def batch():
        xs = rng.uniform(*DOMAIN, N_POINTS)
        es = rng.choice(np.asarray(TRAIN_ENERGIES), N_POINTS)
        # 按能量分组后整批向量化求值（逐个 python 循环会让每批慢上百倍）
        target = np.empty(N_POINTS, dtype=np.float64)
        for ev in np.unique(es):
            mask = es == ev
            target[mask] = harmonic(energy_to_index(float(ev)), xs[mask])
        x = torch.tensor(np.stack([xs, es], axis=1), dtype=torch.float64)
        y = torch.tensor(target.reshape(-1, 1), dtype=torch.float64)
        return x, y

    def step(epoch: int, trainer: Trainer) -> dict:
        x, y = batch()
        return {"loss": total_loss(losses, supervised=losses.supervised(net(x), y))}

    history = Trainer(net, step, config).fit()

    grid = np.linspace(*DOMAIN, 1001)
    results: dict = {
        "environment": describe_environment(),
        "seed": seed,
        "epochs": epochs,
        "width": width,
        "depth": depth,
        "final_loss": history.loss[-1] if history.loss else None,
        "energies": {},
    }
    with torch.no_grad():
        for e in (*TRAIN_ENERGIES, *INTERP_ENERGIES):
            inp = torch.tensor(np.stack([grid, np.full_like(grid, e)], axis=1), dtype=torch.float64)
            pred = net(inp).numpy().reshape(-1)
            ref = reference_for(e, grid)
            is_interp = abs(energy_to_index(e) + 0.5 - e) > 1e-9
            results["energies"][f"{e:.2f}"] = {
                "max_abs_err": float(np.max(np.abs(pred - ref))),
                "is_interpolated": bool(is_interp),
            }
    out = out_dir or (ROOT / "runs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp1_regularized.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="实验一：正则化网络")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None, help="结果目录（默认 runs/）")
    parser.add_argument("--threads", type=int, default=None, help="限制 torch 线程数（并行跑种子时用）")
    args = parser.parse_args()
    if not has_torch():
        print("本脚本需要 PyTorch（pip install torch）。当前环境：", describe_environment())
        return 2
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    results = run(seed=args.seed, epochs=args.epochs, width=args.width, depth=args.depth, out_dir=args.out_dir)
    for energy, info in results["energies"].items():
        tag = "插值" if info["is_interpolated"] else "训练"
        print(f"E = {energy:>4s}  [{tag}]  max|psi_model - psi_ref| = {info['max_abs_err']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())