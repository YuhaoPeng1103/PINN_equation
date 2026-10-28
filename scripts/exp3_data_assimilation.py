"""实验三：数据同化（物理残差 + 少量观测）—— 最接近真实量子实验的设定。

论文场景：HHL 论文的量子线路实验里，“正则网络”的训练数据来自实际设备；
任何真实设备都有噪声与查询次数上限，因此“用少量测量 + 物理约束”训练，
比“只靠大量标签”更现实。

本实验模拟：
  * 只在 x 的若干个随机点上获得 ψ 的观测（含高斯噪声）；
  * 同时用物理残差 + 归一化 + 边界约束补足全场信息；
  * 对比三种配置：(a) 纯 PINN（无观测）、(b) 少量观测（k=16）、(c) 更多观测（k=64）。
    预期：(b)(c) 在观测覆盖率提高时，波形 overlap 与能量误差同时改善。

依赖 PyTorch；当前沙箱未安装 torch，脚本会打印环境信息后退出（退出码 2）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.backends import describe_environment, has_torch  # noqa: E402

DOMAIN = (0.0, 1.0)
N_COLLOCATION = 2000


def run(n_obs: int, noise: float = 0.01, epochs: int = 4000, seed: int = 0, out_dir: Path | None = None) -> dict:
    import torch

    from nnschrodinger.physics.spectrum import infinite_square_well
    from nnschrodinger.solver.evaluate import evaluate_model
    from nnschrodinger.solver.losses import LossWeights, Losses, total_loss
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer, evaluate_on_grid

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ref = infinite_square_well(1)  # 无限深阱基态 E = pi^2/2
    potential = lambda t: torch.zeros_like(t)  # noqa: E731
    net = build_mlp(in_dim=1, width=128, depth=4, fourier=True, seed=seed)
    weights = LossWeights(residual=1.0, normalization=100.0, boundary=100.0, supervised=10.0)
    # np.linspace 的 2000 个点间距是 1/1999，不是 1/2000
    losses = Losses(weights, dx=(DOMAIN[1] - DOMAIN[0]) / (N_COLLOCATION - 1))
    config = TrainConfig(mode="data_assimilation", epochs=epochs, lr=1e-3, n_collocation=N_COLLOCATION, log_every=500)

    obs_x = np.sort(rng.uniform(*DOMAIN, n_obs)) if n_obs else np.empty(0)
    obs_y = ref.psi(obs_x) + (rng.normal(0.0, noise, n_obs) if n_obs else np.empty(0))

    def step(epoch: int, trainer: Trainer) -> dict:
        x = torch.tensor(rng.uniform(*DOMAIN, N_COLLOCATION).reshape(-1, 1), dtype=torch.float64, requires_grad=True)
        terms = {
            "residual": losses.residual(net, x, ref.energy, potential),
            "normalization": losses.normalization(net, x),
            "boundary": losses.boundary(net, torch.tensor([[0.0], [1.0]], dtype=torch.float64)),
        }
        if n_obs:
            terms["supervised"] = losses.supervised(
                net(torch.tensor(obs_x.reshape(-1, 1), dtype=torch.float64)),
                torch.tensor(obs_y.reshape(-1, 1), dtype=torch.float64),
            )
        return {"loss": total_loss(losses, **terms)}

    history = Trainer(net, step, config).fit()
    grid = np.linspace(*DOMAIN, 2001)
    psi = evaluate_on_grid(net, grid)
    metrics = evaluate_model(psi, grid, ref.psi, ref.energy, lambda t: np.zeros_like(t)).to_dict()
    result = {"environment": describe_environment(), "n_obs": n_obs, "noise": noise, "seed": seed,
              "epochs": epochs, "metrics": metrics,
              "final_loss": history.loss[-1] if history.loss else None}
    out = out_dir or (ROOT / "runs")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"exp3_data_assimilation_k{n_obs}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="实验三：数据同化")
    parser.add_argument("--n-obs", type=int, nargs="+", default=[0, 16, 64])
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None, help="结果目录（默认 runs/）")
    parser.add_argument("--threads", type=int, default=None, help="限制 torch 线程数（并行跑种子时用）")
    args = parser.parse_args()
    if not has_torch():
        print("本脚本需要 PyTorch（pip install torch）。当前环境：", describe_environment())
        return 2
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    for n_obs in args.n_obs:
        result = run(n_obs=n_obs, noise=args.noise, epochs=args.epochs, seed=args.seed,
                     out_dir=args.out_dir)
        m = result["metrics"]
        print(f"k={n_obs:>3d}  overlap={m['overlap']:.4f}  E_err={m['energy_abs_err']:.3e}  inf={m['inf_norm']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())