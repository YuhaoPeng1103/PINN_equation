"""实验二：非正则化（无监督 PINN）求解本征态 —— 与实验一对照。

实验一的网络有 ψ_target 作为监督信号；本实验把标签完全去掉，只用物理约束：

    L = w_r·mean[(−½ψ″ + Vψ − Eψ)²] + w_e·Rayleigh 商约束 + w_n·(∫|ψ|²−1)² + w_b·ψ(∂Ω)²

对照点（论文的核心结论，也是复现时要特别小心的地方）：
  * 正则化网络在给定能量下可直接调用，泛化平滑；非正则化 PINN 训练更慢、
    对初始化敏感，且多个激发态同时训练时需要正交化约束；
  * **PINN 的一个常见误判**：Rayleigh 商很接近 E，但波形与真解正交
    （与真解正交的污染分量对 Rayleigh 商是一阶不敏感的）。因此评测必须同时看
    overlap 与 inf_norm，不能只看能量。

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

DOMAIN = (-6.0, 6.0)
N_COLLOCATION = 2500


def run(state: int = 0, epochs: int = 5000, width: int = 128, depth: int = 4, seed: int = 0,
        grad_balance: bool = False, orthogonalize: bool = False, use_energy: bool = True,
        energy_sampling: str = "grid", lr: float = 1e-3, energy_weight: float = 10.0,
        residual_energy: str = "reference", residual_energy_init: float = 1.0,
        out_dir: Path | None = None, save_wavefunction: bool = False) -> dict:
    """训练一个 PINN 并评测。

    use_energy=False      复现"仅残差损失"对照（无 Rayleigh 商）；
    energy_sampling="grid" 能量项用固定等距网格（推荐，无采样噪声）；
    energy_sampling="random" 沿用随机配点估计 Rayleigh 商（复现 1.9% 噪声的对照）；
    residual_energy="reference" 残差里的 E 用真实本征值（旧配置，等价于告诉网络答案）；
    residual_energy="learned"   残差里的 E 是可学习标量（真正的无监督本征值问题）。
    正交约束（orthogonalize）依赖固定网格上的投影基，不能与随机采样同用。
    """
    import torch

    from nnschrodinger.physics.spectrum import harmonic_oscillator
    from nnschrodinger.solver.evaluate import evaluate_model
    from nnschrodinger.solver.losses import LossWeights, Losses, total_loss
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer, evaluate_on_grid

    if energy_sampling not in ("grid", "random"):
        raise ValueError(f"energy_sampling 必须是 'grid' 或 'random'，收到 {energy_sampling!r}")
    if orthogonalize and energy_sampling != "grid":
        raise ValueError("正交约束的投影基定义在固定网格上，不能与随机能量采样同用")
    if residual_energy not in ("reference", "learned"):
        raise ValueError(f"residual_energy 必须是 'reference' 或 'learned'，收到 {residual_energy!r}")
    if residual_energy == "learned" and orthogonalize:
        raise ValueError("激发态 deflation 目前固定用参考能量，不支持学习能量")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ref = harmonic_oscillator(state)
    # x → ψ 的无监督网络：只输入 x（能量作为常数损失项，不进入网络）
    net = build_mlp(in_dim=1, width=width, depth=depth, fourier=True, seed=seed)
    if residual_energy == "learned":
        # 把 E 注册成网络模块的参数，Trainer 的 Adam 会自动优化它。
        # 这才是“残差损失能否定基态”的正确对照：E 不被提前告知，
        # 任一 (ψ_n, E_n) 都让残差为零，解在本征态之间是退化的。
        # 初值不要用 ref.energy（那等于把答案塞给网络），默认取 1.0。
        net.residual_energy = torch.nn.Parameter(
            torch.tensor(float(residual_energy_init), dtype=torch.float64)
        )
    weights = LossWeights(residual=1.0, energy=energy_weight, normalization=100.0, boundary=0.0)
    # dx 必须与固定网格的实际间距一致：np.linspace(-6, 6, N) 的间隔是 12/(N-1)，
    # 用 12/N 会让积分（归一化、Rayleigh 商）带 4e-4 的相对偏差。
    losses = Losses(weights, dx=(DOMAIN[1] - DOMAIN[0]) / (N_COLLOCATION - 1))
    potential = lambda t: 0.5 * t**2  # noqa: E731
    config = TrainConfig(
        mode="physics",
        epochs=epochs,
        lr=lr,
        n_collocation=N_COLLOCATION,
        log_every=500,
        grad_balance=grad_balance,
        # 关键：step_fn 返回的是未加权分项，权重必须在这里显式传入。
        # 旧版本 Trainer 直接 sum(terms)，导致 normalization=100 实际按 1 生效，
        # 残差项可以靠 ψ→0 把损失降到 0（见 docs/engineering-log.md 第 13 条）。
        term_weights=weights.as_dict(),
    )

    # 能量项用**固定等距网格**：随机点会让 Rayleigh 商带 1.9% 的采样噪声
    # （见 docs/results.md），表现为能量曲线永远在抖动。
    energy_grid = np.linspace(DOMAIN[0], DOMAIN[1], N_COLLOCATION)

    # 求激发态时的 deflation：把网络输出投影到低阶态张成空间的**正交补**上。
    # 这是"变分法的下一阶"（Hylleraas–Undheim / 序列变分）的做法：
    #   基态：min_{ψ⊥空格} R(ψ)  -> n=0 即普通变分
    #   第 n 态：min_{ψ⊥{ψ_0..ψ_{n-1}}} R(ψ)
    # 这里用**解析参考态**构建投影（而非让网络互相正交），因为我们关心的是
    # "能量+正交约束能否稳定收敛到第 n 态"，用已知正交基可把正交性误差从变量中剔除。
    lower = [harmonic_oscillator(k) for k in range(state)] if orthogonalize else []
    basis = None
    if lower:
        # 基在固定网格上正交归一（|φ_k(x_i)| 表），再按格点索引做投影。
        # 注意投影只在 energy_grid 上成立，因此不能对任意点求值——
        # 需要求值的地方一律使用同一张网格。
        gl = np.linspace(DOMAIN[0], DOMAIN[1], N_COLLOCATION)
        hl = float(gl[1] - gl[0])
        cols = []
        for s in lower:
            v = np.asarray(s.psi(gl), dtype=float)
            v = v / np.sqrt(float(np.sum(v**2)) * hl)
            cols.append(v)
        basis = torch.tensor(np.stack(cols, axis=1), dtype=torch.float64)  # (N, K)
        grid_steps = basis.shape[0]

    def project(psi):
        """ψ -= Σ_k (ψ·φ_k) φ_k（φ_k 为单位正交基，离散内积含 dx）。

        仅在 energy_grid 上定义：调用方必须传入与 basis 同形状的 (N, 1) 张量。
        """
        if basis is None:
            return psi
        coeff = basis.T @ psi * losses.dx
        return psi - basis @ coeff

    def residual_energy_value():
        """残差项里的 E：真实本征值（旧对照）或可学习标量（正确的无监督设定）。"""
        return net.residual_energy if residual_energy == "learned" else ref.energy

    def step(epoch: int, trainer: Trainer) -> dict:
        # 残差项通常用重采样的随机配点（避免在固定网格上过拟合）。
        # 但正交约束的 projector 是按 energy_grid 的**行索引**构造的矩阵，
        # 只能作用在同一张网格上；若喂随机点，投影会混合毫不相干的点。
        if orthogonalize:
            x_np = energy_grid
        else:
            x_np = rng.uniform(*DOMAIN, N_COLLOCATION)
        x = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float64)
        # 能量网格必须开启 requires_grad：Rayleigh 商要对 x 求 dψ/dx，
        # 输入若非叶子且不需要梯度，autograd 拿不到导数（曾导致能量算成 6.38 vs 1.5）
        if energy_sampling == "grid":
            xg_np = energy_grid
        else:
            # 对照配置：每步重采随机点。mean 就成了蒙特卡洛积分，
            # Rayleigh 商自带 1.9% 的估计噪声（见 docs/engineering-log.md 第 10 条）。
            xg_np = rng.uniform(*DOMAIN, N_COLLOCATION)
        xg = torch.tensor(xg_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)
        # 归一化项始终用固定网格：随机模式只改变 Rayleigh 商的估计方式，
        # 这样对照实验里只有一个变量（能量项的采样噪声）。
        x_norm = torch.tensor(energy_grid.reshape(-1, 1), dtype=torch.float64)

        # 关键：不能在模型外面先投影再求导（d/dx 会被错误地削掉一半，
        # 实测把 |dpsi| 从 1.062 削到 0.531，能量变成 1.25 vs 1.5）。
        # 正确做法：交给 energy_projected，它先求导、只对波函数本身做投影。
        if orthogonalize:
            energy, _ = losses.energy_projected(net, xg, potential, project)
        elif use_energy:
            energy, _ = losses.energy(net, xg, potential)
        else:
            energy = None
        # 无监督基态求解的标准做法：同时最小化 Rayleigh 商（变分下界）
        # 与残差。只用残差会收敛到“任意本征态”，而基态需要能量项把解推向最低能级。
        # 返回"分项"而不是加权和：由 Trainer 决定是否做梯度均衡
        e_res = residual_energy_value()
        if orthogonalize:
            residual = losses.residual_projected(net, x, e_res, potential, project)
            normalization = losses.normalization(project, x_norm)
        else:
            residual = losses.residual(net, x, e_res, potential)
            normalization = losses.normalization(net, x_norm)
        terms = {"residual": residual, "normalization": normalization}
        if energy is not None:
            terms["energy"] = energy
        info = {"terms": terms}
        if energy is not None:
            info["energy"] = float(energy.detach())
        elif residual_energy == "learned":
            info["energy"] = float(e_res.detach())
        return info

    # 正交化时投影只在 energy_grid 上成立，评估也必须用同一张网格
    grid_eval = np.linspace(DOMAIN[0], DOMAIN[1], N_COLLOCATION) if orthogonalize else np.linspace(*DOMAIN, 2001)

    # 验证/早停必须用**与训练目标同口径**的加权损失，而不是未归一化残差。
    # 只看残差时，ψ≈0 会让残差任意小——早停会回滚到训练初期的平凡解，
    # 归一化后就是杂波（实测 residual_only 的 E=8.6，见工程记录第 13 条）。
    val_grid = energy_grid
    val_x = torch.tensor(val_grid.reshape(-1, 1), dtype=torch.float64, requires_grad=True)

    def validation() -> float:
        if orthogonalize:
            residual = losses.residual_projected(net, val_x, residual_energy_value(), potential, project)
            normalization = losses.normalization(project, val_x)
            total = weights.residual * residual + weights.normalization * normalization
            if use_energy:
                energy, _ = losses.energy_projected(net, val_x, potential, project)
                total = total + weights.energy * energy
        else:
            residual = losses.residual(net, val_x, residual_energy_value(), potential)
            normalization = losses.normalization(net, val_x)
            total = weights.residual * residual + weights.normalization * normalization
            if use_energy:
                energy, _ = losses.energy(net, val_x, potential)
                total = total + weights.energy * energy
        return float(total.detach())

    history = Trainer(net, step, config).fit(validation=validation)
    psi = evaluate_on_grid(net, grid_eval)
    if orthogonalize:
        g = torch.tensor(grid_eval.reshape(-1, 1), dtype=torch.float64)
        with torch.no_grad():
            psi = project(net(g)).numpy().reshape(-1)
    metrics = evaluate_model(psi, grid_eval, ref.psi, ref.energy, potential).to_dict()
    result = {
        "environment": describe_environment(),
        "state": state,
        "seed": seed,
        "epochs": epochs,
        "width": width,
        "depth": depth,
        "grad_balance": grad_balance,
        "orthogonalize": orthogonalize,
        "use_energy": use_energy,
        "energy_sampling": energy_sampling,
        "lr": lr,
        "energy_weight": energy_weight,
        "residual_energy": residual_energy,
        "residual_energy_init": residual_energy_init if residual_energy == "learned" else None,
        "learned_energy": (float(net.residual_energy.detach()) if residual_energy == "learned" else None),
        "metrics": metrics,
        "history": history.to_dict(),
    }
    out = out_dir or (ROOT / "runs")
    out.mkdir(parents=True, exist_ok=True)
    if save_wavefunction:
        # 后验证书需要 ψ(x) 本身，而不仅是能量/overlap 指标。
        # 存成 npz 比塞进 JSON 小，也便于 certify 脚本直接读取。
        np.savez(
            out / "wavefunction.npz",
            x=np.asarray(grid_eval, dtype=float),
            psi=np.asarray(psi, dtype=float),
            potential=np.asarray(potential(grid_eval), dtype=float),
        )
    (out / "exp2_pinn.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="实验二：无监督 PINN")
    parser.add_argument("--state", type=int, default=0, help="本征态序号（0=基态）")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--grad-balance", action="store_true", help="启用梯度 RMS 均衡")
    parser.add_argument("--orthogonalize", action="store_true",
                        help="对低阶态做 deflation（求激发态时几乎必需）")
    parser.add_argument("--residual-only", action="store_true",
                        help="去掉 Rayleigh 商，只保留残差（复现对照表第一行）")
    parser.add_argument("--energy-sampling", choices=("grid", "random"), default="grid",
                        help="Rayleigh 商的采样方式：grid=固定等距网格（推荐），random=每步随机点")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率（敏感性扫描用）")
    parser.add_argument("--energy-weight", type=float, default=10.0, help="Rayleigh 商损失权重")
    parser.add_argument("--residual-energy", choices=("reference", "learned"), default="reference",
                        help="残差里的 E：reference=用真实本征值（旧对照），learned=可学习标量")
    parser.add_argument("--residual-energy-init", type=float, default=1.0,
                        help="learned 模式下 E 的初值（默认 1.0，刻意不用 ref.energy）")
    parser.add_argument("--out-dir", type=Path, default=None, help="结果目录（默认 runs/）")
    parser.add_argument("--threads", type=int, default=None, help="限制 torch 线程数（并行跑种子时用）")
    args = parser.parse_args()
    if not has_torch():
        print("本脚本需要 PyTorch（pip install torch）。当前环境：", describe_environment())
        return 2
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    result = run(state=args.state, epochs=args.epochs, width=args.width, depth=args.depth,
                 seed=args.seed, grad_balance=args.grad_balance, orthogonalize=args.orthogonalize,
                 use_energy=not args.residual_only, energy_sampling=args.energy_sampling,
                 lr=args.lr, energy_weight=args.energy_weight,
                 residual_energy=args.residual_energy,
                 residual_energy_init=args.residual_energy_init, out_dir=args.out_dir)
    for key, value in result["metrics"].items():
        print(f"{key:>16s} = {value:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())