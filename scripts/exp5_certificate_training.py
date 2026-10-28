"""实验五：证书同口径训练（离散残差 + 边界 + 归一化）。

动机：exp2 的残差项用随机配点 + autograd，而证书用固定网格 + 二阶差分 H_h。
网络可以在配点之间/边界附近留下高频噪声：autograd 残差不大，但 H_h 残差
被 1/h² 放大，Temple 区间很宽。本脚本把**证书实际使用的离散残差**直接
放进训练目标，检验证书能否变紧。

这不是主实验的唯一做法；它是“证书驱动训练/早停”的最小可行原型。

支持三种势（``--potential``）：谐振子、无限深阱、有限深阱。激发态默认关掉
Rayleigh 项——固定 E 的残差已经把目标态钉死，而 Rayleigh 会把它拉回基态。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DOMAIN = (-6.0, 6.0)
N_GRID = 2001


def _problem(potential: str, state: int):
    """返回 (domain, V(x), 参考解, E_ref)。"""
    from nnschrodinger.physics.spectrum import (
        finite_square_well,
        harmonic_oscillator,
        infinite_square_well,
    )

    if potential == "harmonic":
        return (-6.0, 6.0), (lambda t: 0.5 * t**2), harmonic_oscillator(state), state
    if potential == "infinite_well":
        return (0.0, 1.0), (lambda t: np.zeros_like(t)), infinite_square_well(state + 1), state
    if potential == "finite_well":
        ref = finite_square_well(depth=50.0, half_width=1.0, n_states=state + 1)
        return (-8.0, 8.0), (lambda t: np.where(np.abs(t) < 1.0, -50.0, 0.0)), ref, state
    raise ValueError(f"未知势: {potential!r}")


def run(
    state: int = 0,
    epochs: int = 3000,
    width: int = 128,
    depth: int = 4,
    seed: int = 0,
    lr: float = 1e-3,
    learned_energy: bool = False,
    rayleigh_energy: bool = False,
    use_energy: bool = True,
    potential: str = "harmonic",
    deflate_from: Path | list[Path] | None = None,
    w_residual: float = 1.0,
    w_normalization: float = 100.0,
    w_boundary: float = 100.0,
    w_energy: float = 10.0,
    out_dir: Path | None = None,
) -> dict:
    import torch

    from nnschrodinger.solver.certify import certify_eigenpair
    from nnschrodinger.solver.evaluate import evaluate_model
    from nnschrodinger.solver.losses import discrete_dirichlet_terms
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer, evaluate_on_grid

    if learned_energy and rayleigh_energy:
        raise ValueError("--learned-energy 与 --rayleigh-energy 只能二选一")

    torch.manual_seed(seed)
    net = build_mlp(in_dim=1, width=width, depth=depth, fourier=True, seed=seed)
    domain, potential_fn, ref, _ = _problem(potential, state)
    if learned_energy:
        net.residual_energy = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    x_np = np.linspace(domain[0], domain[1], N_GRID)
    h = float(x_np[1] - x_np[0])
    v_np = np.asarray(potential_fn(x_np), dtype=float)
    x = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float64)
    v = torch.tensor(v_np.reshape(-1, 1), dtype=torch.float64)

    # 学出来的低能态（不是解析参考态）：用于激发态的硬正交投影。
    # 这是 C 线的原型——deflation 的参考来自同一套训练，不读真解。
    # 支持多个参考态（链式 deflation）：求 n=2 时同时投掉训练得到的
    # ψ₀ 与 ψ₁，否则单个参考投不干净，优化器可能又落回 ψ₁。
    if deflate_from is None:
        deflate_paths: list[Path] = []
    elif isinstance(deflate_from, (str, Path)):
        deflate_paths = [Path(deflate_from)]
    else:
        deflate_paths = [Path(p) for p in deflate_from]

    phi_np_list: list[np.ndarray] = []
    phi_t_list: list[torch.Tensor] = []
    for path in deflate_paths:
        data = np.load(path)
        x_phi = np.asarray(data["x"], dtype=float)
        if x_phi.shape != x_np.shape or not np.allclose(x_phi, x_np):
            raise ValueError(f"{path} 的网格与当前网格不一致")
        phi = np.asarray(data["psi"], dtype=float).reshape(-1).copy()
        phi[0] = phi[-1] = 0.0
        norm = float(np.sqrt(h * np.sum(phi[1:-1] ** 2)))
        if norm < 1e-12:
            raise ValueError(f"{path} 的波函数范数为零")
        phi /= norm
        phi_np_list.append(phi)
        phi_t_list.append(torch.tensor(phi.reshape(-1, 1), dtype=torch.float64))

    def project(psi: torch.Tensor) -> torch.Tensor:
        """把网络输出投到 span{φ_j} 的正交补（逐个 Gram–Schmidt）。"""
        if not phi_t_list:
            return psi
        p = psi.reshape(-1)
        interior = p[1:-1]
        for phi_t in phi_t_list:
            phi_i = phi_t.reshape(-1)[1:-1]
            coefficient = h * torch.sum(phi_i * interior)
            interior = interior - coefficient * phi_i
        out = p.clone()
        out[1:-1] = interior
        return out.reshape(psi.shape)

    def rayleigh_quotient(psi: torch.Tensor) -> torch.Tensor:
        """当前 ψ 的离散 Rayleigh 商（Dirichlet 延拓，与能量项同口径）。

        残差 ``H_hψ − Eψ`` 对 E 的最优选择就是 η=⟨ψ,H_hψ⟩/⟨ψ,ψ⟩：它使
        ``‖H_hψ − Eψ‖`` 最小。把它直接代进残差有两个好处：

        * 不需要给 E 单独调学习率，也不存在“E 停在 −9.8、真值 −49”的
          不收敛（工程记录 21）；
        * 训练目标与证书口径完全一致——证书用的也是这个 η。

        ``detach`` 是安全的：在极值点 η 对 ψ 的一阶导数为零，去掉的只是
        二阶小量；保留它反而会让 E 与网络参数争梯度。
        """
        with torch.no_grad():
            p = psi.reshape(-1)
            p_i = p[1:-1]
            # 必须先 reshape 成 (N-2,)：v 是 (N,1)，v[1:-1]*p_i 会广播成
            # (N-2,N-2) 的矩阵，把 Rayleigh 商算成 1e5（冒烟测试抓到）。
            v_i = v.reshape(-1)[1:-1]
            zeros = p.new_zeros(1)
            p_b = torch.cat((zeros, p_i, zeros))
            lap = (p_b[2:] - 2.0 * p_i + p_b[:-2]) / h**2
            h_psi = -0.5 * lap + v_i * p_i
            num = h * torch.sum(p_i * h_psi)
            den = h * torch.sum(p_i**2)
            return num / den.clamp_min(1e-30)

    def energy_value(psi: torch.Tensor):
        if rayleigh_energy:
            return rayleigh_quotient(psi)
        return net.residual_energy if learned_energy else float(ref.energy)

    def terms(psi: torch.Tensor) -> dict[str, torch.Tensor]:
        # 端点必须先置零再差分：否则能量项（Rayleigh 商）没有下界，
        # 训练会靠放大端点把损失推到负无穷（工程记录 18）。
        psi = project(psi)
        return discrete_dirichlet_terms(
            psi, v, h, energy_value(psi), use_energy=use_energy
        )

    weights = {
        "residual": w_residual,
        "normalization": w_normalization,
        "boundary": w_boundary,
        "energy": w_energy,
    }
    config = TrainConfig(
        mode="physics",
        epochs=epochs,
        lr=lr,
        n_collocation=N_GRID,
        log_every=max(1, epochs // 10),
        validation_every=max(1, epochs // 20),
        patience=12,
        term_weights=weights,
    )
    def validation_loss() -> float:
        # 早停只看“误差型”项（残差/归一化/边界）；energy 是变分目标，
        # 不是误差，混进来会让早停偏向“能量低但残差大”的 iterate。
        with torch.no_grad():
            values = terms(net(x))
            return float(
                weights["residual"] * values["residual"]
                + weights["normalization"] * values["normalization"]
                + weights["boundary"] * values["boundary"]
            )

    history = Trainer(net, lambda _epoch, _trainer: {"terms": terms(net(x))}, config).fit(
        validation=validation_loss
    )

    psi = np.asarray(evaluate_on_grid(net, x_np), dtype=float).reshape(-1)
    deflate_overlap = None
    deflate_overlaps: list[float] | None = None
    if phi_np_list:
        # 物理态 = 投影后的网络输出；证书必须认证它，而不是未投影的 ψ。
        psi = psi.copy()
        for phi in phi_np_list:
            coefficient = h * float(np.sum(phi[1:-1] * psi[1:-1]))
            psi[1:-1] -= coefficient * phi[1:-1]
        # 投影后再取一次内积：浮点意义下应为 0，用来验证投影确实做干净了。
        deflate_overlaps = [h * float(np.sum(phi[1:-1] * psi[1:-1]))
                            for phi in phi_np_list]
        deflate_overlap = max(abs(v) for v in deflate_overlaps)
    metrics = evaluate_model(psi, x_np, ref.psi, ref.energy, potential_fn).to_dict()
    cert = certify_eigenpair(psi, x_np, v_np, isolate=True)
    cert_dict = cert.to_dict()
    cert_dict["width"] = cert.width
    # 报告实际进入残差的 E，便于审计“训练目标是否自洽”（工程记录 21）：
    # rayleigh 模式下它就是最终 ψ 的 Rayleigh 商；learned 模式是自由标量；
    # reference 模式是参考能量（泄漏）。
    if rayleigh_energy:
        energy_used = float(rayleigh_quotient(torch.tensor(
            np.asarray(psi, dtype=float).reshape(-1, 1), dtype=torch.float64)))
    elif learned_energy:
        energy_used = float(net.residual_energy.detach())
    else:
        energy_used = float(ref.energy)
    result = {
        "potential": potential,
        "state": state,
        "seed": seed,
        "epochs": epochs,
        "width": width,
        "depth": depth,
        "learned_energy": bool(learned_energy),
        "rayleigh_energy": bool(rayleigh_energy),
        "deflate_from": ([str(p) for p in deflate_paths] or None),
        "deflate_overlap": deflate_overlap,
        "deflate_overlaps": deflate_overlaps,
        # 默认（learned_energy=False 且 rayleigh_energy=False）时**所有**态的
        # residual 都用参考能量，不只是 state>0；这和 exp2 的
        # residual_energy=reference 一样是泄漏。要得到真正的无监督样本必须
        # 传 --rayleigh-energy（推荐）或 --learned-energy。写进 JSON，
        # 证书报告才能按泄漏类型分组。
        "residual_energy": ("rayleigh" if rayleigh_energy
                            else "learned" if learned_energy else "reference"),
        "use_energy": bool(use_energy),
        "weights": weights,
        "learned_energy_value": (
            float(net.residual_energy.detach()) if learned_energy else None
        ),
        "residual_energy_value": energy_used,
        "metrics": metrics,
        "certificate": cert_dict,
        "history": history.to_dict(),
    }
    out = out_dir or (ROOT / "runs" / "certificate_training")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "wavefunction.npz",
        x=x_np,
        psi=np.asarray(psi, dtype=float),
        potential=v_np,
    )
    (out / "exp5_certificate_training.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="实验五：证书同口径训练")
    parser.add_argument(
        "--potential", choices=("harmonic", "infinite_well", "finite_well"), default="harmonic"
    )
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--learned-energy", action="store_true")
    parser.add_argument("--rayleigh-energy", action="store_true",
                        help="残差里的 E 用当前 ψ 的 Rayleigh 商（推荐；无泄漏）")
    parser.add_argument("--deflate-from", type=Path, action="append", default=None,
                        help="wavefunction.npz；用它学出来的 ψ 做硬正交，求激发态。"
                             "可重复给出（链式 deflation：求 n=2 时同时投掉 ψ₀ 与 ψ₁）")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(1)
    result = run(
        state=args.state,
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        seed=args.seed,
        lr=args.lr,
        learned_energy=args.learned_energy,
        rayleigh_energy=args.rayleigh_energy,
        # 有 deflation 时能量项必须在正交补里起作用（它选出补空间的最低态）；
        # 没有 deflation 的激发态仍应显式传 --no-energy，否则 Rayleigh 会拉回基态。
        use_energy=not args.no_energy,
        potential=args.potential,
        deflate_from=args.deflate_from,
        out_dir=args.out_dir,
    )
    cert = result["certificate"]
    print(
        f"[exp5] {result['potential']} n={result['state']} "
        f"E={result['metrics']['energy_model']:.6f} "
        f"E_used({result['residual_energy']})={result['residual_energy_value']:.6f} "
        f"overlap={result['metrics']['overlap']:.6f} "
        f"cert_valid={cert['valid']} width={cert['width']} "
        f"angle={cert['eigenfunction_angle_bound']} "
        f"residual={cert['residual_norm']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
