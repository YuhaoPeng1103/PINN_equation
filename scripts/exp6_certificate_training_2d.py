"""实验六：二维证书同口径训练（原型）。

与实验五（一维）相同的思路：损失里用的是**证书实际使用的离散残差**
（五点差分 H_h、Dirichlet 边界、Rayleigh 商能量），训练完直接对网络
输出做二维后验证书。二维的意义在于：

* 一维的三对角 Sturm 计数换成稀疏 LU 的 Sylvester 惯性计数
  （``certify2d``），证书逻辑（Temple/夹角/区间验证）完全复用；
* 二维的简并是**结构性**的（各向同性谐振子/方阱的第一激发态是二重简并），
  证书必须拒绝认证，而不是给出一个看似很紧的假界；
* 非可分的平滑势（``anharmonic2d``）没有解析解，是真正的“无参考”场景。

三个势：
    harmonic2d    V = ½(x²+y²), [-3,3]², 解析 E₀ = 1（截断误差 ~1e-4）
    well2d        V = 0, [0,1]², 解析 E₀ = π²（DST 基下精确）
    anharmonic2d  V = ½(x²+y²) + ½x²y², [-3,3]², 无解析解
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _problem2d(name: str):
    """返回 (domain, V(x, y), E_analytic, ψ_analytic) —— 后两项可为 None。"""
    if name == "harmonic2d":
        domain = (-3.0, 3.0)
        potential = lambda x, y: 0.5 * (x**2 + y**2)
        energy = 1.0
        psi = lambda x, y: np.exp(-0.5 * (x**2 + y**2)) / math.sqrt(math.pi)
        return domain, potential, energy, psi
    if name == "well2d":
        domain = (0.0, 1.0)
        potential = lambda x, y: np.zeros_like(x + y)
        energy = math.pi**2
        psi = lambda x, y: 2.0 * np.sin(math.pi * x) * np.sin(math.pi * y)
        return domain, potential, energy, psi
    if name == "anharmonic2d":
        domain = (-3.0, 3.0)
        potential = lambda x, y: 0.5 * (x**2 + y**2) + 0.5 * (x * y) ** 2
        return domain, potential, None, None
    raise ValueError(f"未知二维势: {name!r}")


def run(
    potential: str = "harmonic2d",
    epochs: int = 4000,
    width: int = 64,
    depth: int = 4,
    seed: int = 0,
    lr: float = 1e-3,
    n_grid: int = 81,
    fourier: bool = False,
    fourier_scale: float = 2.5,
    use_energy: bool = True,
    deflate_from: Path | list[Path] | None = None,
    w_residual: float = 1.0,
    w_normalization: float = 100.0,
    w_boundary: float = 100.0,
    w_energy: float = 10.0,
    out_dir: Path | None = None,
) -> dict:
    import torch

    from nnschrodinger.solver.certify2d import (
        Grid2D,
        approximate_spectrum_2d,
        certificate_dict,
        certify_eigenpair_2d,
        discrete_eigenvalue_2d,
    )
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer

    torch.manual_seed(seed)
    # 二维默认**不用**傅里叶特征：随机方向的高频投影在 2D 比 1D 更难
    # 被优化器组织起来（实测 1500 epoch 下 E=4.8 vs 1.01，见工程记录 24）。
    net = build_mlp(in_dim=2, width=width, depth=depth, fourier=fourier,
                    fourier_scale=fourier_scale, seed=seed)
    domain, potential_fn, energy_analytic, psi_analytic = _problem2d(potential)

    x_np = np.linspace(domain[0], domain[1], n_grid)
    y_np = np.linspace(domain[0], domain[1], n_grid)
    X, Y = np.meshgrid(x_np, y_np, indexing="ij")
    v_np = np.asarray(potential_fn(X, Y), dtype=float)
    grid = Grid2D(x_np, y_np, v_np)
    hx, hy = grid.hx, grid.hy

    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
    v_i = torch.tensor(v_np[1:-1, 1:-1], dtype=torch.float64)

    # 硬边界约束：把网络输出乘以在四条边上为零的掩码。二维里边界只占
    # O(1/N) 的点，软惩罚很难把它压到机器精度；实测（N=41，3000 epoch）
    # 软惩罚的方阱 E=10.6（真值 9.87）、残差 5.6，硬约束的 E=9.8645、
    # 残差 1.3e-2。掩码是**给定的边界条件**，不是参考解，不构成泄漏。
    a, b = domain
    mask_np = (X - a) * (b - X) * (Y - a) * (b - Y)
    mask = torch.tensor(mask_np, dtype=torch.float64)

    # 硬正交 deflation：把网络输出投到已认证的低能态的正交补，用来求激发态
    # （二维第一激发态是二重简并的，见实验八）。参考态来自**同一套训练**
    # 保存的波函数，不读真解，因此不构成泄漏。
    if deflate_from is None:
        deflate_paths: list[Path] = []
    elif isinstance(deflate_from, (str, Path)):
        deflate_paths = [Path(deflate_from)]
    else:
        deflate_paths = [Path(p) for p in deflate_from]
    phi_flat: list[torch.Tensor] = []
    phi_np_flat: list[np.ndarray] = []
    for path in deflate_paths:
        data = np.load(path)
        if not (np.allclose(data["x"], x_np) and np.allclose(data["y"], y_np)):
            raise ValueError(f"{path} 的网格与当前网格不一致")
        phi = np.asarray(data["psi"], dtype=float).copy()
        phi[0, :] = phi[-1, :] = 0.0
        phi[:, 0] = phi[:, -1] = 0.0
        norm = float(np.sqrt(grid.cell * np.sum(phi[1:-1, 1:-1] ** 2)))
        if norm < 1e-12:
            raise ValueError(f"{path} 的波函数范数为零")
        phi /= norm
        phi_flat.append(
            torch.tensor(phi[1:-1, 1:-1].reshape(-1), dtype=torch.float64)
        )
        phi_np_flat.append(phi[1:-1, 1:-1].reshape(-1).copy())

    def project(u: torch.Tensor) -> torch.Tensor:
        """逐个 Gram–Schmidt，把 u（内部节点）投到 span{φ_j} 的正交补。"""
        if not phi_flat:
            return u
        p = u.reshape(-1)
        for phi in phi_flat:
            coefficient = grid.cell * torch.sum(phi * p)
            p = p - coefficient * phi
        return p.reshape(u.shape)

    def forward_grid() -> torch.Tensor:
        return net(xy).reshape(n_grid, n_grid) * mask

    def laplacian(p: torch.Tensor) -> torch.Tensor:
        """五点 Laplacian（p 为整张网格，含边界）。"""
        interior = p[1:-1, 1:-1]
        lap = (
            (p[2:, 1:-1] - 2.0 * interior + p[:-2, 1:-1]) / hx**2
            + (p[1:-1, 2:] - 2.0 * interior + p[1:-1, :-2]) / hy**2
        )
        return lap

    def terms(p: torch.Tensor) -> dict[str, torch.Tensor]:
        """尺度无关的损失：所有项都作用在 u = ψ/‖ψ‖ 上。

        直接用未归一化的 ψ 会让 ψ≡0 成为一个**驻点**（残差项与归一化项
        的梯度都含因子 ψ），二维训练实测会塌缩到零。把 u 显式归一化后
        损失与尺度无关，零解被排除；输出的整体幅度在保存前统一归一。
        """
        interior = p[1:-1, 1:-1]
        norm2 = grid.cell * torch.sum(interior**2)
        scale = torch.sqrt(norm2.clamp_min(1e-30))
        u = interior / scale
        if phi_flat:
            u = project(u)
            u = u / torch.sqrt(
                (grid.cell * torch.sum(u**2)).clamp_min(1e-30)
            )

        padded = torch.nn.functional.pad(u, (1, 1, 1, 1))
        lap_b = (
            (padded[2:, 1:-1] - 2.0 * u + padded[:-2, 1:-1]) / hx**2
            + (padded[1:-1, 2:] - 2.0 * u + padded[1:-1, :-2]) / hy**2
        )
        h_psi_b = -0.5 * lap_b + v_i * u
        eta = grid.cell * torch.sum(u * h_psi_b)

        # 有 deflation 时残差必须作用在**投影后**的场上，否则被投掉的分量
        # 仍然通过 H_h 返回；无 deflation 时保持原路径（含 p 的边界值）。
        if phi_flat:
            h_psi = h_psi_b
        else:
            h_psi = (-0.5 * laplacian(p) + v_i * interior) / scale
        r = h_psi - eta.detach() * u
        boundary = (
            (p[0, :] ** 2).sum()
            + (p[-1, :] ** 2).sum()
            + (p[:, 0] ** 2).sum()
            + (p[:, -1] ** 2).sum()
        ) / norm2.clamp_min(1e-30)
        out = {"residual": torch.mean(r**2), "boundary": boundary}
        if use_energy:
            out["energy"] = eta
        return out

    def evaluate_terms() -> dict[str, torch.Tensor]:
        return terms(forward_grid())

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
        n_collocation=n_grid * n_grid,
        log_every=max(1, epochs // 10),
        validation_every=max(1, epochs // 20),
        patience=12,
        term_weights=weights,
    )

    def validation_loss() -> float:
        with torch.no_grad():
            values = evaluate_terms()
            return float(
                weights["residual"] * values["residual"]
                + weights["boundary"] * values["boundary"]
            )

    history = Trainer(
        net, lambda _e, _t: {"terms": evaluate_terms()}, config
    ).fit(validation=validation_loss)

    with torch.no_grad():
        psi = forward_grid().cpu().numpy()
    # 损失尺度无关，保存前把内部节点归一化到 1（证书内部还会再归一化）。
    interior = psi[1:-1, 1:-1]
    psi = psi / float(np.sqrt(grid.cell * np.sum(interior**2)))

    # 有 deflation 时保存**投影后**的场：硬正交约束作用在 (I−P₀)ψ 上，
    # 保存原始输出会让证书去认证一个未被约束的对象（工程记录 30）。
    psi_raw = None
    if phi_np_flat:
        psi_raw = psi
        u = psi[1:-1, 1:-1].reshape(-1).copy()
        for phi in phi_np_flat:
            u = u - grid.cell * float(np.sum(phi * u)) * phi
        u = u / float(np.sqrt(grid.cell * np.sum(u**2)))
        psi = np.zeros_like(psi_raw)
        psi[1:-1, 1:-1] = u.reshape(n_grid - 2, n_grid - 2)

    # 近似谱（eigsh）只用于提议区间/报告参考值；证书本身用惯性计数验证。
    approx = approximate_spectrum_2d(grid, 6)
    cert = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
    cert_d = certificate_dict(cert)

    # 计数二分给出的独立参考值（比 eigsh 慢，但由 Sylvester 计数验证）。
    reference_index = len(phi_flat)
    try:
        discrete_ref = discrete_eigenvalue_2d(grid, reference_index, approx=approx)
    except ValueError:
        discrete_ref = float(approx[reference_index])

    metrics: dict = {
        "energy_model": cert.energy,
        "energy_analytic": energy_analytic,
        "energy_error_vs_analytic": (
            abs(cert.energy - energy_analytic) if energy_analytic is not None else None
        ),
        "energy_discrete_ref": discrete_ref,
        "energy_error_vs_discrete": abs(cert.energy - discrete_ref),
        "overlap": None,
        "relative_l2": None,
        "amplitude": float(np.max(np.abs(psi))),
    }
    if psi_analytic is not None and not phi_flat:
        ref = np.asarray(psi_analytic(X, Y), dtype=float)
        ref = ref / np.sqrt(grid.cell * np.sum(ref[1:-1, 1:-1] ** 2))
        p = psi.copy()
        p[0, :] = p[-1, :] = 0.0
        p[:, 0] = p[:, -1] = 0.0
        p = p / np.sqrt(grid.cell * np.sum(p[1:-1, 1:-1] ** 2))
        overlap = float(grid.cell * np.sum(p[1:-1, 1:-1] * ref[1:-1, 1:-1]))
        metrics["overlap"] = abs(overlap)
        metrics["relative_l2"] = float(
            np.sqrt(grid.cell * np.sum((p[1:-1, 1:-1] - ref[1:-1, 1:-1]) ** 2))
        )

    result = {
        "experiment": "exp6_certificate_training_2d",
        "potential": potential,
        "seed": seed,
        "n_grid": int(n_grid),
        "epochs": int(epochs),
        "width": int(width),
        "depth": int(depth),
        "fourier": bool(fourier),
        "fourier_scale": float(fourier_scale),
        "residual_energy": "rayleigh",
        "use_energy": bool(use_energy),
        "deflate_from": [str(p) for p in deflate_paths] or None,
        "reference_index": int(len(phi_flat)),
        "weights": weights,
        "hx": hx,
        "hy": hy,
        "domain": list(domain),
        "metrics": metrics,
        "certificate": cert_d,
        "approx_spectrum": [float(v) for v in approx],
        "history": history.to_dict(),
    }

    out = out_dir or (ROOT / "runs" / "certificates2d")
    out.mkdir(parents=True, exist_ok=True)
    if psi_raw is None:
        np.savez(out / "wavefunction2d.npz", x=x_np, y=y_np, psi=psi, potential=v_np)
    else:
        np.savez(out / "wavefunction2d.npz", x=x_np, y=y_np, psi=psi,
                 psi_raw=psi_raw, potential=v_np)
    (out / "exp6_certificate_training_2d.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="实验六：二维证书同口径训练（原型）")
    parser.add_argument(
        "--potential", choices=("harmonic2d", "well2d", "anharmonic2d"), default="harmonic2d"
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--fourier", action="store_true",
                        help="启用随机傅里叶特征（2D 默认关闭，见工程记录 24）")
    parser.add_argument("--fourier-scale", type=float, default=2.5)
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--deflate-from", type=Path, action="append", default=None,
                        help="已认证低能态的 wavefunction2d.npz；可重复给出（链式 deflation）")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    result = run(
        potential=args.potential,
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        seed=args.seed,
        lr=args.lr,
        n_grid=args.n_grid,
        fourier=args.fourier,
        fourier_scale=args.fourier_scale,
        use_energy=not args.no_energy,
        deflate_from=args.deflate_from,
        out_dir=args.out_dir,
    )
    cert = result["certificate"]
    print(
        f"[exp6] {result['potential']} seed={result['seed']} "
        f"E={cert['energy']:.6f} "
        f"err_discrete={result['metrics']['energy_error_vs_discrete']:.2e} "
        f"overlap={result['metrics']['overlap']} "
        f"cert_valid={cert['valid']} width={cert['width']} "
        f"angle={cert['eigenfunction_angle_bound']} "
        f"residual={cert['residual_norm']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
