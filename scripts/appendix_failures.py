"""复现附录 A 的两个训练失败模式（软边界惩罚 / 未归一化残差）。

正文附录 ``paper/sections/08_appendix.tex`` 把这两种失败写成"可复现"，
本脚本就是那份记录：同一网络、同一网格、同一势（方阱 well2d，二维），
只改动一个设计选择。

两个变体：

``soft_boundary``
    不乘掩码（``M ≡ 1``），改为在损失里加边界惩罚（权重 100）。其余与
    实验六相同：残差项作用在归一化后的场上。报告最终能量、残差、以及
    边界节点上的最大幅度——软惩罚压不下去的正是这个量。

``unnormalized``
    乘掩码，但残差项作用在**未归一化**的 ψ 上：``‖H_hψ − ηψ‖²``。
    该损失在 ψ → cψ 下按 c² 缩放（η 是 Rayleigh 商，尺度不变），
    因此 ψ ≡ 0 是一个全局极小点。报告内部节点范数平方 ``‖ψ‖²``
    随时间的变化、最终能量与形状重叠度。

两个变体都保存 ``wavefunction2d.npz`` 与 JSON；输出目录
``runs/appendix_failures/<variant>/``。
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


def _well2d():
    """方阱 [0,1]²：V=0，解析 E=π²，ψ=2 sin(πx) sin(πy)。"""
    domain = (0.0, 1.0)
    potential = lambda x, y: np.zeros_like(x + y)
    energy = math.pi**2
    psi = lambda x, y: 2.0 * np.sin(math.pi * x) * np.sin(math.pi * y)
    return domain, potential, energy, psi


def run(
    variant: str = "soft_boundary",
    epochs: int = 3000,
    width: int = 64,
    depth: int = 4,
    seed: int = 0,
    lr: float = 1e-3,
    n_grid: int = 41,
    w_boundary: float = 100.0,
    w_energy: float = 10.0,
    out_dir: Path | None = None,
) -> dict:
    import torch

    from nnschrodinger.solver.certify2d import (
        Grid2D,
        approximate_spectrum_2d,
        certify_eigenpair_2d,
        discrete_eigenvalue_2d,
    )
    from nnschrodinger.solver.networks import build_mlp
    from nnschrodinger.solver.train import TrainConfig, Trainer

    if variant not in ("soft_boundary", "unnormalized"):
        raise ValueError(f"未知变体: {variant!r}")

    torch.manual_seed(seed)
    net = build_mlp(in_dim=2, width=width, depth=depth, fourier=False, seed=seed)
    domain, potential_fn, energy_analytic, psi_analytic = _well2d()

    x_np = np.linspace(domain[0], domain[1], n_grid)
    y_np = np.linspace(domain[0], domain[1], n_grid)
    X, Y = np.meshgrid(x_np, y_np, indexing="ij")
    v_np = np.asarray(potential_fn(X, Y), dtype=float)
    grid = Grid2D(x_np, y_np, v_np)
    hx, hy = grid.hx, grid.hy

    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
    v_i = torch.tensor(v_np[1:-1, 1:-1], dtype=torch.float64)

    a, b = domain
    mask_np = (X - a) * (b - X) * (Y - a) * (b - Y)
    if variant == "soft_boundary":
        mask_np = np.ones_like(mask_np)
    mask = torch.tensor(mask_np, dtype=torch.float64)

    def forward_grid() -> torch.Tensor:
        return net(xy).reshape(n_grid, n_grid) * mask

    def laplacian(p: torch.Tensor) -> torch.Tensor:
        interior = p[1:-1, 1:-1]
        return (
            (p[2:, 1:-1] - 2.0 * interior + p[:-2, 1:-1]) / hx**2
            + (p[1:-1, 2:] - 2.0 * interior + p[1:-1, :-2]) / hy**2
        )

    def terms(p: torch.Tensor) -> dict[str, torch.Tensor]:
        interior = p[1:-1, 1:-1]
        norm2 = grid.cell * torch.sum(interior**2)
        if variant == "unnormalized":
            # 关键差别：不除以 ‖ψ‖。残差与 η 的尺度行为使 ψ≡0 成为极小点。
            u = interior
            scale = torch.ones((), dtype=p.dtype, device=p.device)
        else:
            scale = torch.sqrt(norm2.clamp_min(1e-30))
            u = interior / scale
        h_psi = (-0.5 * laplacian(p) + v_i * interior) / scale
        eta = grid.cell * torch.sum(u * h_psi) / norm2.clamp_min(1e-300)
        r = h_psi - eta.detach() * u
        boundary = (
            (p[0, :] ** 2).sum()
            + (p[-1, :] ** 2).sum()
            + (p[:, 0] ** 2).sum()
            + (p[:, -1] ** 2).sum()
        ) / norm2.clamp_min(1e-30)
        out = {"residual": torch.mean(r**2), "energy": eta}
        if variant == "soft_boundary":
            out["boundary"] = boundary
        return out

    weights = {"residual": 1.0, "energy": w_energy, "boundary": w_boundary}
    if variant == "unnormalized":
        weights["boundary"] = 0.0  # 掩码已精确施加边界条件

    config = TrainConfig(
        mode="physics",
        epochs=epochs,
        lr=lr,
        n_collocation=n_grid * n_grid,
        log_every=max(1, epochs // 10),
        validation_every=max(1, epochs // 20),
        patience=10**9,  # 复现失败模式，不做早停
        term_weights=weights,
    )

    trace: list[dict] = []
    state = {"epoch": 0}

    def step_fn(epoch: int, trainer) -> dict:
        with torch.no_grad():
            p = forward_grid()
            interior = p[1:-1, 1:-1]
            norm2 = float(grid.cell * torch.sum(interior**2))
            edge = float(
                max(
                    p[0, :].abs().max(),
                    p[-1, :].abs().max(),
                    p[:, 0].abs().max(),
                    p[:, -1].abs().max(),
                )
            )
        if epoch % max(1, epochs // 20) == 0:
            trace.append(
                {"epoch": int(epoch), "norm2": norm2, "edge_max": edge}
            )
        state["epoch"] = epoch
        # 追踪量在 no_grad 里算，损失本身必须重新前向（带梯度图）。
        return {"terms": terms(forward_grid())}

    def validation_loss() -> float:
        with torch.no_grad():
            values = terms(forward_grid())
            total = torch.zeros((), dtype=torch.float64)
            for name, value in values.items():
                weight = weights.get(name, 1.0)
                if weight:
                    total = total + weight * value
            return float(total)

    history = Trainer(net, step_fn, config).fit(validation=validation_loss)

    with torch.no_grad():
        psi = forward_grid().cpu().numpy()
    interior = psi[1:-1, 1:-1]
    norm2 = float(grid.cell * np.sum(interior**2))
    # 证书输入必须是单位范数；这里先记录**原始**范数，再归一化后送证书。
    psi_unit = psi / math.sqrt(norm2) if norm2 > 0 else psi.copy()

    approx = approximate_spectrum_2d(grid, 6)
    cert = certify_eigenpair_2d(psi_unit, grid, isolate=True, approx=approx)
    try:
        discrete_ref = discrete_eigenvalue_2d(grid, 0, approx=approx)
    except ValueError:
        discrete_ref = float(approx[0])

    ref = np.asarray(psi_analytic(X, Y), dtype=float)
    ref = ref / math.sqrt(grid.cell * np.sum(ref[1:-1, 1:-1] ** 2))
    p = psi_unit.copy()
    p[0, :] = p[-1, :] = 0.0
    p[:, 0] = p[:, -1] = 0.0
    p = p / math.sqrt(grid.cell * np.sum(p[1:-1, 1:-1] ** 2))
    overlap = abs(float(grid.cell * np.sum(p[1:-1, 1:-1] * ref[1:-1, 1:-1])))

    result = {
        "experiment": "appendix_failures",
        "variant": variant,
        "potential": "well2d",
        "seed": int(seed),
        "n_grid": int(n_grid),
        "epochs": int(epochs),
        "width": int(width),
        "depth": int(depth),
        "weights": weights,
        "metrics": {
            "norm2_raw": norm2,
            "amplitude_raw": float(np.max(np.abs(psi))),
            "edge_max_raw": float(
                max(
                    np.abs(psi[0, :]).max(),
                    np.abs(psi[-1, :]).max(),
                    np.abs(psi[:, 0]).max(),
                    np.abs(psi[:, -1]).max(),
                )
            ),
            "energy_model": cert.energy,
            "energy_analytic": energy_analytic,
            "energy_discrete_ref": discrete_ref,
            "energy_error_vs_discrete": abs(cert.energy - discrete_ref),
            "overlap": overlap,
            "residual_norm": cert.residual_norm,
            "certificate_valid": cert.valid,
            "certificate_width": cert.width,
            "angle_bound": cert.eigenfunction_angle_bound,
            "certificate_reason": cert.reason,
        },
        "trace": trace,
        "history": history.to_dict(),
    }

    out = out_dir or (ROOT / "runs" / "appendix_failures" / variant)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "wavefunction2d.npz", x=x_np, y=y_np, psi=psi_unit, psi_raw=psi)
    (out / f"appendix_failures_{variant}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="附录 A 失败模式复现")
    parser.add_argument(
        "--variant", choices=("soft_boundary", "unnormalized"), required=True
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-grid", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    result = run(
        variant=args.variant,
        epochs=args.epochs,
        n_grid=args.n_grid,
        seed=args.seed,
        width=args.width,
        depth=args.depth,
    )
    m = result["metrics"]
    print(
        "[%s] norm2_raw=%.3e edge_max=%.3e E=%.4f (ref %.4f) overlap=%.4f "
        "residual=%.3e valid=%s"
        % (
            args.variant,
            m["norm2_raw"],
            m["edge_max_raw"],
            m["energy_model"],
            m["energy_discrete_ref"],
            m["overlap"],
            m["residual_norm"],
            m["certificate_valid"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
