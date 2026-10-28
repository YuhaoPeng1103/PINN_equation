"""单次实验 worker：供 run_seed_sweep.py 以子进程方式调用。

把"跑一个 (实验, 配置, 种子) 组合"封装成独立进程，有两个好处：
  * 每个进程可以单独限制 torch 线程数，避免多个进程把 32 个核全部抢占；
  * 某个种子崩溃不会影响其它种子（manifests 会记录返回码与日志路径）。

一般不需要直接调用；用 `python scripts/run_seed_sweep.py` 即可。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行单个实验配置（种子扫描的底层 worker）")
    parser.add_argument("--experiment", required=True, choices=("exp1", "exp2", "exp3", "exp5"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    # exp1 / exp2 网络规模
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    # exp2
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--grad-balance", action="store_true")
    parser.add_argument("--residual-only", action="store_true")
    parser.add_argument("--energy-sampling", choices=("grid", "random"), default="grid")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--residual-energy", choices=("reference", "learned"), default="reference")
    parser.add_argument("--residual-energy-init", type=float, default=1.0)
    parser.add_argument("--save-wavefunction", action="store_true",
                        help="额外保存 wavefunction.npz（后验证书需要 ψ(x)）")
    # exp3
    parser.add_argument("--n-obs", type=int, default=0)
    # exp5（证书同口径训练）
    parser.add_argument("--potential", choices=("harmonic", "infinite_well", "finite_well"),
                        default="harmonic")
    parser.add_argument("--learned-energy", action="store_true")
    parser.add_argument("--rayleigh-energy", action="store_true")
    parser.add_argument("--deflate-from", type=Path, action="append", default=None,
                        help="可重复；链式 deflation 时依次给出 ψ0, ψ1, …")
    parser.add_argument("--no-energy", action="store_true")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))

    if args.experiment == "exp1":
        import exp1_regularized_network as mod
        result = mod.run(
            seed=args.seed,
            epochs=args.epochs,
            width=args.width,
            depth=args.depth,
            out_dir=args.out_dir,
        )
        metrics = {"final_loss": result.get("final_loss")}
        for energy, info in result["energies"].items():
            metrics[f"E={energy}"] = info["max_abs_err"]
        kind = "baseline"
    elif args.experiment == "exp2":
        import exp2_unregularized_pinn as mod
        result = mod.run(
            state=args.state,
            epochs=args.epochs,
            seed=args.seed,
            width=args.width,
            depth=args.depth,
            grad_balance=args.grad_balance,
            orthogonalize=args.orthogonalize,
            use_energy=not args.residual_only,
            energy_sampling=args.energy_sampling,
            lr=args.lr,
            energy_weight=args.energy_weight,
            residual_energy=args.residual_energy,
            residual_energy_init=args.residual_energy_init,
            out_dir=args.out_dir,
            save_wavefunction=args.save_wavefunction,
        )
        metrics = result["metrics"]
        kind = f"state={args.state}"
    elif args.experiment == "exp5":
        import exp5_certificate_training as mod
        result = mod.run(
            state=args.state,
            epochs=args.epochs,
            width=args.width,
            depth=args.depth,
            seed=args.seed,
            lr=args.lr,
            learned_energy=args.learned_energy,
            rayleigh_energy=args.rayleigh_energy,
            use_energy=not args.no_energy,
            potential=args.potential,
            deflate_from=args.deflate_from,
            out_dir=args.out_dir,
        )
        metrics = result["metrics"]
        kind = f"{args.potential} state={args.state}"
    else:
        import exp3_data_assimilation as mod
        result = mod.run(
            n_obs=args.n_obs,
            epochs=args.epochs,
            seed=args.seed,
            out_dir=args.out_dir,
        )
        metrics = result["metrics"]
        kind = f"k={args.n_obs}"

    summary = ", ".join(
        f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )
    print(f"[worker] {args.experiment} {kind} seed={args.seed} -> {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
