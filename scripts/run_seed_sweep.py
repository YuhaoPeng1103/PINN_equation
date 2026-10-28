"""多种子复现扫描：把三个实验的每个配置在多个随机种子上重跑。

为什么需要它：HANDOFF 第 6 节 P0 指出，现有结果是**单次运行**，不能直接引用。
本脚本把每个 (实验, 配置, 种子) 作为独立子进程运行，写入

    runs/seeds/<group>/<config>/seed<k>/...

并生成 runs/seeds/manifest.json（含命令、返回码、耗时、日志路径）。
之后用 `python scripts/aggregate_seeds.py` 汇总均值 ± 标准差。

用法：
    python scripts/run_seed_sweep.py                    # 全部实验，种子 0/1/2
    python scripts/run_seed_sweep.py --groups exp2 --seeds 0 1 2 3 4
    python scripts/run_seed_sweep.py --jobs 8 --threads 4
    python scripts/run_seed_sweep.py --dry-run          # 只打印任务清单

默认每个进程 4 个 torch 线程、同时 6 个进程（24 线程），避免 32 核机器被
小网络训练全部占满——对这么小的 MLP，线程太多反而被同步开销拖慢。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "_seed_worker.py"

# 每个配置的固定参数；--seed / --out-dir 在生成任务时补上
CONFIGS: dict[str, list[dict]] = {
    "exp1": [
        {"name": "baseline", "args": ["--width", "64", "--depth", "3"], "epochs": 3000},
    ],
    "exp2": [
        {"name": "residual_only", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                            "--residual-only"], "epochs": 10000},
        {"name": "residual_learned", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                               "--residual-only", "--residual-energy", "learned"],
         "epochs": 10000},
        # E 的初值放在第一激发态能量上：同样的残差损失，若仍收敛到 E=0.5/ψ0，
        # 说明残差损失其实偏向基态；若停在 1.5/ψ1，则证明解在能级间退化。
        {"name": "residual_learned_e15", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                                  "--residual-only", "--residual-energy", "learned",
                                                  "--residual-energy-init", "1.5"],
         "epochs": 10000},
        # 真正的“全无监督”主配置：E 可学习（不告诉网络答案）+ Rayleigh 商。
        # 与 residual_learned 单变量对照，回答“加 Rayleigh 是否有用”。
        {"name": "rayleigh_learned", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                              "--residual-energy", "learned"], "epochs": 10000},
        {"name": "rayleigh_random", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                             "--energy-sampling", "random"], "epochs": 5000},
        {"name": "rayleigh_random_gradbalance", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                                         "--energy-sampling", "random", "--grad-balance"],
         "epochs": 5000},
        {"name": "rayleigh_grid", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                           "--energy-sampling", "grid"], "epochs": 5000},
    ],
    "exp3": [
        {"name": "k0", "args": ["--n-obs", "0"], "epochs": 2000},
        {"name": "k16", "args": ["--n-obs", "16"], "epochs": 2000},
        {"name": "k64", "args": ["--n-obs", "64"], "epochs": 2000},
    ],
    "states": [
        # 与 exp2 主表共用 width=128/depth=4，否则 state0 无法直接对照
        {"name": "state0", "args": ["--state", "0", "--width", "128", "--depth", "4"],
         "epochs": 5000},
        {"name": "state1", "args": ["--state", "1", "--width", "128", "--depth", "4",
                                     "--orthogonalize"], "epochs": 5000},
        {"name": "state2", "args": ["--state", "2", "--width", "128", "--depth", "4",
                                     "--orthogonalize"], "epochs": 5000},
        {"name": "state3", "args": ["--state", "3", "--width", "128", "--depth", "4",
                                     "--orthogonalize"], "epochs": 5000},
    ],
    # 后验证书实验：必须保存 ψ(x)，否则无法在训练后计算残差/证书。
    # learned0 是真正无监督（E 可学习、无参考态）；state1 用解析基做 deflation，
    # 带有已知泄漏，正好用来测试证书能否识别“能量好但方法泄漏”的解。
    "certs": [
        {"name": "learned0", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                      "--residual-energy", "learned", "--save-wavefunction"],
         "epochs": 5000},
        {"name": "state1_deflated", "args": ["--state", "1", "--width", "128", "--depth", "4",
                                             "--orthogonalize", "--save-wavefunction"],
         "epochs": 5000},
    ],
    # 证书协议的多势/多态扫描：训练目标与证书同口径，输出 wavefunction.npz。
    # 势 × 态 = 3 × 2 = 6 个配置，配 3 个种子共 18 个 run。
    # 基态用 --learned-energy：E 是可学习标量，残差里没有参考能量，这 9 个
    # run 才是 §5.1 说的 leakage-free 样本。只加 --no-energy 不够——残差
    # 仍然会用参考 E（见工程记录 20）。基态跑 10000 epoch：有限深阱在 3000
    # epoch 时会停在 E≈−7 的非本征态，10000 才收敛到 −48.97。
    "certs5": [
        {"name": f"{pot}_{state}", "args": ["--potential", pot, "--state", str(state),
                                            *(["--no-energy"] if state > 0
                                              else ["--learned-energy"])],
         "epochs": 10000 if state == 0 else 3000}
        for pot in ("harmonic", "infinite_well", "finite_well")
        for state in (0, 1)
    ],
    # 泄漏对照：与 certs5 基态同架构、同 10000 epoch，只把残差里的 E 换成
    # 参考值。用来量化“把答案告诉优化器”对能量/overlap/证书的实际影响。
    "certs5_refenergy": [
        {"name": f"{pot}_0", "args": ["--potential", pot, "--state", "0"],
         "epochs": 10000}
        for pot in ("harmonic", "infinite_well", "finite_well")
    ],
    # P0 修复：残差里的 E 用当前 ψ 的 Rayleigh 商，而不是可学习标量。
    # 可学习 E 在有限深阱上停在 −9.8（真值 −48.97），残差项实际没被满足
    # （工程记录 21）；Rayleigh 商是 ‖Hψ−Eψ‖ 对 E 的最优选择，且与证书同口径。
    "certs5_rayleigh": [
        {"name": f"{pot}_0", "args": ["--potential", pot, "--state", "0",
                                      "--rayleigh-energy"], "epochs": 10000}
        for pot in ("harmonic", "infinite_well", "finite_well")
    ],
    # C 线最小原型：激发态用**同一 seed 的 rayleigh 基态**做硬正交投影
    # （deflation 的参考来自训练输出，不读解析解），残差 E 仍用 Rayleigh 商。
    # 依赖 certs5_rayleigh 先跑完；路径里的 {seed} 在 build_jobs 里展开。
    "certs5_deflated": [
        {"name": f"{pot}_1", "args": [
            "--potential", pot, "--state", "1", "--rayleigh-energy",
            "--deflate-from",
            f"runs/seeds/certs5_rayleigh/{pot}_0/seed{{seed}}/wavefunction.npz",
        ], "epochs": 10000}
        for pot in ("harmonic", "infinite_well", "finite_well")
    ],
    # 链式 deflation：n=2 同时投掉同 seed 学出的 ψ₀（certs5_rayleigh）与
    # ψ₁（certs5_deflated）。依赖前两组先跑完；{seed} 在 build_jobs 里展开。
    "certs5_deflated2": [
        {"name": f"{pot}_2", "args": [
            "--potential", pot, "--state", "2", "--rayleigh-energy",
            "--deflate-from",
            f"runs/seeds/certs5_deflated/{pot}_1/seed{{seed}}/wavefunction.npz",
            "--deflate-from",
            f"runs/seeds/certs5_rayleigh/{pot}_0/seed{{seed}}/wavefunction.npz",
        ], "epochs": 10000}
        for pot in ("harmonic", "infinite_well", "finite_well")
    ],
    # 超参敏感性：只改一个变量，其余与 rayleigh_grid 基线一致
    "exp2_hparams": [
        {"name": "lr5e-4", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                     "--lr", "5e-4"], "epochs": 5000},
        {"name": "lr2e-3", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                     "--lr", "2e-3"], "epochs": 5000},
        {"name": "energy_weight1", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                            "--energy-weight", "1"], "epochs": 5000},
        {"name": "energy_weight100", "args": ["--state", "0", "--width", "128", "--depth", "4",
                                              "--energy-weight", "100"], "epochs": 5000},
        {"name": "epochs2000", "args": ["--state", "0", "--width", "128", "--depth", "4"],
         "epochs": 2000},
    ],
}

EXPERIMENT_OF = {"exp1": "exp1", "exp2": "exp2", "exp3": "exp3", "states": "exp2",
                 "exp2_hparams": "exp2", "certs": "exp2", "certs5": "exp5",
                 "certs5_refenergy": "exp5", "certs5_rayleigh": "exp5",
                 "certs5_deflated": "exp5", "certs5_deflated2": "exp5"}
OUTPUT_NAME = {
    "exp1": lambda _: "exp1_regularized.json",
    "exp2": lambda _: "exp2_pinn.json",
    "exp3": lambda n_obs: f"exp3_data_assimilation_k{n_obs}.json",
    "exp5": lambda _: "exp5_certificate_training.json",
}


def n_obs_of(args: list[str]) -> int:
    return int(args[args.index("--n-obs") + 1]) if "--n-obs" in args else 0


def build_jobs(groups: list[str], seeds: list[int], out_root: Path) -> list[dict]:
    jobs = []
    for group in groups:
        experiment = EXPERIMENT_OF[group]
        for config in CONFIGS[group]:
            for seed in seeds:
                out_dir = out_root / group / config["name"] / f"seed{seed}"
                # {seed} 用于依赖同一 seed 基态输出的配置（certs5_deflated）
                args = [a.format(seed=seed) if "{" in a else a for a in config["args"]]
                cmd = [
                    sys.executable, str(WORKER),
                    "--experiment", experiment,
                    "--seed", str(seed),
                    "--epochs", str(config["epochs"]),
                    "--out-dir", str(out_dir),
                    *args,
                ]
                jobs.append({
                    "group": group,
                    "config": config["name"],
                    "seed": seed,
                    "experiment": experiment,
                    "out_dir": str(out_dir),
                    "expected_json": str(out_dir / OUTPUT_NAME[experiment](n_obs_of(config["args"]))),
                    "cmd": cmd,
                })
    return jobs


def run_job(job: dict, threads: int) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # 同时限制 OpenMP/MKL，避免 torch.set_num_threads 之外的线程池继续抢占核
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    job["cmd"] = [*job["cmd"], "--threads", str(threads)]
    log_path = Path(job["out_dir"]) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(job["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    elapsed = time.perf_counter() - start
    return {
        **job,
        "returncode": proc.returncode,
        "seconds": round(elapsed, 1),
        "log": str(log_path),
        "ok": proc.returncode == 0 and Path(job["expected_json"]).exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="多种子复现扫描")
    parser.add_argument("--groups", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=6, help="同时运行的子进程数")
    parser.add_argument("--threads", type=int, default=4, help="每个子进程的 torch 线程数")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "seeds")
    parser.add_argument("--force", action="store_true", help="已有结果的配置也重跑")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务清单，不执行")
    args = parser.parse_args()

    jobs = build_jobs(args.groups, args.seeds, args.out_root)
    if not args.force:
        pending = [j for j in jobs if not Path(j["expected_json"]).exists()]
    else:
        pending = jobs
    print(f"任务总数 {len(jobs)}，待运行 {len(pending)}，并发 {args.jobs}，每进程 {args.threads} 线程")
    if args.dry_run:
        for job in pending:
            print(f"  [{job['group']}/{job['config']}/seed{job['seed']}] {' '.join(job['cmd'])}")
        return 0
    if not pending:
        print("全部结果已存在（用 --force 重跑）。")
        return 0

    results: list[dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_job, job, args.threads): job for job in pending}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else "FAIL"
            if not result["ok"]:
                failed += 1
            print(f"[{i}/{len(pending)}] {status} {result['group']}/{result['config']}/"
                  f"seed{result['seed']} ({result['seconds']}s)", flush=True)

    manifest_path = args.out_root / "manifest.json"
    previous: dict[tuple[str, str, int], dict] = {}
    if manifest_path.exists():
        # 分批跑（例如先跑主实验、再跑超参扫描）时合并清单，不覆盖上一批的记录
        for old in json.loads(manifest_path.read_text(encoding="utf-8")).get("jobs", []):
            previous[(old["group"], old["config"], old["seed"])] = old
    for result in results:
        previous[(result["group"], result["config"], result["seed"])] = result
    merged = sorted(previous.values(), key=lambda r: (r["group"], r["config"], r["seed"]))
    manifest = {
        "seeds": sorted({r["seed"] for r in merged}),
        "jobs": merged,
        "n_ok": sum(1 for r in merged if r["ok"]),
        "n_failed": sum(1 for r in merged if not r["ok"]),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"完成：{manifest['n_ok']} 成功，{failed} 失败；清单 {args.out_root / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
