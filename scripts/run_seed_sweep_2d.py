"""二维部分（实验六）的多种子扫描。

与 ``run_seed_sweep.py`` 分开：二维训练每个 run 要几十秒到几分钟，
``_seed_worker`` 的分派逻辑与一维实验耦合；这里直接用子进程跑
``exp6_certificate_training_2d.py``，输出

    runs/seeds2d/<potential>/seed<k>/{wavefunction2d.npz, exp6_...json, run.log}

然后对整棵树运行 ``filter_certificates2d.py``。

用法：
    python scripts/run_seed_sweep_2d.py --jobs 2
    python scripts/run_seed_sweep_2d.py --potentials harmonic2d --seeds 0
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
SCRIPT = ROOT / "scripts" / "exp6_certificate_training_2d.py"


def build_jobs(potentials, seeds, out_root, epochs, n_grid, width, depth) -> list[dict]:
    jobs = []
    for potential in potentials:
        for seed in seeds:
            out_dir = out_root / potential / f"seed{seed}"
            cmd = [
                sys.executable, str(SCRIPT),
                "--potential", potential,
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--n-grid", str(n_grid),
                "--width", str(width),
                "--depth", str(depth),
                "--out-dir", str(out_dir),
            ]
            jobs.append({
                "potential": potential,
                "seed": seed,
                "out_dir": str(out_dir),
                "expected_json": str(out_dir / "exp6_certificate_training_2d.json"),
                "cmd": cmd,
            })
    return jobs


def run_job(job: dict, threads: int) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
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
    parser = argparse.ArgumentParser(description="二维多种子扫描（实验六）")
    parser.add_argument("--potentials", nargs="+",
                        choices=("harmonic2d", "well2d", "anharmonic2d"),
                        default=["harmonic2d", "well2d", "anharmonic2d"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=2, help="同时运行的子进程数（内存受限，建议 ≤2）")
    parser.add_argument("--threads", type=int, default=2, help="每个子进程的 torch 线程数")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "seeds2d")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.potentials, args.seeds, args.out_root, args.epochs,
                      args.n_grid, args.width, args.depth)
    pending = jobs if args.force else [
        j for j in jobs if not Path(j["expected_json"]).exists()
    ]
    print(f"任务总数 {len(jobs)}，待运行 {len(pending)}，并发 {args.jobs}")
    if args.dry_run:
        for job in pending:
            print("  ", " ".join(job["cmd"]))
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
            print(f"[{i}/{len(pending)}] {status} {result['potential']}/seed{result['seed']} "
                  f"{result['seconds']:.0f}s")

    manifest_path = args.out_root / "manifest_2d.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": results, "failed": failed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"写出 {manifest_path}；失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
