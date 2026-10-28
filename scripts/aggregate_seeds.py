"""汇总 run_seed_sweep.py 的多种子结果：均值 ± 标准差 + Markdown 表。

输出：
    runs/seeds/summary.json   机器可读的逐配置统计（含每个种子的原始值）
    runs/seeds/summary.md     可直接贴进 docs/results.md 的表格

统计口径：样本标准差（ddof=1，n≥2 时）；n=1 时 std 记 0 并在输出里标注。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUMMARY_METRICS = ("energy_model", "energy_abs_err", "energy_rel_err", "overlap", "inf_norm", "residual_rms")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals) if vals else float("nan"),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def discover(out_root: Path, group: str, config: str, name: str) -> list[dict]:
    """返回 config 下每个种子的结果 dict（按种子号排序）。"""
    runs = []
    base = out_root / group / config
    if not base.exists():
        return runs
    for seed_dir in sorted(base.glob("seed*"), key=lambda p: int(p.name[4:])):
        path = seed_dir / name
        if path.exists():
            runs.append(read_json(path))
    return runs


def summarize_exp2(out_root: Path, group: str, config: str) -> dict | None:
    runs = discover(out_root, group, config, "exp2_pinn.json")
    if not runs:
        return None
    metrics = {key: stats([r["metrics"][key] for r in runs]) for key in SUMMARY_METRICS}
    out = {
        "n": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "metrics": metrics,
        "config": {k: runs[0].get(k) for k in
                   ("state", "orthogonalize", "grad_balance", "use_energy", "energy_sampling",
                    "epochs", "lr", "energy_weight", "residual_energy", "residual_energy_init")},
    }
    learned = [r["learned_energy"] for r in runs if r.get("learned_energy") is not None]
    if learned:
        out["learned_energy"] = stats(learned)
        # 学习到的 E 与波函数自身 Rayleigh 商的差：接近 0 才说明残差方程
        # 是自洽解出的本征对，而不是 E 与 ψ 各走各的。
        rayleigh = [r["metrics"]["energy_model"] for r in runs
                    if r.get("learned_energy") is not None]
        out["learned_vs_rayleigh_gap"] = stats([abs(e - q) for e, q in zip(learned, rayleigh)])
        # 谐振子 E_n = n + 1/2；仅用于判断收敛到了哪个本征态
        out["learned_state_index"] = [int(round(e - 0.5)) for e in learned]
        ref = float(runs[0]["metrics"]["energy_ref"])
        out["learned_near_ref_fraction"] = sum(abs(e - ref) <= 0.05 for e in learned) / len(learned)
    return out


def summarize_exp3(out_root: Path, config: str, n_obs: int) -> dict | None:
    runs = discover(out_root, "exp3", config, f"exp3_data_assimilation_k{n_obs}.json")
    if not runs:
        return None
    metrics = {key: stats([r["metrics"][key] for r in runs]) for key in SUMMARY_METRICS}
    return {
        "n": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "n_obs": n_obs,
        "metrics": metrics,
    }


def summarize_exp1(out_root: Path, config: str = "baseline") -> dict | None:
    runs = discover(out_root, "exp1", config, "exp1_regularized.json")
    if not runs:
        return None
    energies = sorted(runs[0]["energies"], key=float)
    per_energy = {}
    for e in energies:
        per_energy[e] = {
            "is_interpolated": bool(runs[0]["energies"][e]["is_interpolated"]),
            "max_abs_err": stats([r["energies"][e]["max_abs_err"] for r in runs]),
        }
    train = [r["energies"][e]["max_abs_err"] for r in runs for e in energies
             if not r["energies"][e]["is_interpolated"]]
    interp = [r["energies"][e]["max_abs_err"] for r in runs for e in energies
              if r["energies"][e]["is_interpolated"]]
    return {
        "n": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "energies": per_energy,
        "train_max_abs_err": stats(train),
        "interp_max_abs_err": stats(interp),
    }


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}e}"


def fmt_pm(entry: dict, digits: int = 3) -> str:
    if not entry or entry["n"] == 0:
        return "—"
    if entry["n"] == 1:
        return fmt(entry["mean"], digits)
    return f"{fmt(entry['mean'], digits)} ± {fmt(entry['std'], digits)}"


def markdown(summary: dict) -> str:
    lines = ["# 多种子复现汇总", "",
             "> 由 `scripts/aggregate_seeds.py` 生成。`±` 后为样本标准差（ddof=1）；n=1 时没有误差棒。", ""]

    exp2 = summary.get("exp2", {})
    if exp2:
        lines += ["## 实验二 · 无监督 PINN（谐振子基态）", "",
                  "| 配置 | n | 残差中的 E | Rayleigh | 采样 | 能量 | 学习到的 E | 绝对误差 | 相对误差 | overlap | inf_norm |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for config, data in exp2.items():
            m = data["metrics"]
            cfg = data["config"]
            learned = data.get("learned_energy")
            e_mode = cfg.get("residual_energy", "reference")
            e_txt = (f"learned({cfg.get('residual_energy_init')})" if e_mode == "learned"
                     else "ref")
            ray_txt = "有" if cfg.get("use_energy", True) else "无"
            # 学习到的 E 逐个种子列出，而不是取均值：不同种子可能收敛到不同本征态，
            # 这时 “均值 ± 标准差” 会掩盖“多解”这个最重要的信息。
            if learned:
                learned_txt = " / ".join(f"{v:.4f}" for v in learned["values"])
            else:
                learned_txt = "—"
            lines.append(
                f"| {config} | {data['n']} | {e_txt} | {ray_txt} | "
                f"{cfg.get('energy_sampling', 'grid')} | {fmt_pm(m['energy_model'], 5)} | "
                f"{learned_txt} | "
                f"{fmt_pm(m['energy_abs_err'])} | {fmt_pm(m['energy_rel_err'])} | "
                f"{fmt_pm(m['overlap'], 5)} | {fmt_pm(m['inf_norm'])} |"
            )
        lines.append("")
        learned_rows = [(c, d) for c, d in exp2.items() if d.get("learned_energy")]
        if learned_rows:
            lines.append(
                "> “学习到的 E” 一列按种子顺序列出实际值（不取均值）：不同种子可能停在"
                "不同本征态，均值会掩盖这种多解。下方一行给出按 E = n + ½ 判定的态序号，"
                "以及学习到的 E 与波函数 Rayleigh 商之差（差≈0 表示残差方程自洽）。"
            )
            parts = []
            for c, d in learned_rows:
                gap = d["learned_vs_rayleigh_gap"]
                states = ",".join(f"n={n}" for n in d["learned_state_index"])
                parts.append(f"`{c}`: [{states}]，|E_learned − E_Rayleigh| = "
                             f"{fmt(gap['mean'], 2)} ± {fmt(gap['std'], 2)}")
            lines += ["", "> " + "；".join(parts) + "。", ""]

    exp3 = summary.get("exp3", {})
    if exp3:
        lines += ["## 实验三 · 数据同化（无限深阱基态）", "",
                  "| k | n | 绝对误差 | 相对误差 | overlap | inf_norm |",
                  "|---|---|---|---|---|---|"]
        for config, data in exp3.items():
            m = data["metrics"]
            lines.append(
                f"| {data['n_obs']} | {data['n']} | {fmt_pm(m['energy_abs_err'])} | "
                f"{fmt_pm(m['energy_rel_err'])} | {fmt_pm(m['overlap'])} | {fmt_pm(m['inf_norm'])} |"
            )
        lines.append("")

    states = summary.get("states", {})
    if states:
        lines += ["## 实验二扩展 · 激发态（正交约束）", "",
                  "| 态 | n | 绝对误差 | 相对误差 | overlap | inf_norm |",
                  "|---|---|---|---|---|---|"]
        for config, data in states.items():
            m = data["metrics"]
            state = data["config"].get("state")
            lines.append(
                f"| n={state} | {data['n']} | {fmt_pm(m['energy_abs_err'])} | "
                f"{fmt_pm(m['energy_rel_err'])} | {fmt_pm(m['overlap'])} | {fmt_pm(m['inf_norm'])} |"
            )
        lines.append("")

    hparams = summary.get("exp2_hparams", {})
    if hparams:
        lines += ["## 实验二 · 超参敏感性（单变量，谐振子基态）", "",
                  "| 配置 | lr | energy 权重 | epochs | n | 绝对误差 | 相对误差 | overlap |",
                  "|---|---|---|---|---|---|---|---|"]
        for config, data in hparams.items():
            m = data["metrics"]
            cfg = data["config"]
            lines.append(
                f"| {config} | {cfg.get('lr')} | {cfg.get('energy_weight')} | {cfg.get('epochs')} | "
                f"{data['n']} | {fmt_pm(m['energy_abs_err'])} | {fmt_pm(m['energy_rel_err'])} | "
                f"{fmt_pm(m['overlap'], 5)} |"
            )
        lines += ["", "基线为 `exp2/rayleigh_grid`（lr=0.001、energy 权重 10、5000 epochs）。", ""]

    exp1 = summary.get("exp1", {})
    if exp1:
        lines += ["## 实验一 · 正则化网络与能量插值", "",
                  "| 能量 E | 类型 | max\\|ψ_model − ψ_ref\\| |", "|---|---|---|"]
        for energy, data in exp1["energies"].items():
            tag = "插值" if data["is_interpolated"] else "训练"
            lines.append(f"| {energy} | {tag} | {fmt_pm(data['max_abs_err'])} |")
        lines += ["",
                  f"训练能量误差（4 个能量 × 各种子合并）：{fmt_pm(exp1['train_max_abs_err'])}；"
                  f"插值能量误差范围（跨种子全部值）："
                  f"{fmt(min(exp1['interp_max_abs_err']['values']), 3)} ~ "
                  f"{fmt(max(exp1['interp_max_abs_err']['values']), 3)}。", ""]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总多种子实验结果")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "seeds")
    args = parser.parse_args()
    out_root: Path = args.out_root

    summary: dict = {"out_root": str(out_root), "exp1": {}, "exp2": {}, "exp3": {}, "states": {},
                     "exp2_hparams": {}}

    exp2_configs = [p.name for p in (out_root / "exp2").glob("*") if p.is_dir()]
    for config in sorted(exp2_configs):
        data = summarize_exp2(out_root, "exp2", config)
        if data:
            summary["exp2"][config] = data

    for config, n_obs in (("k0", 0), ("k16", 16), ("k64", 64)):
        data = summarize_exp3(out_root, config, n_obs)
        if data:
            summary["exp3"][config] = data

    state_configs = [p.name for p in (out_root / "states").glob("*") if p.is_dir()]
    for config in sorted(state_configs):
        data = summarize_exp2(out_root, "states", config)
        if data:
            summary["states"][config] = data

    for config in sorted(p.name for p in (out_root / "exp2_hparams").glob("*") if p.is_dir()):
        data = summarize_exp2(out_root, "exp2_hparams", config)
        if data:
            summary["exp2_hparams"][config] = data

    data = summarize_exp1(out_root)
    if data:
        summary["exp1"] = data

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = markdown(summary)
    (out_root / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n写入 {out_root / 'summary.json'} 与 {out_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
