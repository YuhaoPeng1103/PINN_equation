"""多态证书表：基态（Rayleigh 残差）与第一激发态（可学习 deflation）。

``go_nogo.py`` 只统计基态——它的判定口径是“无泄漏的基态证书是否包含参考
本征值”。激发态是另一回事：没有 deflation 时网络要么停在基态、要么混态
（见工程记录 21/22），所以激发态必须先做正交投影。本脚本把两组结果并排
列出来，回答“同一套证书流程能否认证激发态”。

**deflation 的参考态是学出来的**：``certs5_deflated`` 用的是同一 seed 的
``certs5_rayleigh`` 基态输出（不是解析解），因此这里没有任何“把答案给
优化器”的泄漏。``deflate_overlap`` 是投影后再取一次内积的残差，理想值 0。

输入是 ``filter_certificates.py`` 的 summary.json；本脚本不重算证书。

用法：
    python scripts/multistate_report.py \
        --ground  runs/certificates_filtered/certs5_rayleigh/summary.json \
        --excited runs/certificates_filtered/certs5_deflated/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_meta(path: str) -> dict:
    original = ROOT / path
    for name in ("exp5_certificate_training.json", "exp2_pinn.json"):
        candidate = original.with_name(name)
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def collect(summary_path: Path, role: str) -> list[dict]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        meta = load_meta(row["path"])
        cert = row["filtered_certificate"]
        error = None
        if cert.get("energy") is not None and row.get("discrete_ref") is not None:
            error = abs(cert["energy"] - row["discrete_ref"])
        metrics = meta.get("metrics") or {}
        out.append({
            "role": role,
            "path": row["path"],
            "potential": meta.get("potential") or "harmonic",
            "target_state": meta.get("state"),
            "deflate_from": meta.get("deflate_from"),
            "deflate_overlap": meta.get("deflate_overlap"),
            "cert_state": cert.get("state_index"),
            "state_ok": bool(row.get("state_matches_target")),
            "energy": cert.get("energy"),
            "discrete_ref": row.get("discrete_ref"),
            "energy_error": error,
            "overlap": metrics.get("overlap"),
            "raw_residual": row["raw_certificate"].get("residual_norm"),
            "raw_valid": bool(row["raw_certificate"].get("valid")),
            "raw_width": row.get("raw_width"),
            "cutoff": row.get("cutoff"),
            "filtered_residual": cert.get("residual_norm"),
            "filtered_width": row.get("filtered_width"),
            "filtered_angle": row.get("filtered_angle"),
            "contains": bool(row.get("filtered_contains_ref")),
            "rel_l2_change": row.get("rel_l2_change"),
        })
    return out


def aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[str, int | None], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["potential"], row["target_state"]), []).append(row)
    table = {}
    for (potential, state), subset in sorted(groups.items()):
        errors = [r["energy_error"] for r in subset if r["energy_error"] is not None]
        widths = [r["filtered_width"] for r in subset if r["filtered_width"] is not None]
        overlaps = [r["overlap"] for r in subset if r["overlap"] is not None]
        deflate = [abs(r["deflate_overlap"]) for r in subset
                   if r["deflate_overlap"] is not None]
        table[f"{potential}_n{state}"] = {
            "potential": potential,
            "state": state,
            "n": len(subset),
            "raw_certified": sum(1 for r in subset if r["raw_valid"]),
            "certified_state_ok": sum(1 for r in subset if r["state_ok"]),
            # 只统计“包含**目标态**的离散参考值”的 run：如果网络跳到 n=2，
            # 即使证书对 n=2 成立也不算完成多态目标（谐振子 seed2 就是这样）。
            "contains_target": sum(1 for r in subset if r["contains"] and r["state_ok"]),
            "contains": sum(1 for r in subset if r["contains"]),
            "median_energy_error": float(np.median(errors)) if errors else None,
            "max_energy_error": float(np.max(errors)) if errors else None,
            "median_overlap": float(np.median(overlaps)) if overlaps else None,
            "min_overlap": float(np.min(overlaps)) if overlaps else None,
            "median_filtered_width": float(np.median(widths)) if widths else None,
            "max_filtered_width": float(np.max(widths)) if widths else None,
            "max_deflate_overlap": float(np.max(deflate)) if deflate else None,
            "runs": [
                {k: r[k] for k in ("path", "cert_state", "energy_error", "overlap",
                                   "filtered_width", "contains", "deflate_overlap")}
                for r in subset
            ],
        }
    return table


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown(rows: list[dict], table: dict) -> str:
    name = {"harmonic": "谐振子", "infinite_well": "无限深阱", "finite_well": "有限深阱"}
    lines = [
        "# 多态证书（n = 0, 1, 2）",
        "",
        "| 势 | n | runs | 原始可认证 | 证书态一致 | 滤波后包含**目标态** | 中位 E 误差 | "
        "中位 overlap | 中位滤波宽 | 最大滤波宽 | max‖⟨φ|ψ⟩‖ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(table, key=lambda k: (table[k]["potential"], table[k]["state"] or 0)):
        s = table[key]
        lines.append(
            f"| {name.get(s['potential'], s['potential'])} | {s['state']} | {s['n']} | "
            f"{s['raw_certified']}/{s['n']} | {s['certified_state_ok']}/{s['n']} | "
            f"{s['contains_target']}/{s['n']} | {_fmt(s['median_energy_error'])} | "
            f"{_fmt(s['median_overlap'], 6)} | {_fmt(s['median_filtered_width'])} | "
            f"{_fmt(s['max_filtered_width'])} | {_fmt(s['max_deflate_overlap'])} |"
        )
    lines += [
        "",
        "> 激发态组的 deflation 参考是**同一 seed 学出来的低能态**"
        "（n=1 投 ψ₀；n=2 链式投 ψ₀ 与 ψ₁），不是解析解；``max‖⟨φ|ψ⟩‖`` 是"
        "投影后再取内积的残差，理想值为 0。滤波宽度与基态组同口径（Temple 区间半宽），"
        "跨势不可直接比较——无限深阱的定义域只有 [0,1]，同样的相对精度对应"
        "更小的绝对宽度。",
        "",
        "## 逐 run",
        "",
        "| 势 | n | run | 证书态 | E 误差 | overlap | 滤波宽 | 包含 | ⟨φ|ψ⟩ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["potential"], r["target_state"] or 0, r["path"])):
        lines.append(
            f"| {name.get(r['potential'], r['potential'])} | {r['target_state']} | "
            f"{r['path']} | {_fmt(r['cert_state'])} | {_fmt(r['energy_error'])} | "
            f"{_fmt(r['overlap'], 6)} | {_fmt(r['filtered_width'])} | "
            f"{'✓' if r['contains'] else '✗'} | {_fmt(r['deflate_overlap'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="多态（基态/激发态）证书汇总")
    parser.add_argument("--ground", type=Path,
                        default=ROOT / "runs" / "certificates_filtered" / "certs5_rayleigh" / "summary.json")
    parser.add_argument("--excited", type=Path, nargs="+",
                        default=[ROOT / "runs" / "certificates_filtered" / "certs5_deflated" / "summary.json",
                                 ROOT / "runs" / "certificates_filtered" / "certs5_deflated2" / "summary.json"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "multistate")
    args = parser.parse_args()

    rows: list[dict] = []
    for path, role in [(args.ground, "ground"), *[(p, "excited") for p in args.excited]]:
        if path.exists():
            rows += collect(path, role)
        elif role == "excited":
            print(f"缺少 {path}；先对该目录运行 filter_certificates.py")
    if not rows:
        print("没有可用输入")
        return 1

    table = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"table": table, "runs": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = markdown(rows, table)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
