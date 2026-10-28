"""汇总泄漏对照：同一架构/种子/epoch，只改残差里的 E 来源。

四组（都读 ``filter_certificates.py`` 的 summary.json）：

* ``rayleigh``  —— ``certs5_rayleigh`` 基态，E=当前 ψ 的 Rayleigh 商（无泄漏）；
* ``learned``   —— ``certs5`` 基态，E 可学习（无泄漏，但可能不收敛）；
* ``reference`` —— ``certs5_refenergy`` 基态，E 固定为参考值（泄漏）；
* ``deflated``  —— ``certs`` 的 ``state1_deflated``，用解析参考态做正交化（泄漏）。

输出每个 run 的能量误差、overlap、原始/滤波后证书宽度，以及按组的中位数。
**不重新计算证书**，只做汇总；证书由 ``filter_certificates.py`` 负责。

用法：
    python scripts/leakage_audit.py \
        --learned runs/certificates_filtered/certs5/summary.json \
        --reference runs/certificates_filtered/certs5_refenergy/summary.json \
        --deflated runs/certificates_filtered/certs/summary.json
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


def classify(path: str, meta: dict) -> str:
    if meta.get("orthogonalize"):
        return "deflated"
    if meta.get("residual_energy") == "rayleigh":
        return "rayleigh"
    if meta.get("residual_energy") == "learned":
        return "learned"
    if meta.get("residual_energy") == "reference":
        return "reference"
    return "unknown"


def collect(summary_path: Path, group_hint: str | None) -> list[dict]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        meta = load_meta(row["path"])
        group = classify(row["path"], meta) if group_hint is None else group_hint
        if group_hint == "deflated" and not meta.get("orthogonalize"):
            # 同一个目录里还有未正交化的 state0，那是另一条对照，不属于本组
            continue
        if group_hint in ("learned", "reference", "rayleigh") and meta.get("state") not in (0, None):
            # 泄漏对照只比基态：certs5 目录里还混着 3000-epoch 的 n=1 run，
            # 它们是“固定参考 E + 无 deflation”的失败样本，混进来会污染中位数。
            continue
        cert = row["filtered_certificate"]
        metrics = meta.get("metrics") or {}
        width = row.get("filtered_width")
        error = None
        if cert.get("energy") is not None and row.get("discrete_ref") is not None:
            error = abs(cert["energy"] - row["discrete_ref"])
        out.append({
            "group": group,
            "path": row["path"],
            "potential": meta.get("potential"),
            "target_state": meta.get("state"),
            "cert_state": cert.get("state_index"),
            "energy_error": error,
            "overlap": metrics.get("overlap"),
            "raw_residual": row["raw_certificate"].get("residual_norm"),
            "raw_valid": bool(row["raw_certificate"].get("valid")),
            "raw_width": row.get("raw_width"),
            "cutoff": row.get("cutoff"),
            "filtered_width": width,
            "filtered_angle": row.get("filtered_angle"),
            "contains": bool(row.get("filtered_contains_ref")),
            "rel_l2_change": row.get("rel_l2_change"),
        })
    return out


def aggregate(rows: list[dict]) -> dict:
    groups = sorted({r["group"] for r in rows})
    summary = {}
    for group in groups:
        subset = [r for r in rows if r["group"] == group]
        widths = [r["filtered_width"] for r in subset if r["filtered_width"] is not None]
        errors = [r["energy_error"] for r in subset if r["energy_error"] is not None]
        overlaps = [r["overlap"] for r in subset if r["overlap"] is not None]
        summary[group] = {
            "n": len(subset),
            "raw_certified": sum(1 for r in subset if r["raw_valid"]),
            "filtered_certified": sum(1 for r in subset if r["contains"]),
            "median_energy_error": float(np.median(errors)) if errors else None,
            "median_overlap": float(np.median(overlaps)) if overlaps else None,
            "median_filtered_width": float(np.median(widths)) if widths else None,
            "max_filtered_width": float(np.max(widths)) if widths else None,
            "median_rel_l2_change": (
                float(np.median([r["rel_l2_change"] for r in subset
                                 if r["rel_l2_change"] is not None]))
                if subset else None
            ),
        }
    return summary


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# 泄漏对照",
        "",
        "| 组 | n | 原始可认证 | 滤波后包含 | 中位 E 误差 | 中位 overlap | "
        "中位滤波宽度 | 最大滤波宽度 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group in sorted(summary):
        s = summary[group]
        lines.append(
            f"| {group} | {s['n']} | {s['raw_certified']}/{s['n']} | "
            f"{s['filtered_certified']}/{s['n']} | {_fmt(s['median_energy_error'])} | "
            f"{_fmt(s['median_overlap'], 6)} | {_fmt(s['median_filtered_width'])} | "
            f"{_fmt(s['max_filtered_width'])} |"
        )
    lines.extend([
        "",
        "## 逐 run",
        "",
        "| 组 | run | 势 | 目标态 | 证书态 | E 误差 | overlap | 原始 r | "
        "原始有效 | 原始宽 | K | 滤波宽 | 包含 | rel L2 改变 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['path']} | {_fmt(r['potential'])} | "
            f"{_fmt(r['target_state'])} | {_fmt(r['cert_state'])} | "
            f"{_fmt(r['energy_error'])} | {_fmt(r['overlap'], 6)} | "
            f"{_fmt(r['raw_residual'])} | {'✓' if r['raw_valid'] else '✗'} | "
            f"{_fmt(r['raw_width'])} | {_fmt(r['cutoff'])} | "
            f"{_fmt(r['filtered_width'])} | {'✓' if r['contains'] else '✗'} | "
            f"{_fmt(r['rel_l2_change'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="泄漏对照汇总")
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rayleigh", type=Path, default=None,
                        help="Rayleigh 商残差组（同样是 leakage-free，E 来自 ψ）")
    parser.add_argument("--deflated", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "leakage_audit")
    args = parser.parse_args()

    rows = []
    rows += collect(args.learned, "learned")
    rows += collect(args.reference, "reference")
    if args.rayleigh and args.rayleigh.exists():
        rows += collect(args.rayleigh, "rayleigh")
    if args.deflated and args.deflated.exists():
        rows += collect(args.deflated, "deflated")

    summary = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "runs": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = markdown(rows, summary)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
