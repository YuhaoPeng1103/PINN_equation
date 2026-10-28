"""按 ``docs/certification.md`` §5.3 汇总 A 线的 go/no-go 判定。

输入是 ``filter_certificates.py`` 写出的 ``summary.json``（每个 run 的原始/
滤波后证书、离散参考值、滤波改变量）。本脚本只读这些文件与训练元数据，
**不重新计算证书**，因此结果是可复核的纯汇总。

无泄漏 run 的定义（与 §5.1 一致）：
* ``runs/seeds/certs5_rayleigh/<potential>_0/seed*``：基态、无 deflation，
  残差里的 E 是当前 ψ 的 Rayleigh 商（A 线主协议；`--rayleigh-energy`）。
  默认的 ``--filtered`` 合并了这一组与下面的 ``certs/learned0``。
* ``runs/seeds/certs5/<potential>_0/seed*``：可学习 E 的对照组
  （``residual_energy == "learned"``），同样算无泄漏。
* ``runs/seeds/certs/learned0/seed*``：E 可学习、无解析态 deflation。
  早期版本的 certs5 基态用参考 E，那批结果已另存为
  ``runs/seeds/certs5_leaked``，只用于泄漏对照，不算进成功率。

用法：
    python scripts/go_nogo.py
    python scripts/go_nogo.py --filtered runs/certificates_filtered/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_meta(original: Path) -> dict:
    for name in ("exp5_certificate_training.json", "exp2_pinn.json"):
        path = original.with_name(name)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def discrete_angle(filtered_path: Path, index: int) -> float | None:
    """滤波后 ψ 与**离散**本征向量的夹角（夹角界的正确比较对象）。

    不能用解析本征态：H_h 与连续算子的本征函数相差 O(h²)，这个差
    (~1e-5 rad) 比滤波后的夹角界 (~1e-9) 大得多，会把“界成立”误判成
    “界被违反”。夹角界证明的是到 H_h 本征向量的距离。

    也不用 ``arccos(overlap)``：overlap 接近 1 时，1e-16 的求和误差会被
    放大成 1e-8 的“夹角”。直接算归一化差向量 ``‖ψ/‖ψ‖ − φ/‖φ‖‖``，
    在 1e-13 的量级仍然可信。
    """
    try:
        from scipy.linalg import eigh_tridiagonal
    except ImportError:  # pragma: no cover
        return None
    data = np.load(filtered_path)
    x = np.asarray(data["x"], dtype=float)
    psi = np.asarray(data["psi"], dtype=float)
    potential = np.asarray(data["potential"], dtype=float)
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    _, vectors = eigh_tridiagonal(diag, off, select="i", select_range=(index, index))
    phi = np.asarray(vectors[:, 0], dtype=float)
    a = psi[1:-1]
    b = phi
    na = float(np.sqrt(h * np.sum(a**2)))
    nb = float(np.sqrt(h * np.sum(b**2)))
    if na == 0.0 or nb == 0.0:
        return None
    a = a / na
    b = b / nb
    if float(np.dot(a, b)) < 0.0:
        b = -b
    return float(np.sqrt(h * np.sum((a - b) ** 2)))


def is_leakage_free(path: str, meta: dict) -> bool:
    parts = Path(path).parts
    state = meta.get("state")
    if state is not None and state != 0:
        return False
    if meta.get("orthogonalize"):
        return False
    if any(part.startswith("certs5") for part in parts):
        # 基态、无 deflation，且残差中的 E 必须来自 ψ 本身：
        #   rayleigh —— E=当前 ψ 的 Rayleigh 商（推荐，无泄漏）
        #   learned  —— E 是可学习标量（无泄漏，但可能不收敛）
        # reference-E 组（certs5_leaked / certs5_refenergy）只用于泄漏对照。
        return meta.get("residual_energy") in ("learned", "rayleigh")
    if "certs" in parts and "learned0" in parts:
        return meta.get("residual_energy") in ("learned", "rayleigh")
    return False


def summarize(rows: list[dict]) -> dict:
    lf = []
    for row in rows:
        original = ROOT / row["path"]
        meta = load_meta(original)
        if not is_leakage_free(row["path"], meta):
            continue
        cert = row["filtered_certificate"]
        metrics = meta.get("metrics") or {}
        width = row.get("filtered_width")
        error = None
        if cert.get("energy") is not None and row.get("discrete_ref") is not None:
            error = abs(cert["energy"] - row["discrete_ref"])
        overlap = metrics.get("overlap")
        true_angle = float(np.arccos(min(1.0, max(-1.0, overlap)))) if overlap is not None else None
        disc_angle = None
        filtered_path = row.get("filtered_path")
        if filtered_path and cert.get("state_index") is not None:
            disc_angle = discrete_angle(ROOT / filtered_path, int(cert["state_index"]))
        lf.append({
            "path": row["path"],
            "potential": meta.get("potential"),
            "target_state": meta.get("state"),
            "cert_state": cert.get("state_index"),
            "certified": bool(cert.get("valid")) and row.get("state_matches_target") is True,
            "contains": bool(row.get("filtered_contains_ref")),
            "width": width,
            "energy_error": error,
            "error_near_roundoff": bool(error is not None and error < 1e-10),
            "width_over_error": (width / error) if width is not None and error else None,
            "angle_bound": row.get("filtered_angle"),
            "true_angle": disc_angle if disc_angle is not None else true_angle,
            "true_angle_analytic": true_angle,
            "rel_l2_change": row.get("rel_l2_change"),
            "raw_width": row.get("raw_width"),
            "raw_certified": bool(row["raw_certificate"].get("valid")),
            "overlap": overlap,
        })

    n = len(lf)
    n_cert = sum(1 for r in lf if r["certified"])
    n_contains = sum(1 for r in lf if r["contains"])
    widths = [r["width"] for r in lf if r["width"] is not None and r["contains"]]
    ratios = [r["width_over_error"] for r in lf
              if r["width_over_error"] is not None and r["contains"]]
    angles_ok = [
        r["angle_bound"] is not None and r["true_angle"] is not None
        and r["angle_bound"] + 1e-9 >= r["true_angle"]
        for r in lf
    ]
    return {
        "n_leakage_free": n,
        "n_certified": n_cert,
        "n_containment": n_contains,
        "containment_rate": (n_contains / n) if n else None,
        "median_filtered_width": float(np.median(widths)) if widths else None,
        "median_width_over_error": float(np.median(ratios)) if ratios else None,
        "max_width_over_error": float(np.max(ratios)) if ratios else None,
        "n_error_near_roundoff": sum(1 for r in lf if r["error_near_roundoff"]),
        "angle_bound_covers_true": sum(angles_ok),
        "runs": lf,
    }


def verdict(summary: dict) -> list[str]:
    lines = []
    rate = summary["containment_rate"]
    if rate is None:
        return ["没有可用的无泄漏 run：无法判定。"]
    lines.append(f"包含率 {summary['n_containment']}/{summary['n_leakage_free']} "
                 f"= {rate:.0%}")
    if rate < 0.90:
        lines.append("NO-GO：包含率低于 90%，离散/边界处理可能有系统性问题。")
    elif rate < 0.95:
        lines.append("NO-GO（临界）：包含率低于 §5.3 的 95% 门槛。")
    else:
        lines.append("包含率满足 Go 门槛（≥95%）。")
    ratio = summary["median_width_over_error"]
    if ratio is None:
        lines.append("没有同时可认证且包含参考值的 run，无法评估紧度。")
    elif ratio < 10:
        lines.append(f"GO：中位 宽度/误差 = {ratio:.2f} < 10。")
    else:
        worst = summary["max_width_over_error"]
        lines.append(f"紧度未达 Go 门槛：中位 宽度/误差 = {ratio:.1f}"
                     f"（最大 {worst:.3g}；Go 要求 <10，No-go 是 >1e3）。")
        lines.append("注意：误差本身接近舍入下限的 run（"
                     f"{summary['n_error_near_roundoff']} 个）会让比值失去意义，"
                     "应看绝对宽度与包含性。")
    good_angles = summary["angle_bound_covers_true"]
    if good_angles < summary["n_leakage_free"]:
        lines.append(f"夹角界覆盖离散本征向量的真实夹角：{good_angles}/"
                     f"{summary['n_leakage_free']}（未覆盖的 run 需单独检查）。")
    return lines


def markdown(summary: dict, verdict_lines: list[str]) -> str:
    lines = [
        "# A 线 go/no-go 汇总",
        "",
        f"* 无泄漏 run：{summary['n_leakage_free']}",
        f"* 滤波后证书有效且状态一致：{summary['n_certified']}",
        f"* 区间包含同一离散算子的参考本征值：{summary['n_containment']}",
        f"* 中位滤波后宽度：{summary['median_filtered_width']}",
        f"* 中位 宽度/|E_filt−λ_h|：{summary['median_width_over_error']}",
        f"* 夹角界覆盖真实 arccos(overlap)：{summary['angle_bound_covers_true']}/"
        f"{summary['n_leakage_free']}（比较对象为离散本征向量）",
        "",
        "## 判定",
        "",
        *[f"* {line}" for line in verdict_lines],
        "",
        "## 逐 run",
        "",
        "| run | potential | state | cert_state | certified | contains | width | "
        "width/|ΔE| | angle | angle(ψ,φ_h) | arccos(overlap_analytic) | rel L2 change |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["runs"]:
        def fmt(v, digits=3):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{digits}g}"
            return str(v)
        lines.append(
            f"| {r['path']} | {fmt(r['potential'])} | {fmt(r['target_state'])} | "
            f"{fmt(r['cert_state'])} | {'✓' if r['certified'] else '✗'} | "
            f"{'✓' if r['contains'] else '✗'} | {fmt(r['width'])} | "
            f"{fmt(r['width_over_error'])} | {fmt(r['angle_bound'])} | "
            f"{fmt(r['true_angle'])} | {fmt(r['true_angle_analytic'])} | "
            f"{fmt(r['rel_l2_change'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="A 线 go/no-go 汇总")
    parser.add_argument("--filtered", type=Path,
                        default=ROOT / "runs" / "certificates_filtered" / "summary.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "certificates")
    args = parser.parse_args()

    if not args.filtered.exists():
        print(f"缺少 {args.filtered}；先运行 scripts/filter_certificates.py")
        return 1
    rows = json.loads(args.filtered.read_text(encoding="utf-8"))
    summary = summarize(rows)
    lines = verdict(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "go_nogo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = markdown(summary, lines)
    (args.out_dir / "go_nogo.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
