"""证书引导的谱滤波：对已保存的波函数做 DST 截断，再重新认证。

用法：
    python scripts/filter_certificates.py runs/seeds/certs
    python scripts/filter_certificates.py path/to/wavefunction.npz

对每个 run 报告：原始残差/宽度（可能不可认证）、扫描得到的最优截断阶数、
滤波后的残差/宽度/夹角界，以及滤波后区间是否包含同一离散 H_h 的参考本征值。
输出 ``runs/certificates_filtered/summary.{json,md}``，并把滤波后的波函数
另存为 ``<out>/<slug>.npz``，便于复核。
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

from nnschrodinger.solver.certify import certify_eigenpair  # noqa: E402
from nnschrodinger.solver.filtering import guided_filter, transported_angle_bound  # noqa: E402
from certify_saved import independent_discrete_ref  # noqa: E402


def _discrete_angle(psi: np.ndarray, x: np.ndarray, potential: np.ndarray,
                    index: int | None) -> float | None:
    """ψ 与 H_h 第 ``index`` 个本征向量的夹角（用归一化差向量的范数）。

    比较对象必须是**离散**本征向量：H_h 与连续算子的本征函数相差 O(h²)，
    这个差比滤波后的夹角界大得多，拿解析解去比会把“界成立”误判成违反。
    """
    if index is None:
        return None
    try:
        from scipy.linalg import eigh_tridiagonal
    except ImportError:  # pragma: no cover
        return None
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    _, vectors = eigh_tridiagonal(diag, off, select="i", select_range=(index, index))
    a = np.asarray(psi, dtype=float)[1:-1]
    b = np.asarray(vectors[:, 0], dtype=float)
    na = float(np.sqrt(h * np.sum(a**2)))
    nb = float(np.sqrt(h * np.sum(b**2)))
    if na == 0.0 or nb == 0.0:
        return None
    a = a / na
    b = b / nb
    if float(np.dot(a, b)) < 0.0:
        b = -b
    return float(np.sqrt(h * np.sum((a - b) ** 2)))


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def process(path: Path, out_dir: Path) -> dict:
    data = np.load(path)
    x = np.asarray(data["x"], dtype=float)
    psi = np.asarray(data["psi"], dtype=float)
    potential = np.asarray(data["potential"], dtype=float)

    raw = certify_eigenpair(psi, x, potential, isolate=True)
    result = guided_filter(psi, x, potential)
    cert = result.certificate

    ref = None
    if cert.state_index is not None:
        try:
            ref, _ = independent_discrete_ref(x, potential, cert.state_index)
        except ValueError:
            ref = None
    contains = (
        ref is not None
        and cert.temple_lower is not None
        and cert.temple_upper is not None
        and cert.temple_lower <= ref <= cert.temple_upper
    )

    # 滤波改变了多少：证书认证的是 ψ_K，必须报告它离网络原始输出有多远。
    h = float(x[1] - x[0])
    diff = psi - result.psi
    l2_change = float(np.sqrt(h * np.sum(diff**2)))
    raw_norm = float(np.sqrt(h * np.sum(psi**2)))
    rel_l2_change = (l2_change / raw_norm) if raw_norm else None

    # 把滤波函数的证书“搬回”网络原始输出（初等三角不等式，见
    # docs/certification.md §5.6.1）。记 u=ψ/‖ψ‖、u_K=ψ_K/‖ψ_K‖、φ 为目标
    # 本征函数：
    #   ‖u − (u·φ)φ‖ ≤ 2‖u−u_K‖ + ‖u_K − (u_K·φ)φ‖
    # 右端第一项直接由两个已保存的函数算出（精确值，不用 δ 放大），
    # 第二项来自滤波后证书的 Davis–Kahan 界。这样即使原始 ψ 不可认证，
    # 也能给出它到真实本征函数的夹角上界——证书约束的是网络输出本身。
    transported = None
    if cert.eigenfunction_angle_bound is not None and raw_norm > 0.0:
        transported = transported_angle_bound(
            psi, x, result.psi, cert.eigenfunction_angle_bound
        )

    target_state = None
    for meta_name in ("exp2_pinn.json", "exp5_certificate_training.json"):
        meta_path = path.with_name(meta_name)
        if meta_path.exists():
            target_state = json.loads(meta_path.read_text(encoding="utf-8")).get("state")
            break
    state_matches_target = (
        None if target_state is None or cert.state_index is None
        else cert.state_index == target_state
    )

    rel = path.parent.relative_to(ROOT) if path.parent.is_relative_to(ROOT) else path.parent
    slug = "__".join(rel.parts)
    out_path = out_dir / f"{slug}.npz"
    np.savez(out_path, x=x, psi=result.psi, potential=potential)

    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "filtered_path": str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
        "cutoff": result.cutoff,
        "reason": result.reason,
        "l2_change": l2_change,
        "rel_l2_change": rel_l2_change,
        "raw_certificate": raw.to_dict(),
        "raw_width": raw.width,
        "filtered_certificate": cert.to_dict(),
        "filtered_width": cert.width,
        "filtered_angle": cert.eigenfunction_angle_bound,
        # 网络原始输出（未滤波）的夹角上界：由滤波位移 + 滤波后证书得到。
        "raw_angle_transported": transported,
        # 真实夹角（到同一离散算子的本征向量），用于检验上面的界是否成立。
        "raw_angle_true_discrete": _discrete_angle(psi, x, potential, cert.state_index),
        "discrete_ref": ref,
        "target_state": target_state,
        "state_matches_target": state_matches_target,
        "filtered_contains_ref": contains,
        "scan": list(result.scan),
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "| run | K | E_raw | E_filt | r_raw | r_filt | width_raw | width_filt | angle_filt | "
        "angle_raw(搬回) | angle_raw(真实) | L2_change | rel_change | contains | ref |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        raw = row["raw_certificate"]
        cert = row["filtered_certificate"]
        lines.append(
            "| {name} | {k} | {er} | {ef} | {rr} | {rf} | {wr} | {wf} | {ang} | {angr} | {angt} | {l2} | {rel} | {cont} | {ref} |".format(
                name=Path(row["path"]).parent.as_posix(),
                k=row["cutoff"],
                er=_fmt(raw.get("energy")),
                ef=_fmt(cert.get("energy")),
                rr=_fmt(raw.get("residual_norm")),
                rf=_fmt(cert.get("residual_norm")),
                wr=_fmt(row["raw_width"]),
                wf=_fmt(row["filtered_width"]),
                ang=_fmt(row["filtered_angle"]),
                angr=_fmt(row.get("raw_angle_transported")),
                angt=_fmt(row.get("raw_angle_true_discrete")),
                l2=_fmt(row.get("l2_change")),
                rel=_fmt(row.get("rel_l2_change")),
                cont="✓" if row["filtered_contains_ref"] else "✗",
                ref=_fmt(row["discrete_ref"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="证书引导谱滤波 + 重新认证")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="wavefunction.npz 或包含它的目录；缺省扫 runs/seeds/certs")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "runs" / "certificates_filtered")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.paths or [ROOT / "runs" / "seeds" / "certs"]:
        if p.is_dir():
            paths.extend(sorted(p.rglob("wavefunction.npz")))
        elif p.suffix == ".npz":
            paths.append(p)
    if not paths:
        print("没有找到 wavefunction.npz")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [process(p, args.out_dir) for p in paths]
    (args.out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = markdown(rows)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)

    n_valid = sum(1 for r in rows if r["filtered_certificate"]["valid"])
    n_contains = sum(1 for r in rows if r["filtered_contains_ref"])
    gain = [
        r["raw_certificate"]["residual_norm"] / r["filtered_certificate"]["residual_norm"]
        for r in rows
        if r["filtered_certificate"]["residual_norm"]
    ]
    if gain:
        print(f"残差中位改善 {np.median(gain):.1f}×；", end="")
    print(f"{n_valid}/{len(rows)} 个滤波后证书有效；{n_contains}/{len(rows)} 个区间包含参考本征值")
    print(f"写出 {args.out_dir / 'summary.json'} 与 summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
