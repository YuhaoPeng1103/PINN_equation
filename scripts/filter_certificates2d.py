"""二维证书引导滤波：对已保存的二维波函数做 DST 截断，再重新认证。

用法：
    python scripts/filter_certificates2d.py runs/seeds/certs6
    python scripts/filter_certificates2d.py path/to/wavefunction2d.npz

对每个 run 报告：原始残差/宽度（可能不可认证）、扫描得到的最优二维截断
阶数 (Kx,Ky)、滤波后的残差/宽度/夹角界、滤波后区间是否包含参考本征值，
以及把证书搬回网络原始输出的夹角上界（``transported_angle_bound_2d``）。

参考本征值来自 shift-invert Lanczos（ARPACK），与证书使用的稀疏 LU
惯性计数是**两个独立算法**；因此“区间包含参考值”是非循环的交叉检验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    approximate_spectrum_2d,
    certify_eigenpair_2d,
)
from nnschrodinger.solver.filtering2d import (  # noqa: E402
    guided_filter_2d,
    transported_angle_bound_2d,
)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _load_meta(path: Path) -> dict:
    candidate = path.with_name("exp6_certificate_training_2d.json")
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _true_angle(psi: np.ndarray, grid: Grid2D, index: int | None) -> float | None:
    """ψ 与 H_h 第 index 个**离散**本征向量的夹角（归一化差向量范数）。"""
    if index is None:
        return None
    try:
        from scipy.sparse.linalg import eigsh
        from nnschrodinger.solver.certify2d import _sparse_operator
    except ImportError:  # pragma: no cover
        return None
    operator = _sparse_operator(grid, 0.0)
    k = min(index + 1, max(1, grid.n_interior - 2))
    sigma = float(np.min(grid.potential)) - 1.0
    values, vectors = eigsh(operator, k=k, sigma=sigma, which="LM", tol=1e-10)
    # eigsh 不保证列顺序；按 Ritz 值排序后取第 index 个
    order = np.argsort(values)
    vec = np.asarray(vectors[:, order[index]], dtype=float).reshape(grid.shape[0] - 2,
                                                                    grid.shape[1] - 2)
    a = psi[1:-1, 1:-1]
    na = float(np.sqrt(grid.cell * np.sum(a**2)))
    nb = float(np.sqrt(grid.cell * np.sum(vec**2)))
    if na == 0.0 or nb == 0.0:
        return None
    a = a / na
    b = vec / nb
    if float(np.sum(a * b)) < 0.0:
        b = -b
    return float(np.sqrt(grid.cell * np.sum((a - b) ** 2)))


def process(path: Path, out_dir: Path) -> dict:
    data = np.load(path)
    grid = Grid2D(data["x"], data["y"], data["potential"])
    psi = np.asarray(data["psi"], dtype=float)

    approx = approximate_spectrum_2d(grid, 6)
    raw = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
    result = guided_filter_2d(psi, grid, approx=approx)
    cert = result.certificate

    ref = None
    if cert.state_index is not None and cert.state_index < approx.size:
        ref = float(approx[cert.state_index])
    contains = (
        ref is not None
        and cert.temple_lower is not None
        and cert.temple_upper is not None
        and cert.temple_lower <= ref <= cert.temple_upper
    )

    diff = psi - result.psi
    l2_change = float(np.sqrt(grid.cell * np.sum(diff**2)))
    raw_norm = float(np.sqrt(grid.cell * np.sum(psi**2)))
    rel_l2_change = (l2_change / raw_norm) if raw_norm else None

    transported = None
    if cert.eigenfunction_angle_bound is not None and raw_norm > 0.0:
        transported = transported_angle_bound_2d(
            psi, grid, result.psi, cert.eigenfunction_angle_bound
        )

    meta = _load_meta(path)
    # exp6 目前只训练基态；若将来加入激发态，JSON 里的 "state" 会覆盖默认值。
    target_state = int(meta.get("state", 0))
    state_matches_target = (
        None if cert.state_index is None else cert.state_index == target_state
    )

    rel = path.parent.relative_to(ROOT) if path.parent.is_relative_to(ROOT) else path.parent
    slug = "__".join(rel.parts)
    out_path = out_dir / f"{slug}.npz"
    np.savez(out_path, x=grid.x, y=grid.y, psi=result.psi, potential=grid.potential)

    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "filtered_path": str(out_path.relative_to(ROOT)) if out_path.is_relative_to(ROOT) else str(out_path),
        "potential": meta.get("potential"),
        "seed": meta.get("seed"),
        "n_grid": int(grid.shape[0]),
        "cutoff": list(result.cutoff),
        "reason": result.reason,
        "l2_change": l2_change,
        "rel_l2_change": rel_l2_change,
        "raw_certificate": raw.to_dict(),
        "raw_width": raw.width,
        "filtered_certificate": cert.to_dict(),
        "filtered_width": cert.width,
        "filtered_angle": cert.eigenfunction_angle_bound,
        "raw_angle_transported": transported,
        "raw_angle_true_discrete": _true_angle(psi, grid, cert.state_index),
        "discrete_ref": ref,
        "target_state": target_state,
        "state_matches_target": state_matches_target,
        "filtered_contains_ref": contains,
        "scan": list(result.scan),
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "# 二维证书（基态，81×81）",
        "",
        "| run | 势 | seed | (Kx,Ky) | E_raw | E_filt | r_raw | r_filt | "
        "width_raw | width_filt | angle_filt | angle_raw(搬回) | angle_raw(真实) | "
        "rel_change | contains | ref |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        raw = row["raw_certificate"]
        cert = row["filtered_certificate"]
        lines.append(
            "| {name} | {pot} | {seed} | {k} | {er} | {ef} | {rr} | {rf} | "
            "{wr} | {wf} | {ang} | {angr} | {angt} | {rel} | {cont} | {ref} |".format(
                name=Path(row["path"]).parent.as_posix(),
                pot=row.get("potential"),
                seed=row.get("seed"),
                k=tuple(row["cutoff"]),
                er=_fmt(raw.get("energy")),
                ef=_fmt(cert.get("energy")),
                rr=_fmt(raw.get("residual_norm")),
                rf=_fmt(cert.get("residual_norm")),
                wr=_fmt(row["raw_width"]),
                wf=_fmt(row["filtered_width"]),
                ang=_fmt(row["filtered_angle"]),
                angr=_fmt(row.get("raw_angle_transported")),
                angt=_fmt(row.get("raw_angle_true_discrete")),
                rel=_fmt(row.get("rel_l2_change")),
                cont="✓" if row["filtered_contains_ref"] else "✗",
                ref=_fmt(row["discrete_ref"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="二维证书引导谱滤波 + 重新认证")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="wavefunction2d.npz 或包含它的目录；缺省 runs/seeds/certs6")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "runs" / "certificates_filtered2d")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.paths or [ROOT / "runs" / "seeds" / "certs6"]:
        if p.is_dir():
            paths.extend(sorted(p.rglob("wavefunction2d.npz")))
        elif p.suffix == ".npz":
            paths.append(p)
    if not paths:
        print("没有找到 wavefunction2d.npz")
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
    print(f"{n_valid}/{len(rows)} 个滤波后证书有效；{n_contains}/{len(rows)} 个区间包含参考本征值")
    print(f"写出 {args.out_dir / 'summary.json'} 与 summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
