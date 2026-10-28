"""对保存的 ``wavefunction.npz`` 计算后验证书，并汇总成 JSON/Markdown。

用法：
    python scripts/certify_saved.py                       # 扫 runs/seeds/certs
    python scripts/certify_saved.py path/to/wavefunction.npz ...
    python scripts/certify_saved.py runs/seeds/certs/learned0/seed0

证书的对象是**离散** H_h（二阶差分 + Dirichlet 边界），因此参考值也用
``discrete_eigenvalue`` 在**同一张网格**上算，而不是连续解析能量。
连续 Rayleigh 商只作报告用；两者相差 O(h²)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify import (  # noqa: E402
    certify_eigenpair,
    continuous_rayleigh_quotient,
    discrete_eigenvalue,
)

try:  # 独立参考：LAPACK 三对角特征值，刻意不用本项目的 Sturm 计数
    from scipy.linalg import eigh_tridiagonal
except ImportError:  # pragma: no cover - 环境没有 scipy 时退回 Sturm
    eigh_tridiagonal = None


def independent_discrete_ref(x: np.ndarray, potential: np.ndarray, index: int) -> tuple[float, str]:
    """用与 Sturm 计数无关的方法算 H_h 的第 index 个本征值。

    **不要直接返回 LAPACK 的本征值**：对称三对角求解器的绝对误差约为
    ``eps·‖H_h‖``。h=1/2000 时 ‖H_h‖≈1.6e7，误差可达 1e-9——比滤波后
    证书的宽度（~1e-10）还大，会把“区间包含参考值”误判成失败。

    改用它返回的**本征向量**做 Rayleigh 商：向量的角度误差是
    ``O(eps·‖H‖/gap)``，Rayleigh 商的误差是这个量的平方再乘 gap，
    通常低 10 个数量级以上。实测该值与解析离散值的差 ~2e-12。
    """
    if eigh_tridiagonal is None:
        return discrete_eigenvalue(x, potential, index), "sturm"
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    _, vectors = eigh_tridiagonal(
        diag, off, select="i", select_range=(index, index)
    )
    v = np.asarray(vectors[:, 0], dtype=float)
    h_v = diag * v
    h_v[1:] += off * v[:-1]
    h_v[:-1] += off * v[1:]
    value = float(np.dot(v, h_v) / np.dot(v, v))
    return value, "scipy_eigh_tridiagonal_rayleigh"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def process(path: Path) -> dict:
    data = np.load(path)
    x = np.asarray(data["x"], dtype=float)
    psi = np.asarray(data["psi"], dtype=float)
    potential = np.asarray(data["potential"], dtype=float)

    cert = certify_eigenpair(psi, x, potential, isolate=True)
    ref_disc = None
    ref_method = None
    ref_sturm = None
    if cert.state_index is not None:
        try:
            ref_disc, ref_method = independent_discrete_ref(x, potential, cert.state_index)
            ref_sturm = discrete_eigenvalue(x, potential, cert.state_index)
        except ValueError:
            ref_disc = None

    row: dict = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "n_points": int(x.size),
        "dx": float(x[1] - x[0]),
        "certificate": cert.to_dict(),
        "continuous_rayleigh": continuous_rayleigh_quotient(psi, x, potential),
        "discrete_ref": ref_disc,
        "discrete_ref_method": ref_method,
        "discrete_ref_sturm": ref_sturm,
        "sturm_vs_independent_gap": (
            None if ref_disc is None or ref_sturm is None else abs(ref_disc - ref_sturm)
        ),
        "contains_discrete_ref": (
            ref_disc is not None
            and cert.temple_lower is not None
            and cert.temple_upper is not None
            and cert.temple_lower <= ref_disc <= cert.temple_upper
        ),
        "state_matches_target": None,
    }

    for meta_name in ("exp2_pinn.json", "exp5_certificate_training.json"):
        meta_path = path.with_name(meta_name)
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target = meta.get("state")
        row["state_matches_target"] = (
            None if target is None or cert.state_index is None else cert.state_index == target
        )
        row.update({
            "meta_file": meta_name,
            "potential": meta.get("potential"),
            "target_state": meta.get("state"),
            "seed": meta.get("seed"),
            "residual_energy": meta.get("residual_energy"),
            "orthogonalize": meta.get("orthogonalize"),
            "learned_energy": meta.get("learned_energy", meta.get("learned_energy_value")),
            "metrics": meta.get("metrics"),
        })
        break
    return row


def markdown(rows: list[dict]) -> str:
    lines = [
        "| run | potential | target | cert_state | E_cert | E_disc_ref | Δsturm | width | angle_bound | contains | residual | overlap | E_err | width/E_err |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cert = row["certificate"]
        name = Path(row["path"]).parent.as_posix()
        width = None
        if cert.get("temple_lower") is not None and cert.get("temple_upper") is not None:
            width = cert["temple_upper"] - cert["temple_lower"]
        metrics = row.get("metrics") or {}
        err = metrics.get("energy_abs_err")
        ratio = None
        if width is not None and err:
            ratio = width / err
        lines.append(
            "| {name} | {pot} | {target} | {state} | {e} | {ref} | {dg} | {w} | {ang} | {contains} | {r} | {ov} | {err} | {ratio} |".format(
                name=name,
                pot=_fmt(row.get("potential")),
                target=_fmt(row.get("target_state")),
                state=_fmt(cert.get("state_index")),
                e=_fmt(cert.get("energy")),
                ref=_fmt(row.get("discrete_ref")),
                dg=_fmt(row.get("sturm_vs_independent_gap")),
                w=_fmt(width),
                ang=_fmt(cert.get("eigenfunction_angle_bound")),
                contains="✓" if row["contains_discrete_ref"] else "✗",
                r=_fmt(cert.get("residual_norm")),
                ov=_fmt(metrics.get("overlap")),
                err=_fmt(err),
                ratio=_fmt(ratio),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="对保存的波函数计算后验证书")
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="wavefunction.npz 或包含它的目录；缺省扫 runs/seeds/certs",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "certificates")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.paths or [ROOT / "runs" / "seeds" / "certs"]:
        if p.is_dir():
            paths.extend(sorted(p.rglob("wavefunction.npz")))
        elif p.suffix == ".npz":
            paths.append(p)
        else:
            paths.extend(sorted(p.rglob("wavefunction.npz")))
    if not paths:
        print("没有找到 wavefunction.npz")
        return 1

    rows = [process(p) for p in paths]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = markdown(rows)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    n_valid = sum(1 for r in rows if r["certificate"]["valid"])
    n_contains = sum(1 for r in rows if r["contains_discrete_ref"])
    ratios = []
    for r in rows:
        cert = r["certificate"]
        if (cert.get("temple_lower") is None or cert.get("temple_upper") is None
                or not r.get("metrics") or not r["metrics"].get("energy_abs_err")):
            continue
        width = cert["temple_upper"] - cert["temple_lower"]
        ratios.append(width / r["metrics"]["energy_abs_err"])
    print(f"{n_valid}/{len(rows)} 个证书有效；{n_contains}/{len(rows)} 个区间包含同一离散算子的参考本征值")
    if ratios:
        print(f"宽度/真实能量误差 的中位数 {np.median(ratios):.1f}（范围 {min(ratios):.1f}–{max(ratios):.1f}）")
    print(f"写出 {args.out_dir / 'summary.json'} 与 summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
