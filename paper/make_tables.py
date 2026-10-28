"""从 runs/ 的 JSON 汇总直接生成论文表格（避免手工转录出错）。

输出 paper/tables/*.tex，每个文件是一个 tabular，由 main.tex \\input。

用法：
    python paper/make_tables.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"

PRETTY = {
    "harmonic": r"Harmonic",
    "infinite_well": r"Infinite well",
    "finite_well": r"Finite well",
    "harmonic2d": r"Harmonic (2D)",
    "well2d": r"Box (2D)",
    "anharmonic2d": r"Anharmonic (2D)",
}
ORDER_1D = ("harmonic", "infinite_well", "finite_well")
ORDER_2D = ("harmonic2d", "well2d", "anharmonic2d")


def _load(*parts):
    return json.loads(ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def _sci(x: float, nd: int = 2) -> str:
    """2 位有效数字，返回可直接放进表格的数学模式；O(1) 的数字写成小数。"""
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    if -1 <= e <= 2:
        return f"${x:.{max(0, 3 - e - 1)}f}".rstrip("0").rstrip(".") + "$"
    v = x / 10**e
    return f"${v:.{nd - 1}f}\\times10^{{{e}}}$"


def _rng(vals, nd: int = 2) -> str:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return "--"
    lo, hi = min(vals), max(vals)
    if hi <= 0:
        return _sci(lo, nd)
    if hi - lo <= 1e-9 * hi:
        return _sci(lo, nd)
    if 0.1 <= lo and hi < 10:
        return f"{_sci(lo, nd)}--{_sci(hi, nd)}"
    elo = int(math.floor(math.log10(abs(lo)))) if lo > 0 else -99
    ehi = int(math.floor(math.log10(hi)))
    if elo == ehi:
        a, b = lo / 10**ehi, hi / 10**ehi
        return (f"$({a:.{nd - 1}f}\\text{{--}}{b:.{nd - 1}f})"
                f"\\times10^{{{ehi}}}$")
    return f"{_sci(lo, nd)}--{_sci(hi, nd)}"


def _potential(path: str) -> str:
    stem = path.replace("\\", "/").split("/")[-3]
    for key in ("infinite_well", "finite_well", "harmonic2d", "well2d",
                "anharmonic2d", "harmonic"):
        if stem.startswith(key):
            return key
    return stem


def _write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_1d() -> None:
    runs = _load("runs", "certificates_filtered", "certs5_rayleigh", "summary.json")
    by_pot: dict[str, list] = {}
    for r in runs:
        by_pot.setdefault(_potential(r["path"]), []).append(r)

    cols = {pot: by_pot[pot] for pot in ORDER_1D}
    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join([""] + [PRETTY[pot] for pot in ORDER_1D]) + r" \\",
        r"\midrule",
    ]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    row("raw cert.", [f"{sum(1 for r in cols[p] if r['raw_certificate']['valid'])}"
                      f"/{len(cols[p])}" for p in ORDER_1D])
    row(r"$|E_f-\lambda_h|$",
        [_rng([abs(r["filtered_certificate"]["energy"] - r["discrete_ref"])
               for r in cols[p]]) for p in ORDER_1D])
    row("width, raw",
        [_rng([r["raw_width"] for r in cols[p] if r["raw_width"] is not None])
         for p in ORDER_1D])
    row("width, filtered",
        [_rng([r["filtered_width"] for r in cols[p]]) for p in ORDER_1D])
    row("angle bound", [_rng([r["filtered_angle"] for r in cols[p]])
                        for p in ORDER_1D])
    row(r"contains $\lambda_h$",
        [f"{sum(1 for r in cols[p] if r['filtered_contains_ref'])}/{len(cols[p])}"
         for p in ORDER_1D])
    _write("tab_1d_ground.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_leakage() -> None:
    summary = _load("runs", "leakage_audit", "summary.json")["summary"]
    order = ["rayleigh", "learned", "reference", "deflated"]
    label = {
        "rayleigh": r"Rayleigh (no leak)",
        "learned": r"Learnable scalar $E_\theta$",
        "reference": r"Reference $E$ (leaky)",
        "deflated": r"Deflation (leaky)",
    }
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Residual energy & raw & filt.\ contains & med.\ $|\Delta E|$ & "
        r"med.\ width & max width \\",
        r"\midrule",
    ]
    for key in order:
        s = summary[key]
        lines.append(
            f"{label[key]} & {s['raw_certified']}/{s['n']} & "
            f"{s['filtered_certified']}/{s['n']} & {_sci(s['median_energy_error'])} & "
            f"{_sci(s['median_filtered_width'])} & {_sci(s['max_filtered_width'])} \\\\"
        )
    _write("tab_leakage.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_multistate() -> None:
    table = _load("runs", "multistate", "summary.json")["table"]
    rows = sorted(table.values(), key=lambda s: (s["potential"], s["state"]))
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Potential & $n$ & raw & match & contains $n$ & "
        r"med.\ $|\Delta E|$ & med.\ width \\",
        r"\midrule",
    ]
    for s in rows:
        lines.append(
            f"{PRETTY[s['potential']]} & {s['state']} & "
            f"{s['raw_certified']}/{s['n']} & {s['certified_state_ok']}/{s['n']} & "
            f"{s['contains_target']}/{s['n']} & {_sci(s['median_energy_error'])} & "
            f"{_sci(s['median_filtered_width'])} \\\\"
        )
    _write("tab_multistate.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_2d() -> None:
    runs = _load("runs", "certificates_filtered2d", "summary.json")
    by_pot: dict[str, list] = {}
    for r in runs:
        by_pot.setdefault(r["potential"], []).append(r)

    # 参考值优先用稠密 LAPACK（scripts/dense_reference_2d.py 生成）：
    # summary.json 里的 discrete_ref 是 ARPACK 的近似本征值，而 ARPACK 同时
    # 用于提议区间，用它当参考会有循环论证的嫌疑。两者在前十位一致。
    dense_path = ROOT / "runs" / "reference2d" / "dense_ref.json"
    dense_ref: dict[str, float] = {}
    if dense_path.exists():
        report = json.loads(dense_path.read_text(encoding="utf-8"))
        for pot, block in report["problems"].items():
            for row in block["runs"]:
                dense_ref[row["path"]] = row["dense"]

    def ref(r: dict) -> float:
        return float(dense_ref.get(r["path"], r["discrete_ref"]))

    cols = {pot: by_pot[pot] for pot in ORDER_2D}
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join([""] + [PRETTY[pot] for pot in ORDER_2D]) + r" \\",
        r"\midrule",
    ]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    def cut(pot: str) -> str:
        ks = sorted({tuple(r["cutoff"]) for r in cols[pot]})
        return f"$({ks[0][0]},{ks[0][1]})$" if len(ks) == 1 else "various"

    row("cutoff $(K_x,K_y)$", [cut(p) for p in ORDER_2D])
    row("raw cert.", [f"{sum(1 for r in cols[p] if r['raw_certificate']['valid'])}"
                      f"/{len(cols[p])}" for p in ORDER_2D])
    row(r"$|E_f-\lambda_h|$",
        [_rng([abs(r["filtered_certificate"]["energy"] - ref(r))
               for r in cols[p]]) for p in ORDER_2D])
    row("width, raw", [_rng([r["raw_width"] for r in cols[p]])
                       for p in ORDER_2D])
    row("width, filtered", [_rng([r["filtered_width"] for r in cols[p]])
                            for p in ORDER_2D])
    row("angle bound", [_rng([r["filtered_angle"] for r in cols[p]])
                        for p in ORDER_2D])
    # 包含性也按稠密参考判断（summary.json 的 filtered_contains_ref 用的是
    # ARPACK 值，而 ARPACK 参与提议区间，不应再作判据）。
    def contains(r: dict) -> bool:
        c = r["filtered_certificate"]
        lo, hi = c.get("temple_lower"), c.get("temple_upper")
        return lo is not None and hi is not None and lo <= ref(r) <= hi

    row(r"contains $\lambda_h$",
        [f"{sum(1 for r in cols[p] if contains(r))}/{len(cols[p])}"
         for p in ORDER_2D])

    # 记录派生检查，供正文引用（runs/reference2d/filtered_containment.json）
    derived = {
        "reference": "dense LAPACK (scripts/dense_reference_2d.py)",
        "rows": [
            {
                "path": r["path"],
                "potential": r["potential"],
                "dense": ref(r),
                "filtered_lower": r["filtered_certificate"].get("temple_lower"),
                "filtered_upper": r["filtered_certificate"].get("temple_upper"),
                "contains": contains(r),
            }
            for pot in ORDER_2D for r in cols[pot]
        ],
    }
    out = ROOT / "runs" / "reference2d" / "filtered_containment.json"
    out.write_text(json.dumps(derived, indent=2), encoding="utf-8")
    _write("tab_2d.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_degeneracy() -> None:
    s = _load("runs", "degeneracy2d", "summary.json")
    names = {
        "ground": r"Ground state",
        "excited_a": r"Excited (ARPACK 1)",
        "excited_b": r"Excited (ARPACK 2)",
        "excited_mix": r"Excited (mixture)",
    }
    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Input state & $E$ & $r$ & certified & width & outcome \\",
        r"\midrule",
    ]
    for row in s["cases"]:
        outcome = "accepted" if row["valid"] else "refused"
        width = _sci(row['temple_width']) if row["temple_width"] else "--"
        lines.append(
            f"{names.get(row['case'], row['case'])} & {row['energy']:.6f} & "
            f"{_sci(row['residual_norm'])} & {'yes' if row['valid'] else 'no'} & "
            f"{width} & {outcome} \\\\"
        )
    _write("tab_degeneracy.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_subspace() -> None:
    s = _load("runs", "degeneracy2d", "subspace.json")
    order = [
        "exact_pair", "exact_rotated", "exact_single", "exact_skipping",
    ]
    by_name = {r["case"]: r for r in s["rows"]}
    neural = [
        r["case"] for r in s["rows"]
        if r["case"].startswith("neural_pair") or r["case"].startswith("neural_mixed")
    ]
    # 强制第二方向的案例放在最后，和“独立 run 共线”的拒绝形成对照
    neural.sort(key=lambda name: ("deflated" in name, "mixed" in name, name))
    label = {
        "exact_pair": r"Exact pair",
        "exact_rotated": r"Exact pair, rotated basis",
        "exact_single": r"One member ($m=1$)",
        "exact_skipping": r"Ground $+$ 4th level",
    }
    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Input & $m$ & $N(b)\!-\!N(a)$ & $\rho_F$ & $\delta$ & "
        r"$\Theta\le$ & $\Theta_{\mathrm{true}}$ & outcome \\",
        r"\midrule",
    ]

    def row(name: str, text: str) -> None:
        r = by_name[name]
        if "ritz_values" in r:
            count = r["verified_multiplicity"]
            count = "--" if count is None else count
            rho = _sci(r["frobenius_residual"])
            delta = _sci(r["margin"]) if r["margin"] is not None else "--"
            bound = _sci(r["angle_bound"]) if r["angle_bound"] is not None else "--"
            true = _sci(r["true_angle"]) if r.get("true_angle") is not None else "--"
        else:
            count = rho = delta = bound = true = "--"
        outcome = "accepted" if r["valid"] else "refused"
        lines.append(
            f"{text} & {r['n_states']} & {count} & {rho} & {delta} & {bound} & "
            f"{true} & {outcome} \\\\"
        )

    for name in order:
        row(name, label[name])
    if neural:
        lines.append(r"\midrule")
    for name in neural:
        parts = name.split("_")
        if name == "neural_pair_seed_deflated":
            text = "Network $+$ partner"
        elif name == "neural_mixed_seed_deflated":
            text = "Network $+$ mix, partner"
        elif name.startswith("neural_pair"):
            a, b = parts[-2].replace("seed", ""), parts[-1]
            text = f"Networks {a}, {b}"
        else:
            a, b = parts[-2].replace("seed", ""), parts[-1]
            text = f"Seeds {a}, {b}, rotated"
        row(name, text)
    _write("tab_subspace.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_resolution() -> None:
    s = _load("runs", "resolution2d", "boundary.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$\lambda$ & split $|E_1-E_2|$ & floor & split/floor & single state & "
        r"$\delta$ & $\Theta\le$ \\",
        r"\midrule",
    ]
    for r in s["rows"]:
        lam = r["lambda"]
        lam_tex = "$0$" if lam == 0 else _sci(lam)
        split = "$<10^{-12}$" if lam == 0 else _sci(r["split"])
        ratio = r["split"] / r["floor"]
        ratio_tex = "$<0.01$" if ratio < 0.01 else f"${ratio:.2f}$"
        c = r["cluster"]
        delta = _sci(c["margin"]) if c.get("margin") is not None else "--"
        bound = _sci(c["angle_bound"]) if c.get("angle_bound") is not None else "--"
        outcome = "accepted" if r["single_accepted"] else "refused"
        lines.append(
            f"{lam_tex} & {split} & {_sci(r['floor'])} & {ratio_tex} & {outcome} & "
            f"{delta} & {bound} \\\\"
        )
    _write("tab_resolution.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_discretization() -> None:
    disc = _load("runs", "discretization2d", "summary.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Pot.\ & $41$ & $61$ & $81$ & $101$ & $121$ & $\lambda^\ast$ \\",
        r"\midrule",
    ]
    for name in ORDER_2D[:2]:
        rows = sorted(disc["problems"][name], key=lambda r: r["n"])
        errs = " & ".join(_sci(r['error_vs_lambda_star']) for r in rows)
        star = rows[0]["lambda_star"]
        lines.append(f"{PRETTY[name]} & {errs} & {star:.7f} \\\\")
    _write("tab_discretization.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def table_continuous() -> None:
    """连续算子的一维/二维夹逼（runs/continuous_bound/bound.json）。"""
    data = _load("runs", "continuous_bound", "bound.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}llrrll@{}}",
        r"\toprule",
        r"Potential & $N$ & $\lambda_h$ & $R(\psi_h)$ & $\lambda^\star$ & in \\",
        r"\midrule",
    ]
    for key, rows, coarse, fine in (
        ("rows_1d", None, 501, 4001),
        ("rows_2d", None, 41, 101),
    ):
        for name in ORDER_2D if key == "rows_2d" else ("harmonic", "infinite_well", "finite_well"):
            selected = [r for r in data[key] if r["potential"] == name and r["N"] in (coarse, fine)]
            for row in sorted(selected, key=lambda r: r["N"]):
                if "discrete_lo" in row:
                    discrete = 0.5 * (row["discrete_lo"] + row["discrete_hi"])
                else:
                    discrete = row["discrete"]
                reference = row["reference"]
                ref = "--" if reference is None else f"{reference:.10f}"
                contains = row["contains_reference"]
                mark = "--" if contains is None else (r"\checkmark" if contains else r"$\times$")
                lines.append(
                    f"{PRETTY[name]} & {row['N']} & {discrete:.10f} & {row['upper']:.10f} & "
                    f"{ref} & {mark} \\\\"
                )
        if key == "rows_1d":
            lines.append(r"\midrule")
    _write("tab_continuous.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table_1d()
    table_leakage()
    table_multistate()
    table_2d()
    table_degeneracy()
    table_subspace()
    table_resolution()
    table_discretization()
    table_continuous()
    for p in sorted(OUT.glob("*.tex")):
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
