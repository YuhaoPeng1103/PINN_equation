"""实验八：二维简并簇的子空间证书。

实验七表明：二维方阱的 (1,2)/(2,1) 二重简并态**无法**用单态 Temple 界
认证——不存在只含一个本征值的可验证区间，证书必须拒绝。本实验把同一个
拒绝变成正面结论：认证对象从 ψ 换成 span(ψ₁,…,ψ_m)。

做法
----
1. 用已验证的惯性计数测出重数 m（不是网络的输入）；
2. 用近似谱提案、计数验证得到簇区间 (a,b)，即 m 个本征值的双侧围道；
3. Rayleigh–Ritz 给出 θ_i、ρ_i、ρ_F；
4. 由 δ = min_i min(θ_i−a, b−θ_i) 得主子角界 sin Θ ≤ ρ_F/δ。

对照
----
* 精确离散本征向量（ARPACK）：证明拒绝不是“训练不好”的产物；
* 同一子空间的不同基（旋转/混合）：证明证书只依赖 span；
* 单个简并成员（m=1）：仍然拒绝——这不是失败，是问题本身不适定；
* 跨簇 / 跳过中间能级的子空间：拒绝（重数不符），说明证书不会
  把“网络漏了一个态”当成成功；
* 两个**独立训练**的网络输出：网络各自落到简并子空间的任意元素，
  残差 ~1e-2，但子空间可被认证。

用法：
    python scripts/exp8_subspace_certificate.py --n-grid 81 --seeds 0 1 2 3
输出 runs/degeneracy2d/subspace.{json,md}
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
    _sparse_operator,
    approximate_spectrum_2d,
    certify_eigenpair_2d,
    discrete_eigenvalue_2d,
)
from nnschrodinger.solver.certify_subspace import (  # noqa: E402
    certify_subspace_2d,
    principal_angle,
)


def _discrete_vectors(grid: Grid2D, k: int):
    from scipy.sparse.linalg import eigsh

    operator = _sparse_operator(grid, 0.0)
    sigma = float(np.min(grid.potential)) - 1.0
    values, vectors = eigsh(operator, k=k, sigma=sigma, which="LM", tol=1e-12)
    order = np.argsort(values)
    return np.asarray(values)[order], np.asarray(vectors)[:, order]


def _embed(vec: np.ndarray, grid: Grid2D) -> np.ndarray:
    psi = np.zeros(grid.shape)
    psi[1:-1, 1:-1] = vec.reshape(grid.shape[0] - 2, grid.shape[1] - 2)
    return psi


def _normalise(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    psi = np.array(psi, dtype=float, copy=True)
    psi[0, :] = psi[-1, :] = 0.0
    psi[:, 0] = psi[:, -1] = 0.0
    norm = np.sqrt(grid.cell * np.sum(psi[1:-1, 1:-1] ** 2))
    return psi / norm


def _in_subspace(psi: np.ndarray, basis: list[np.ndarray], grid: Grid2D) -> float:
    """‖P_E ψ‖/‖ψ‖：网络输出落在精确简并子空间里的比例（离散加权范数）。"""
    w = grid.cell

    def flat(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.shape == grid.shape:
            v = v[1:-1, 1:-1]
        return v.reshape(-1)

    E = np.stack([flat(v) for v in basis], axis=1)
    q = flat(psi)
    coeff = np.linalg.solve(w * (E.T @ E), w * (E.T @ q))
    proj = E @ coeff
    return float(
        np.sqrt(w * np.sum(proj**2)) / np.sqrt(w * np.sum(q**2))
    )


def _row(name: str, cert, true_angle: float | None, extra: dict | None = None) -> dict:
    row = {
        "case": name,
        "n_states": cert.n_states,
        "verified_multiplicity": cert.verified_multiplicity,
        "first_index": cert.first_index,
        "ritz_values": list(cert.ritz_values),
        "residual_norms": list(cert.residual_norms),
        "frobenius_residual": cert.frobenius_residual,
        "interval": (
            [cert.interval_lower, cert.interval_upper]
            if cert.interval_lower is not None
            else None
        ),
        "margin": cert.margin,
        "angle_bound": cert.angle_bound,
        "true_angle": true_angle,
        "valid": cert.valid,
        "reason": cert.reason,
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="二维简并簇的子空间证书")
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument(
        "--neural-root", type=Path, default=ROOT / "runs" / "degeneracy2d" / "neural"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "degeneracy2d")
    parser.add_argument(
        "--partner", type=Path, default=None,
        help="对基态与 seed0 同时 deflation 训练出的第二方向（wavefunction2d.npz）",
    )
    parser.add_argument("--skip-slow-refs", action="store_true")
    args = parser.parse_args()

    n = int(args.n_grid)
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))
    values, vectors = _discrete_vectors(grid, 6)
    approx = approximate_spectrum_2d(grid, 8)

    exact_pair = [_embed(vectors[:, 1], grid), _embed(vectors[:, 2], grid)]
    exact_pair = [_normalise(v, grid) for v in exact_pair]
    ground = _normalise(_embed(vectors[:, 0], grid), grid)
    third = _normalise(_embed(vectors[:, 3], grid), grid)

    rows: list[dict] = []

    cases: dict[str, list[np.ndarray]] = {
        "exact_pair": exact_pair,
        "exact_rotated": [exact_pair[0], 0.6 * exact_pair[0] + 0.8 * exact_pair[1]],
        "exact_single": [exact_pair[0]],
        "exact_skipping": [ground, third],
    }
    for name, basis in cases.items():
        cert = certify_subspace_2d(basis, grid, approx=approx)
        true_angle = None
        if cert.valid and name != "exact_skipping":
            true_angle = principal_angle(basis, exact_pair, grid.cell)
        rows.append(_row(name, cert, true_angle))

    # 单个简并成员的单态证书：仍然拒绝（对照）
    single = certify_eigenpair_2d(exact_pair[0], grid, isolate=True, approx=approx)
    rows.append({
        "case": "exact_single_state_temple",
        "n_states": 1,
        "valid": single.valid,
        "reason": single.reason,
        "temple_width": single.width,
        "angle_bound": single.eigenfunction_angle_bound,
    })

    # ---- 神经网络输出 -------------------------------------------------------
    neural: dict[int, np.ndarray] = {}
    for seed in args.seeds:
        path = args.neural_root / f"seed{seed}" / "wavefunction2d.npz"
        if not path.exists():
            print(f"[exp8] 跳过 seed{seed}（缺少 {path}）")
            continue
        data = np.load(path)
        if not (np.allclose(data["x"], x) and np.allclose(data["y"], x)):
            raise ValueError(f"{path} 的网格与当前网格不一致")
        psi = _normalise(data["psi"], grid)
        neural[seed] = psi
        single = certify_eigenpair_2d(psi, grid, isolate=True, approx=approx)
        rows.append({
            "case": f"neural_seed{seed}_temple",
            "n_states": 1,
            "valid": single.valid,
            "reason": single.reason,
            "temple_width": single.width,
            "energy": single.energy,
            "residual": single.residual_norm,
            "in_subspace": _in_subspace(psi, exact_pair, grid),
        })

    seeds = sorted(neural)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            basis = [neural[a], neural[b]]
            cert = certify_subspace_2d(basis, grid, approx=approx)
            true_angle = (
                principal_angle(basis, exact_pair, grid.cell) if cert.valid else None
            )
            rows.append(_row(
                f"neural_pair_seed{a}_{b}", cert, true_angle,
                {
                    "in_subspace": [
                        _in_subspace(neural[a], exact_pair, grid),
                        _in_subspace(neural[b], exact_pair, grid),
                    ]
                },
            ))
    # 混合（旋转后再认证）：一个共线的对（0,1）和一个张成的对（0,最后一个），
    # 说明证书只依赖 span。
    mixture_pairs = []
    if len(seeds) >= 2:
        mixture_pairs.append((seeds[0], seeds[1]))
    if len(seeds) >= 3:
        mixture_pairs.append((seeds[0], seeds[-1]))
    for a, b in mixture_pairs:
        mixed = 0.6 * neural[a] + 0.8 * neural[b]
        cert = certify_subspace_2d([neural[a], mixed], grid, approx=approx)
        true_angle = (
            principal_angle([neural[a], mixed], exact_pair, grid.cell)
            if cert.valid else None
        )
        rows.append(_row(f"neural_mixed_seed{a}_{b}", cert, true_angle))

    # 两个独立 run 的向量如果几乎共线，certify_subspace_2d 会以秩亏拒绝；
    # 记录 |⟨ψ_a, ψ_b⟩| 让读者看到拒绝的原因，而不是只看到 valid=False。
    overlaps = {
        f"seed{a}_seed{b}": abs(float(grid.cell * np.sum(neural[a] * neural[b])))
        for i, a in enumerate(seeds) for b in seeds[i + 1:]
    }

    # ---- 强制第二方向的网络：对基态**和** seed0 输出同时 deflation ----------
    partner_path = args.partner
    if partner_path is None:
        partner_path = args.neural_root.parent / "neural_partner" / "seed0" / "wavefunction2d.npz"
    if partner_path.exists() and seeds:
        data = np.load(partner_path)
        if not (np.allclose(data["x"], x) and np.allclose(data["y"], x)):
            raise ValueError(f"{partner_path} 的网格与当前网格不一致")
        partner = _normalise(data["psi"], grid)
        first = seeds[0]
        single = certify_eigenpair_2d(partner, grid, isolate=True, approx=approx)
        rows.append({
            "case": "neural_partner_temple",
            "n_states": 1,
            "valid": single.valid,
            "reason": single.reason,
            "temple_width": single.width,
            "energy": single.energy,
            "residual": single.residual_norm,
            "in_subspace": _in_subspace(partner, exact_pair, grid),
            "overlap_with_seed": abs(float(grid.cell * np.sum(partner * neural[first]))),
        })
        basis = [neural[first], partner]
        cert = certify_subspace_2d(basis, grid, approx=approx)
        rows.append(_row(
            "neural_pair_seed_deflated", cert,
            principal_angle(basis, exact_pair, grid.cell) if cert.valid else None,
            {"in_subspace": [
                _in_subspace(neural[first], exact_pair, grid),
                _in_subspace(partner, exact_pair, grid),
            ]},
        ))
        mixed = 0.6 * neural[first] + 0.8 * partner
        cert = certify_subspace_2d([neural[first], mixed], grid, approx=approx)
        rows.append(_row(
            "neural_mixed_seed_deflated", cert,
            principal_angle([neural[first], mixed], exact_pair, grid.cell)
            if cert.valid else None,
        ))
    else:
        print(f"[exp8] 没有找到 deflation 伙伴 {partner_path}，跳过强制第二方向的案例")

    # 独立参考值（计数二分，慢）
    refs = None
    if not args.skip_slow_refs:
        refs = {
            "lambda_first": discrete_eigenvalue_2d(grid, 1, approx=approx),
            "lambda_third": discrete_eigenvalue_2d(grid, 3, approx=approx),
        }

    summary = {
        "n_grid": n,
        "exact_spectrum": [float(v) for v in values[:4]],
        "degenerate_split": float(abs(values[1] - values[2])),
        "count_reference": refs,
        "neural_overlaps": overlaps,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "subspace.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 二维简并簇的子空间证书（方阱，{}×{}）".format(n, n),
        "",
        "精确离散谱：E₀={:.9f}，E₁={:.9f}，E₂={:.9f}，|E₁−E₂|={:.2e}，"
        "E₃={:.6f}".format(
            values[0], values[1], values[2], abs(values[1] - values[2]), values[3]
        ),
        "",
        "| case | m | 验证重数 | θ | ρ_i | 区间 | δ | sinΘ 界 | 真实 sinΘ | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "ritz_values" not in r:
            lines.append(
                f"| {r['case']} | 1 | — | — | — | — | — | {r.get('angle_bound')} | — | "
                f"{'接受' if r['valid'] else '拒绝'} |"
            )
            continue
        theta = ", ".join(f"{t:.9f}" for t in r["ritz_values"])
        rho = ", ".join(f"{v:.1e}" for v in r["residual_norms"])
        iv = (
            f"({r['interval'][0]:.6f}, {r['interval'][1]:.6f})"
            if r["interval"] else "—"
        )
        delta = f"{r['margin']:.6f}" if r["margin"] is not None else "—"
        bound = f"{r['angle_bound']:.3e}" if r["angle_bound"] is not None else "—"
        true = (
            f"{r['true_angle']:.3e}" if r.get("true_angle") is not None else "—"
        )
        lines.append(
            f"| {r['case']} | {r['n_states']} | {r['verified_multiplicity']} | {theta} | "
            f"{rho} | {iv} | {delta} | {bound} | {true} | "
            f"{'接受' if r['valid'] else '拒绝'} |"
        )
    (args.out_dir / "subspace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in rows:
        print(
            f"{r['case']:26s} valid={str(r['valid']):5s} "
            f"count={r.get('verified_multiplicity')} "
            f"angle={r.get('angle_bound')} true={r.get('true_angle')}"
            + ("" if r["valid"] else f"  reason={r['reason'][:60]}")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
