"""生成论文插图与绘图数据（不依赖 matplotlib）。

本机 matplotlib 与 numpy 2.x ABI 不兼容（3.7.2 编译于 numpy 1.x），
所以：

* 二维波函数热图用 Pillow 直接从 numpy 数组写 PNG；
* 曲线图导出 CSV，由 pgfplots（LaTeX 内）绘制——矢量、可复现。

输出：
    paper/figures/fig_<potential>_net.png    网络输出 |ψ|（每幅归一到自身最大值）
    paper/figures/fig_<potential>_ref.png    同一网格上的参考 |ψ_ref|
    paper/figures/data/width_vs_cutoff.csv   Temple 宽度 vs 截断阶数 K（一维）
    paper/figures/data/conv2d.csv            二维离散误差 vs h
    paper/figures/data/finite_well_alignment.csv  有限深阱：λ_h − λ* vs N（§6.5）

用法：
    python paper/make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify2d import Grid2D, _sparse_operator  # noqa: E402

OUT = ROOT / "paper" / "figures"
DATA = OUT / "data"


# 近似 viridis 的锚点颜色（0 → 1），足够印刷；不追求与 matplotlib 完全一致。
_ANCHORS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
], dtype=float)


def _cmap(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    pos = t * (len(_ANCHORS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_ANCHORS) - 1)
    frac = (pos - lo)[..., None]
    rgb = _ANCHORS[lo] * (1 - frac) + _ANCHORS[hi] * frac
    return rgb.astype(np.uint8)


def _render(values: np.ndarray, path: Path, scale: int = 6, border: int = 3) -> None:
    """values[i, j]（i↔x, j↔y）→ 上-北朝上的 PNG。"""
    img = np.abs(values).T          # 行= y，列= x
    img = np.flipud(img)            # y 轴向上
    peak = float(np.max(img)) or 1.0
    rgb = _cmap(img / peak)
    rgb = np.pad(rgb, ((border, border), (border, border), (0, 0)),
                 constant_values=255)
    pil = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.BILINEAR
    )
    pil.save(path)
    print(f"wrote {path.relative_to(ROOT)}  {values.shape}")


def _load(name: str):
    npz = np.load(ROOT / "runs" / "seeds2d" / name / "seed0" / "wavefunction2d.npz")
    return npz["x"], npz["y"], npz["psi"], npz["potential"]


def _reference(x: np.ndarray, y: np.ndarray, potential: np.ndarray):
    """同一网格上 H_h 的基态（ARPACK；仅作显示参考，不进入任何误差估计）。

    旧版对 harmonic/well 直接写解析式、对 anharmonic 用零势解——后者的
    "参考"其实是方阱基态，与网络输出比的是别的势。
    """
    from scipy.sparse.linalg import eigsh

    grid = Grid2D(x, y, np.asarray(potential, dtype=float))
    values, vectors = eigsh(
        _sparse_operator(grid, 0.0), k=1,
        sigma=float(np.min(grid.potential)) - 1.0, which="LM", tol=1e-12,
    )
    ref = np.zeros((x.size, y.size))
    ref[1:-1, 1:-1] = vectors[:, 0].reshape(x.size - 2, y.size - 2)
    ref = ref / np.max(np.abs(ref))
    return ref


def fig_heatmaps() -> None:
    for name in ("harmonic2d", "well2d", "anharmonic2d"):
        x, y, psi, potential = _load(name)
        psi = psi / np.max(np.abs(psi))
        _render(psi, OUT / f"fig_{name}_net.png")
        _render(_reference(x, y, potential), OUT / f"fig_{name}_ref.png")


def fig_degenerate() -> None:
    """两幅网络输出 vs 精确简并对的两位成员（|ψ|，各自归一）。"""
    base = ROOT / "runs" / "degeneracy2d" / "neural"
    a = np.load(base / "seed0" / "wavefunction2d.npz")
    # seed3 是四个独立 run 里唯一落到正交伙伴上的那个；没有它就退回去用
    # 强制 deflation 的伙伴，保证两幅图展示的是两个不同方向。
    partner = (ROOT / "runs" / "degeneracy2d" / "neural_partner" / "seed0"
               / "wavefunction2d.npz")
    seed3 = base / "seed3" / "wavefunction2d.npz"
    b_path = seed3 if seed3.exists() else partner
    b = np.load(b_path if b_path.exists() else base / "seed1" / "wavefunction2d.npz")
    _render(a["psi"] / np.max(np.abs(a["psi"])), OUT / "fig_deg_net_a.png")
    _render(b["psi"] / np.max(np.abs(b["psi"])), OUT / "fig_deg_net_b.png")

    from scipy.sparse.linalg import eigsh

    grid = Grid2D(a["x"], a["y"], a["potential"])
    values, vectors = eigsh(
        _sparse_operator(grid, 0.0), k=3,
        sigma=float(np.min(grid.potential)) - 1.0, which="LM", tol=1e-12,
    )
    order = np.argsort(values)
    vectors = vectors[:, order]
    for i, tag in ((1, "a"), (2, "b")):
        psi = np.zeros(grid.shape)
        psi[1:-1, 1:-1] = vectors[:, i].reshape(
            grid.shape[0] - 2, grid.shape[1] - 2
        )
        _render(psi / np.max(np.abs(psi)), OUT / f"fig_deg_exact_{tag}.png")


def fig_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    # (a) 一维：Temple 宽度 vs 截断阶数（certs5_rayleigh 基态）
    src = ROOT / "runs" / "certificates_filtered" / "certs5_rayleigh" / "summary.json"
    runs = json.loads(src.read_text(encoding="utf-8"))
    picked = {}
    for run in runs:
        key = run["path"].replace("\\", "/").split("/")[-3]  # e.g. harmonic_0
        if key.endswith("_0") and key not in picked:
            picked[key] = run["scan"]
    for key, scan in picked.items():
        name = key.rsplit("_", 1)[0]
        with (DATA / f"width_vs_cutoff_{name}.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["cutoff", "width", "residual"])
            for row in scan:
                if row.get("width") is None:
                    continue
                w.writerow([row["cutoff"], f"{row['width']:.6e}",
                            f"{row['residual_norm']:.6e}"])
    print("wrote paper/figures/data/width_vs_cutoff_{harmonic,infinite_well,finite_well}.csv")

    # (b) 二维：|λ_h − λ*| vs h
    src = ROOT / "runs" / "discretization2d" / "summary.json"
    disc = json.loads(src.read_text(encoding="utf-8"))
    for name, rows in disc["problems"].items():
        with (DATA / f"conv2d_{name}.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["n", "h", "error"])
            for row in sorted(rows, key=lambda r: r["n"]):
                w.writerow([row["n"], f"{row['h']:.6f}",
                            f"{row['error_vs_lambda_star']:.6e}"])
    print("wrote paper/figures/data/conv2d_{harmonic2d,well2d}.csv")

    # (c) 证书宽度 vs 真实离散误差（一维 + 二维）
    pts = []
    one_d = ROOT / "runs" / "certificates_filtered" / "certs5_rayleigh" / "summary.json"
    for run in json.loads(one_d.read_text(encoding="utf-8")):
        pts.append(("1D " + run["path"].replace("\\", "/").split("/")[-3].rsplit("_", 1)[0],
                    abs(run["filtered_certificate"]["energy"] - run["discrete_ref"]),
                    run["filtered_width"]))
    # summary.json 的 discrete_ref 在二维下来自 ARPACK（提议区间用的同一个
    # 求解器），故改用 runs/reference2d/dense_ref.json 的稠密 LAPACK 值。
    two_d = ROOT / "runs" / "certificates_filtered2d" / "summary.json"
    dense = {}
    dense_path = ROOT / "runs" / "reference2d" / "dense_ref.json"
    for _, block in json.loads(dense_path.read_text(encoding="utf-8"))["problems"].items():
        for row in block["runs"]:
            dense[row["path"]] = row["dense"]
    for run in json.loads(two_d.read_text(encoding="utf-8")):
        lam = float(dense.get(run["path"], run["discrete_ref"]))
        pts.append(("2D " + run["potential"],
                    abs(run["filtered_certificate"]["energy"] - lam),
                    run["filtered_width"]))
    series_keys = sorted({p[0] for p in pts})
    for key in series_keys:
        fname = "scatter_" + key.replace(" ", "_").replace("(", "").replace(")", "") + ".csv"
        with (DATA / fname).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["dE", "width"])
            for series, de, width in pts:
                if series == key:
                    w.writerow([f"{de:.6e}", f"{width:.6e}"])
    print("wrote paper/figures/data/scatter_*.csv:", series_keys)


def fig_alignment() -> None:
    """有限深阱：固定势、只挪网格，离散基态误差随 N 变号（§6.5）。

    只取 N=1000..1011 这段窗口；N=4001 的对照点离得太远，会压扁横轴，
    写在正文与图注里。
    """
    DATA.mkdir(parents=True, exist_ok=True)
    src = ROOT / "runs" / "continuous_bound" / "alignment.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    with (DATA / "finite_well_alignment.csv").open("w", newline="",
                                                   encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["N", "error", "offset"])
        for row in payload["rows"]:
            if row["N"] > 2000:          # 4001 只作正文中的对照
                continue
            offset = abs(row["jumps"][0]["offset_h"])
            w.writerow([row["N"], f"{row['error']:.6e}", f"{offset:.4f}"])
    print("wrote paper/figures/data/finite_well_alignment.csv")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_heatmaps()
    fig_data()
    fig_alignment()
    if (ROOT / "runs" / "degeneracy2d" / "neural" / "seed1").exists():
        fig_degenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
