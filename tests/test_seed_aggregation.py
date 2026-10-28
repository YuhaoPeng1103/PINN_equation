"""多种子汇总脚本的回归测试（纯 numpy/标准库，不需要 torch）。

钉住两件事：
1. 均值/标准差的口径（ddof=1），避免以后改成总体标准差却不自知；
2. 目录发现逻辑（seed0/seed1/...）与 JSON 字段映射，避免汇总静默漏掉种子。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_seeds as agg  # noqa: E402


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _metrics(err: float, overlap: float = 0.999) -> dict:
    return {
        "energy_model": 0.5 + err,
        "energy_ref": 0.5,
        "energy_abs_err": err,
        "energy_rel_err": 2.0 * err,
        "overlap": overlap,
        "inf_norm": 0.01,
        "residual_rms": 0.3,
    }


def test_stats_uses_sample_std() -> None:
    s = agg.stats([1.0, 3.0])
    assert s["n"] == 2
    assert s["mean"] == 2.0
    # 样本标准差 = sqrt(((1-2)^2 + (3-2)^2)/1) = sqrt(2)
    assert abs(s["std"] - 2.0**0.5) < 1e-12
    assert agg.stats([5.0])["std"] == 0.0


def test_exp2_summary_discovers_each_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for seed, err in ((0, 1e-3), (1, 3e-3), (2, 5e-3)):
            _write_json(
                root / "exp2" / "rayleigh_grid" / f"seed{seed}" / "exp2_pinn.json",
                {"seed": seed, "state": 0, "metrics": _metrics(err), "history": {}},
            )
        data = agg.summarize_exp2(root, "exp2", "rayleigh_grid")
        assert data is not None
        assert data["n"] == 3
        assert data["seeds"] == [0, 1, 2]
        err = data["metrics"]["energy_abs_err"]
        assert abs(err["mean"] - 3e-3) < 1e-12
        assert abs(err["std"] - 2e-3) < 1e-12


def test_learned_energy_reports_per_seed_state() -> None:
    """可学习 E 的配置必须逐个种子报告，否则“不同种子停在不同本征态”会被均值掩盖。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # seed 1 收敛到第一激发态（E≈1.5），其余两个停在基态（E≈0.5）
        for seed, e_learned, e_rayleigh in (
            (0, 0.5001, 0.5000),
            (1, 1.4998, 1.5000),
            (2, 0.5003, 0.5002),
        ):
            metrics = _metrics(abs(e_rayleigh - 0.5))
            metrics["energy_model"] = e_rayleigh
            _write_json(
                root / "exp2" / "residual_learned" / f"seed{seed}" / "exp2_pinn.json",
                {
                    "seed": seed,
                    "state": 0,
                    "residual_energy": "learned",
                    "residual_energy_init": 1.0,
                    "learned_energy": e_learned,
                    "metrics": metrics,
                    "history": {},
                },
            )
        data = agg.summarize_exp2(root, "exp2", "residual_learned")
        assert data is not None
        # 原始值按种子排序保留，不做平均
        assert data["learned_energy"]["values"] == [0.5001, 1.4998, 0.5003]
        assert data["learned_state_index"] == [0, 1, 0]
        # |E_learned − E_Rayleigh| 分别为 1e-4, 2e-4, 1e-4 -> 均值 4/3e-4
        gap = data["learned_vs_rayleigh_gap"]
        assert abs(gap["mean"] - (1e-4 + 2e-4 + 1e-4) / 3) < 1e-12
        assert data["learned_near_ref_fraction"] == 2 / 3
        md = agg.markdown({"exp2": {"residual_learned": data}, "exp1": {}})
        assert "0.5001 / 1.4998 / 0.5003" in md
        assert "n=0,n=1,n=0" in md


def test_exp3_summary_filename_includes_k() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_json(
            root / "exp3" / "k64" / "seed0" / "exp3_data_assimilation_k64.json",
            {"seed": 0, "n_obs": 64, "metrics": _metrics(1e-3)},
        )
        data = agg.summarize_exp3(root, "k64", 64)
        assert data is not None and data["n_obs"] == 64
        # 用错 k 的文件名必须查不到，防止汇总把不同观测数的结果混在一起
        assert agg.summarize_exp3(root, "k64", 16) is None


def test_exp1_summary_separates_train_and_interpolation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for seed in (0, 1):
            _write_json(
                root / "exp1" / "baseline" / f"seed{seed}" / "exp1_regularized.json",
                {
                    "seed": seed,
                    "energies": {
                        "0.50": {"max_abs_err": 1e-2 + seed * 1e-3, "is_interpolated": False},
                        "1.00": {"max_abs_err": 4e-1 + seed * 1e-1, "is_interpolated": True},
                    },
                },
            )
        data = agg.summarize_exp1(root)
        assert data is not None and data["n"] == 2
        assert data["energies"]["0.50"]["is_interpolated"] is False
        assert data["energies"]["1.00"]["is_interpolated"] is True
        assert data["interp_max_abs_err"]["mean"] == 0.45
