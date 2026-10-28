"""评测指标：用解析解自检“完美解应得满分、杂波解必须被发现”。"""

from __future__ import annotations

import numpy as np

from nnschrodinger.physics.spectrum import get_reference
from nnschrodinger.solver.evaluate import evaluate_model, format_table, residual_rms


def test_perfect_solution_scores_perfectly():
    ref = get_reference("harmonic", 0)
    x = np.linspace(-6, 6, 2001)
    potential = lambda xs: 0.5 * xs**2  # noqa: E731
    m = evaluate_model(ref.psi(x), x, ref.psi, ref.energy, potential)
    assert abs(m.overlap - 1.0) < 1e-6
    assert m.inf_norm < 1e-6
    assert m.energy_abs_err < 1e-4
    assert m.residual_rms < 1e-2


def test_orthogonal_noise_is_caught():
    ref = get_reference("harmonic", 0)
    x = np.linspace(-6, 6, 2001)
    potential = lambda xs: 0.5 * xs**2  # noqa: E731
    # “能量准但波形是杂波”的典型失败：向真解叠加一个正交分量
    noise = np.sin(7.3 * x) * np.exp(-0.1 * x**2)
    noisy = ref.psi(x) + 0.8 * noise
    m = evaluate_model(noisy, x, ref.psi, ref.energy, potential)
    assert m.overlap < 0.99
    assert m.inf_norm > 0.1


def test_rayleigh_energy_matches_for_exact_state():
    ref = get_reference("harmonic", 2)
    x = np.linspace(-10, 10, 4001)
    m = evaluate_model(ref.psi(x), x, ref.psi, ref.energy, lambda xs: 0.5 * xs**2)
    assert abs(m.energy_model - ref.energy) < 1e-3


def test_residual_rms_of_exact_state_small():
    ref = get_reference("infinite_well", 1)
    x = np.linspace(0.0, 1.0, 4001)
    v = np.zeros_like(x)
    assert residual_rms(ref.psi(x), x, ref.energy, v) < 1e-1


def test_format_table():
    table = format_table([{"a": 1.0, "b": 2.5}])
    assert table.splitlines()[0] == "| a | b |"
    assert len(table.splitlines()) == 3
