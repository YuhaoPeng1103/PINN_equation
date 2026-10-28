"""数据层：目标函数、网格与相似度度量。"""

from __future__ import annotations

import numpy as np
import pytest

from nnschrodinger.data import HHL_FUNCTIONS, HHL_GRID_POINTS, HHL_H, HHL_RANGE, make_grid, make_target, similarity


def test_grid_defaults():
    grid = make_grid()
    assert grid.size == HHL_GRID_POINTS == 2500
    assert grid[0] == HHL_RANGE[0] and grid[-1] == HHL_RANGE[1]


def test_target_values_within_h():
    grid = make_grid()
    for name in HHL_FUNCTIONS:
        vals = make_target(name, grid)
        assert np.all(np.isfinite(vals))
        assert np.max(np.abs(vals)) <= HHL_H


def test_unknown_target():
    with pytest.raises(KeyError):
        make_target("nope", np.linspace(-1, 1, 5))


def test_similarity_identity_and_orthogonality():
    x = np.linspace(-3, 3, 2001)
    a = np.exp(-(x**2))
    b = x * np.exp(-(x**2))
    assert similarity(a, a, x) == pytest.approx(1.0)
    assert similarity(-2.0 * a, a, x) == pytest.approx(1.0)
    assert similarity(a, b, x) < 1e-8
    assert similarity(np.zeros_like(x), a, x) == 0.0
