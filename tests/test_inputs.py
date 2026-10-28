"""采样器：形状、覆盖范围与可复现性。"""

from __future__ import annotations

import numpy as np

from nnschrodinger.solver.inputs import (
    dirichlet_points,
    energy_blocks,
    hhl_input_map,
    hhl_pairs,
    uniform_collocation,
    well_aware_collocation,
)


def test_uniform_collocation():
    rng = np.random.default_rng(0)
    s = uniform_collocation(-1.0, 1.0, 500, rng, energy=2.0)
    assert s.x.shape == (500,)
    assert s.x.min() >= -1.0 and s.x.max() <= 1.0
    assert np.all(np.diff(s.x) >= 0)
    assert s.as_inputs().shape == (500, 1)
    assert s.as_inputs(energy_augmented=True).shape == (500, 2)
    assert np.allclose(s.as_inputs(energy_augmented=True)[:, 1], 2.0)


def test_energy_blocks_one_per_energy():
    blocks = energy_blocks(0.0, 1.0, [1.0, 2.0, 3.0], 100, np.random.default_rng(1))
    assert [b.E for b in blocks] == [1.0, 2.0, 3.0]
    assert all(b.x.size == 100 for b in blocks)


def test_well_aware_collocation_covers_edges():
    rng = np.random.default_rng(2)
    s = well_aware_collocation(0.0, 1.0, (0.5,), 400, rng)
    assert s.x.min() >= 0.0 and s.x.max() <= 1.0
    # 跳变点附近应有加密样本
    assert np.sum(np.abs(s.x - 0.5) < 1e-3) >= 8


def test_hhl_input_map_is_affine():
    x = np.array([-1.0, 0.0, 1.0])
    out = hhl_input_map(x, c=0.5, M=2.0)
    assert np.allclose(out, [-3.0, -1.0, 1.0])


def test_hhl_pairs_shape():
    pairs, es = hhl_pairs(-5.0, 5.0, [0.5, 1.5], 1000, np.random.default_rng(0))
    assert pairs.shape == (1000, 2)
    assert set(np.unique(es)) <= {0.5, 1.5}
    assert np.all(np.abs(pairs[:, 0]) <= 5.0)


def test_dirichlet_points():
    pts = dirichlet_points(-2.0, 3.0)
    assert pts.shape == (2, 1)
    assert pts.tolist() == [[-2.0], [3.0]]
