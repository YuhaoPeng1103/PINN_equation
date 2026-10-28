"""势函数库。"""

from __future__ import annotations

import numpy as np
import pytest

from nnschrodinger.physics.potentials import POTENTIALS, get_potential


def test_harmonic_potential():
    spec = get_potential("harmonic")
    x = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(spec(x), [2.0, 0.0, 2.0])
    assert spec.domain == (-6.0, 6.0)
    assert spec.deep_well


def test_infinite_well_potential_jumps():
    spec = get_potential("infinite_well")
    assert spec(np.array([0.25, 0.75])).tolist() == pytest.approx([0.0, 0.0])
    assert spec(np.array([-1.0, 2.0])).min() > 1e5
    assert spec.discontinuities == (0.0, 1.0)


def test_finite_well_potential():
    spec = get_potential("finite_well")
    assert spec(np.array([0.0]))[0] == pytest.approx(-50.0)
    assert spec(np.array([3.0]))[0] == pytest.approx(0.0)
    assert spec.domain == (-8.0, 8.0)


def test_poschl_teller_symmetric():
    spec = get_potential("poschl_teller")
    x = np.linspace(-5, 5, 101)
    vals = spec(x)
    assert np.allclose(vals, vals[::-1])
    assert vals.min() < 0.0


def test_unknown_potential():
    with pytest.raises(KeyError):
        get_potential("not_a_potential")
    assert "harmonic" in POTENTIALS
