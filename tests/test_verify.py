"""交叉校验器本身的行为（它可以慢，但必须诚实）。"""

from __future__ import annotations

from nnschrodinger.physics import verify


def test_harmonic_ground_state_cross_check():
    row = verify.check("harmonic", 0)
    assert row["passed"]
    assert row["energy_abs_err"] < 5e-3
    assert row["psi_max_abs_err"] < 2e-3


def test_finite_well_cross_check():
    row = verify.check("finite_well", 0)
    assert row["passed"]


def test_convergence_reduces_error():
    rows = verify.convergence("harmonic", 0, resolutions=(2001, 8001))
    e_ref = verify.get_reference("harmonic", 0).energy
    coarse = abs(rows[0]["energy"] - e_ref)
    fine = abs(rows[1]["energy"] - e_ref)
    assert fine < coarse
