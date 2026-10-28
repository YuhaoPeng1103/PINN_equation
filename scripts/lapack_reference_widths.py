"""Certificate widths for classical LAPACK eigenvectors of the 1D H_h.

§6.9 of the paper compares the certified width of a trained network with the
widest certificate the same machinery can produce, applied to a classical
eigenvector.  Until now those three widths lived only in the prose; this
script recomputes them from the saved grids and writes
``runs/reference1d/lapack_certificates.json`` so the comparison is auditable.

The eigenvector comes from SciPy's ``eigh_tridiagonal`` (LAPACK), applied to
the interior tridiagonal block; the value returned by the solver is *not* used
as a reference — it carries an absolute error of order ``eps·||H_h||`` — the
Rayleigh quotient of the returned vector is used instead, exactly as in
``scripts/certify_saved.py``.

Usage:
    python scripts/lapack_reference_widths.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nnschrodinger.solver.certify import certify_eigenpair  # noqa: E402

POTENTIALS = ("harmonic", "infinite_well", "finite_well")
OUT = ROOT / "runs" / "reference1d" / "lapack_certificates.json"


def _grid(potential: str):
    npz = np.load(
        ROOT / "runs" / "seeds" / "certs5_rayleigh" / f"{potential}_0" / "seed0"
        / "wavefunction.npz"
    )
    return npz["x"], npz["potential"]


def _lapack_eigenvector(x: np.ndarray, potential: np.ndarray, index: int = 0):
    h = float(x[1] - x[0])
    diag = 1.0 / h**2 + potential[1:-1]
    off = np.full(diag.size - 1, -0.5 / h**2)
    values, vectors = eigh_tridiagonal(
        diag, off, select="i", select_range=(index, index)
    )
    v = np.asarray(vectors[:, 0], dtype=float)
    psi = np.zeros_like(x)
    psi[1:-1] = v / np.linalg.norm(v)
    eta = float(np.dot(v, diag * v) + 2.0 * np.dot(v[1:] * off, v[:-1]))
    return psi, eta, float(values[0])


def main() -> int:
    rows = []
    for name in POTENTIALS:
        x, potential = _grid(name)
        psi, eta, value = _lapack_eigenvector(x, potential)
        cert = certify_eigenpair(psi, x, potential, isolate=True)
        rows.append(
            {
                "potential": name,
                "grid_points": int(x.size),
                "eigenvector_solver": "scipy.linalg.eigh_tridiagonal (LAPACK)",
                "rayleigh_quotient": eta,
                "lapack_value_used": False,
                "lapack_eigenvalue_abs_error_order": (
                    "eps*||H_h||; not used as a reference"
                ),
                "valid": bool(cert.valid),
                "width": float(cert.width) if cert.width is not None else None,
                "angle_bound": (
                    float(cert.eigenfunction_angle_bound)
                    if cert.eigenfunction_angle_bound is not None
                    else None
                ),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['potential']:14s} valid={row['valid']} "
            f"width={row['width']:.4e}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
