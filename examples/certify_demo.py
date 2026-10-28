"""Minimal, self-contained demonstration of the certificate API.

Run from the repository root:

    python examples/certify_demo.py

It needs only NumPy and SciPy (no PyTorch, no trained network): the
"approximations" below are analytic functions sampled on the grid, which is
exactly the input the certificate expects from a neural solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnschrodinger.solver.certify import certify_eigenpair  # noqa: E402
from nnschrodinger.solver.certify2d import (  # noqa: E402
    Grid2D,
    _sparse_operator,
    certify_eigenpair_2d,
)
from nnschrodinger.solver.certify_subspace import certify_subspace_2d  # noqa: E402


def demo_1d() -> None:
    """Harmonic oscillator ground state: is exp(-x^2/2) good enough to certify?"""
    x = np.linspace(-6.0, 6.0, 2001)
    V = 0.5 * x**2
    psi = np.exp(-0.5 * x**2)

    cert = certify_eigenpair(psi, x, V, isolate=True)
    print("1D harmonic oscillator, analytic trial function")
    print(f"  energy            {cert.energy:.12f}")
    print(f"  isolating interval "
          f"({cert.interval_lower:.9f}, {cert.interval_upper:.9f})")
    print(f"  certified width   {cert.width:.3e}")
    print(f"  state index       {cert.state_index}")
    print(f"  angle bound       {cert.eigenfunction_angle_bound:.3e}")
    print(f"  valid             {cert.valid}  ({cert.reason})")
    print()


def demo_2d_cluster() -> None:
    """A degenerate level: refused as a state, certified as a subspace."""
    n = 61
    x = np.linspace(0.0, 1.0, n)
    grid = Grid2D(x, x.copy(), np.zeros((n, n)))

    from scipy.sparse.linalg import eigsh

    values, vectors = eigsh(
        _sparse_operator(grid, 0.0), k=3,
        sigma=-1.0, which="LM", tol=1e-12,
    )
    order = np.argsort(values)
    vectors = vectors[:, order]

    def embed(vec: np.ndarray) -> np.ndarray:
        psi = np.zeros(grid.shape)
        psi[1:-1, 1:-1] = vec.reshape(n - 2, n - 2)
        return psi

    psi = embed(vectors[:, 1])

    single = certify_eigenpair_2d(psi, grid, isolate=True)
    print(f"2D box, one member of the degenerate pair (E = {single.energy:.9f})")
    print(f"  single-state certificate: valid={single.valid}")
    print(f"  reason: {single.reason}")
    print()

    basis = [embed(vectors[:, 1]), embed(vectors[:, 2])]
    cluster = certify_subspace_2d(basis, grid)
    print("2D box, both members as a trial subspace")
    print(f"  verified multiplicity {cluster.verified_multiplicity}")
    print(f"  interval              "
          f"({cluster.interval_lower:.6f}, {cluster.interval_upper:.6f})")
    print(f"  rho_F                 {cluster.frobenius_residual:.3e}")
    print(f"  margin delta          {cluster.margin:.4f}")
    print(f"  angle bound           {cluster.angle_bound:.3e}")
    print(f"  valid                 {cluster.valid}")


if __name__ == "__main__":
    demo_1d()
    demo_2d_cluster()
