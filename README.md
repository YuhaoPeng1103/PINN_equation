# nnschrodinger — neural eigenpairs of the Schrödinger equation, certified

**Paper.** *Certifying neural eigenpairs and degenerate eigenspaces of the
Schrödinger equation: a self-verifying a posteriori protocol and its
implementation* — Chunyang Chen and Yuhao Peng (equal contribution), School
of Mathematics and Physics, China University of Geosciences (Wuhan).
Compiled manuscript: [`paper/main.pdf`](paper/main.pdf) (55 pages).
Chinese version of this file: [`README.zh.md`](README.zh.md).

Given an approximate eigenfunction of the finite-difference Schrödinger
operator, the package returns a two-sided enclosure of the discrete
eigenvalue, a verified state index, and a bound on the angle to the true
eigenfunction (Kato–Temple and Davis–Kahan). The hypothesis these classical
bounds require — that the interval contains exactly one eigenvalue — is
*checked* by counting the spectrum below a shift: Sturm sequences in one
dimension, the Sylvester inertia of a no-pivot sparse LU in two. The count is
re-derived in directed-rounding interval arithmetic in one dimension and for
separable two-dimensional potentials, so that there it is a theorem about the
stored matrix rather than a guarded estimate. Degenerate levels, where no
single eigenfunction exists, are certified as invariant subspaces with the
multiplicity measured by counting; and a variational bound on the P1/Q1
interpolant carries the upper side of the guarantee to the continuous
operator.

```bash
pip install -r requirements.txt
python tests/run_tests.py          # 143 tests
python examples/certify_demo.py    # one certificate in ~4 s (NumPy/SciPy only)
```

To regenerate every table and figure of the paper, see
[`paper/README.md`](paper/README.md); the measured artefacts behind each
number are under [`runs/`](runs/). Licensed under MIT
([`LICENSE`](LICENSE)).

---

## What this project is about

A complete pipeline built around **solving the stationary Schrödinger
equation with PINNs (physics-informed neural networks)**: reference solution →
network / loss / training → evaluation → reproducible experiment, with tests
and recorded evidence at every layer.

```
  physics layer          solver layer             experiment layer
  ─────────────          ────────────             ────────────────
  analytic + Numerov  →  PINN / (x,E) network  →  Exp. 1 supervised regression
  cross-checked 10/10    Rayleigh / residual loss  Exp. 2 unsupervised PINN
  pure NumPy, no deps    classical baseline        Exp. 3 data assimilation
```

> The full narrative is in **[docs/project-overview.md](docs/project-overview.md)**.

---

## The problem

The one-dimensional stationary Schrödinger equation

```
    Ĥψ = Eψ,      Ĥ = −½ d²/dx² + V(x)     (ħ = m = 1)
```

is addressed at three levels:

1. **What is the truth?** — provide trustworthy reference eigenpairs and
   cross-check them with a second method;
2. **How does a network solve it?** — three training paradigms: unsupervised
   PINN, regularized `(x, E)` regression, and data assimilation;
3. **How good is the solution?** — report energy, waveform similarity,
   worst-point deviation and equation residual together.

## Quick start

```bash
pip install -e .          # NumPy core only (torch is not needed)

# (1) reference-solution cross-check (pure NumPy)
python -m nnschrodinger.physics.verify
# prints: 10/10 passed

# (2) unit tests (pytest if available, otherwise the bundled runner)
python tests/run_tests.py
# prints: 143 passed, 0 failed, 0 skipped

# (3) smallest example of the certificate API (no torch, no training)
python examples/certify_demo.py
# 1D harmonic oscillator: certified interval and angle bound for an analytic
# trial function; 2D square box, degenerate pair: the state certificate
# refuses (level gap below the resolution) while the cluster certificate
# accepts
```

To train networks:

```bash
pip install -e .[torch]   # or pip install torch
python scripts/exp1_regularized_network.py   # experiment 1
python scripts/exp2_unregularized_pinn.py    # experiment 2
python scripts/exp3_data_assimilation.py     # experiment 3

# multi-seed reproduction (seeds 0/1/2, parallel; 4 threads per process)
python scripts/run_seed_sweep.py --jobs 6 --threads 4
python scripts/aggregate_seeds.py            # -> runs/seeds/summary.{json,md}

# classical numerical baseline (Numerov, same metrics as the network)
python scripts/classical_baseline.py

# two dimensions (experiment 6): seed sweep -> filtering + certificate summary
python scripts/run_seed_sweep_2d.py --jobs 2
python scripts/filter_certificates2d.py
# 2D degeneracy: state-level refusal / subspace certificate / discretization gap
python scripts/exp7_degeneracy_2d.py
python scripts/exp8_subspace_certificate.py
python scripts/exp9_resolution_boundary.py   # resolution boundary between the
                                             # state and the cluster certificate
python scripts/discretization_study_2d.py --sizes 41 61 81 101 121
python scripts/bench_count.py                # counting/certificate timings
python scripts/dense_reference_2d.py         # independent 2D reference (dense
                                             # LAPACK, not ARPACK)
python scripts/appendix_failures.py --variant soft_boundary   # appendix A
python scripts/appendix_failures.py --variant unnormalized
python scripts/residual_vs_error.py runs/appendix_failures/unnormalized
python scripts/lapack_reference_widths.py    # certificate widths for classical
                                             # LAPACK eigenvectors (Sec. 6.9)
```

## Current verification status

| Check | Result | Evidence |
|---|---|---|
| Reference cross-check (analytic ⟷ Numerov) | **10 / 10 passed** | [runs/physics_verification.txt](runs/physics_verification.txt) |
| Unit tests (including torch) | **143 / 143 passed** | [runs/tests_output_torch.txt](runs/tests_output_torch.txt) |
| Experiment 1, supervised regression | run | [runs/exp1_regularized.json](runs/exp1_regularized.json) |
| Experiment 2, unsupervised PINN | run | [runs/exp2_pinn.json](runs/exp2_pinn.json) |
| Experiment 3, data assimilation | run | [runs/exp3_data_assimilation_k*.json](runs/) |

### Key measured results

Multi-seed numbers below are means over seeds 0/1/2 with the sample standard
deviation ([runs/seeds/summary.md](runs/seeds/summary.md)); single-run
numbers are the recorded values in the linked JSON.

| Experiment | Metric | Value |
|---|---|---|
| Exp. 2 PINN (residual with Rayleigh quotient, grid sampling) | energy error / overlap | **1.63e-3 ± 2.70e-3 / 0.999985** |
| Exp. 2 PINN (learned E, weight 1.5) | energy error / overlap | 3.81e+0 ± 0.97 / 0.123 |
| Exp. 2 PINN (learned E, no Rayleigh term) | energy error / overlap | 3.48e+0 ± 1.26 / 0.110 |
| Exp. 3 data assimilation k=0 → 64 | energy error | 2.78e-2 → **5.58e-3** |
| Exp. 1 trained energy → interpolated energy | waveform error | 1.8e-2 → 0.25–1.78 |
| Exp. 5 certificate at matched training budget (prototype) | discrete residual / Temple width | 0.361 / 0.260 (valid) |
| Exp. 5 + certificate-guided spectral filtering | discrete residual / Temple width | **0.0051 / 5.2e-5** |
| Exp. 5 multi-potential ground states (3 potentials × 3 seeds, Rayleigh-quotient residual) | interval contains reference | **12 / 12** (median width 5.0e-4) |
| Exp. 5 excited states n=1 / n=2 (learnable deflation) | target state reached and certified | **8 / 9 · 8 / 9** (previously 0/6 for n=1) |
| Exp. 6 two dimensions (3 potentials × 3 seeds, 81×81) | certificate valid after filtering / contains λ_h | **9 / 9 · 9 / 9** (widths 2.0e-11 – 1.4e-3) |
| Exp. 6 2D degenerate refusal (square-well excited state) | exact eigenvector (r≈4e-12) | **refused correctly** (gap 7e-9 < resolution 1.3e-5) |
| Exp. 8 2D degenerate **subspace** certificate (81×81) | exact pair: multiplicity / principal-angle bound | **2 / 8.2e-13** (true angle 2.3e-16); single members still refused |
| Exp. 8 network pairs (4 seeds, deflation) | accepted spanning pairs / collapsed pairs | **3 pairs accepted at 5.6–6.3e-3** (true 2.4–3.0e-4); a fourth accepted but priced at 0.68, the ambiguous case; collapsed pair refused (ρ_F=16.3 > δ=6.1) |
| Exp. 9 resolution boundary of the state certificate (λ(x²−y²) splitting) | state-level switch / cluster certificate | gap/floor from 0.95 → 1.01 turns **refusal** into **acceptance** (the switch sits at the counting resolution); the cluster certificate holds throughout with δ=7.397, bound ~8e-13 |
| Directed-rounding counting (3 potentials × 121 shifts, N=2001) | certified / agreement with float and dense spectra | **361 / 363** (the two refusals occur exactly where the shift equals an eigenvalue) |
| Directed-rounding counting (2D, 81×81, separable potentials) | certified shifts / disagreements | **25 / 25 · 0** (non-separable potentials explicitly refused) |
| Continuous-operator one-sided bound (six problems × 2 grids; five have a reference) | reference lies in [λ_h, R(ψ_h)] | coarse grid **5/5**; fine grid **4/5** (the finite-well counterexample below); for the 2D anharmonic potential no reference exists, so the bound is one-sided |
| Independent 2D reference (dense LAPACK `eigh` + Rayleigh quotient) | difference from ARPACK / containment | **≤3.1e-13 / 9 / 9** (ARPACK only proposes intervals) |
| Appendix A failure modes (square well, N=41, 3000 epochs) | soft boundary / unnormalized / hard-mask control | 5.0e-2, 5.72 / ‖ψ‖²=6.0e-5 / 1.2e-6, 1.2e-2 (`runs/appendix_failures/`) |

Experiment 2's single-run orthogonality-constraint sweep of excited states
predates the loss-weight correction described under finding 1 below and is
therefore not quoted; excited states are reported instead through the
deflated, certified runs of experiment 5 ([runs/multistate/summary.md](runs/multistate/summary.md)).

The most valuable findings:

1. **Every term of a composite loss must actually enter the objective.** The
   `normalization=100` term of the original experiment-2 code was applied as
   1, so the residual-only configuration converged to ψ→0 (E=8.6 at
   evaluation). With the weights fixed the same configurations recover the
   ground state to 1.6e-3; the sweep in
   [runs/seeds/summary.md](runs/seeds/summary.md) is the evidence, and the
   earlier "21% / 25×" claim is withdrawn. See [docs/results.md](docs/results.md).
2. **Physical constraints and a few observations are complementary.** 64
   observations with 1% noise (against 2000 collocation points) drive the
   relative energy error down to 0.11%.
3. **An `(x, E)` two-input network does not "learn" the energy→wavefunction
   map.** The error on trained energies is 1.8e-2; on interpolated energies it
   is 0.25–1.78, one to two orders of magnitude larger. A capacity ablation
   (width 64→256) does **not** improve it, which rules out "the network is too
   small". This is inconsistent with the interpolability claimed in the HHL
   paper.
4. **Rayleigh's quotient on a random collocation set adds 1.9% noise** (2500
   points); switching to an equidistant grid brings it down to 1e-8. The
   energy term needs a deterministic grid — otherwise the energy curve keeps
   jittering and is easily misread as training instability.
5. **A posteriori certificates give error bars that do not use the reference
   solution, but "tight" holds only for smooth potentials.** The filtered
   intervals of 12 leak-free ground-state runs contain the discrete reference
   eigenvalue 12/12 with median width 5.0e-4; for the harmonic oscillator the
   width is 5.5–19× the true error (usable), for the finite well 946–2920×
   (not usable). Using the **Rayleigh quotient** instead of a learnable scalar
   for E in the residual reduces the finite-well width from 1.87–2.72 to
   0.37–0.73 without leakage. See [docs/certification.md](docs/certification.md).
6. **A discrete certificate is not a continuous one.** By min–max, the P1/Q1
   interpolant of a sampled function gives an upper bound on the continuous
   ground-state energy that needs no interpolation constant; the other
   direction is not free. For the finite well at N=4001 a jump lands on a node
   and pushes the discrete energy 3.67e-3 *above* the continuous value, while
   moving that single node from outside the well to inside it (0 → −50) on the
   same grid flips it 3.73e-3 *below* the true value — the whole error is that
   one grid point, and a finer grid does not remove it. For the box,
   `sinθ ≤ θ` gives a strict two-sided bound. Evidence:
   [runs/continuous_bound/alignment.json](runs/continuous_bound/alignment.json).

The full report is in **[docs/results.md](docs/results.md)**.

## Problems implemented

| Problem | Potential | Energy | Reference solution |
|---|---|---|---|
| `harmonic` | ½x² | n + ½ | analytic (Hermite recursion) |
| `infinite_well` | 0 (0<x<L) | n²π²/(2L²) | analytic (sine) |
| `finite_well` | −V₀ (\|x\|<a) | no closed form | Numerov + shooting |
| `hydrogen` | −1/\|x\| | −1/(2n²) | analytic (Laguerre) |

```python
from nnschrodinger.physics import get_reference

ref = get_reference("harmonic", 2)
print(ref.energy)        # 2.5
print(ref.psi([0.0, 1.0]))  # normalized wavefunction (∫|ψ|²dx = 1)
```

## Repository layout

```
src/nnschrodinger/
├── physics/    reference solutions, Numerov, Jacobi, potentials, cross-check
│               (pure NumPy)
├── data/       the ten target functions and similarity measures of the HHL
│               paper
└── solver/     sampling / networks / losses / training / evaluation
scripts/        experiment entry points + certificates / filtering / seed
                sweeps / aggregation / classical baseline
tests/          143 unit tests (including the subspace certificate)
docs/           project narrative, structure, literature, experiments,
                engineering log
runs/           run evidence
paper/          manuscript sources, generated tables and figures
```

The complete map is in **[docs/structure.md](docs/structure.md)**.

## Documentation

| Document | Content |
|---|---|
| [docs/project-overview.md](docs/project-overview.md) | ★ research question → code → evidence |
| [docs/structure.md](docs/structure.md) | directory responsibilities and data flow |
| [docs/literature.md](docs/literature.md) | literature skeleton of the three technical threads |
| [docs/experiments.md](docs/experiments.md) | setup, metrics and expectations of the three experiments |
| [docs/engineering-log.md](docs/engineering-log.md) | mistakes made and how they were fixed (Chinese) |
| [docs/results.md](docs/results.md) | ★ measured results and their reading (Chinese) |
| [HANDOFF.md](HANDOFF.md) | ★★ handover notes, sufficient to continue on another machine (Chinese) |
| [paper/README.md](paper/README.md) | how the manuscript is built and regenerated |
| [paper/main.pdf](paper/main.pdf) | ★★ the manuscript (CPC regular article, 55 pages, 24 references, 143 tests) |

## The manuscript

Every table and figure under `paper/` is **regenerated from `runs/*.json` by
script**, never transcribed by hand:

```bash
python paper/make_tables.py     # -> paper/tables/*.tex
python paper/make_figures.py    # -> paper/figures/*.png + figures/data/*.csv
cd paper && latexmk -pdf main.tex
```

`paper/` also holds the journal-side files: `highlights.docx`,
`declaration-of-interests.docx`, `cover_letter.pdf` and
`submission/README-SUBMISSION.md` (the upload checklist). All 24 entries of
`refs.bib` were resolved against Crossref on 2026-10-28 (`tancik2020` has no
DOI and is cited as NeurIPS 33, 7462–7473).

## Design principles

1. **A reference solution must be reproducible by a second method** —
   otherwise every accuracy claim is void;
2. **The metrics must expose "energy right, waveform wrong"** — otherwise the
   evaluator lies;
3. **An experiment that has not been run is reported as not run** — no
   invented numbers;
4. **Optimisation and correctness are kept apart** — every performance
   optimisation needs its own test, and is reverted if it disagrees with the
   verified implementation.

## Licence

MIT (see [`LICENSE`](LICENSE)); copyright 2026 Chunyang Chen and Yuhao Peng.
