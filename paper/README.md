# Paper build

The manuscript is a single LaTeX project (`main.tex`), written for
*Computer Physics Communications* (elsarticle, preprint, 12 pt, author–year).

## Build

```bash
cd paper
latexmk -pdf main.tex        # or: pdflatex main; bibtex main; pdflatex main; pdflatex main
```

The build is clean: 0 errors, 0 overfull boxes, 0 undefined references.

`python check_refs.py` (run from `paper/`) reports duplicate labels,
labels that are never referenced, and `\ref` targets without a label. The
LaTeX log remains authoritative.

## What is generated and what is not

| Path | Status |
|---|---|
| `main.tex`, `sections/*.tex` | written by hand |
| `tables/*.tex` | **generated** by `python paper/make_tables.py` from `runs/**/*.json` — never edit by hand |
| `figures/*.png`, `figures/data/*.csv` | **generated** by `python paper/make_figures.py` |
| `refs.bib` | written by hand |

Every number in every table therefore traces back to a JSON record produced by
a driver in `scripts/`; the manuscript text quotes those same numbers.

## Regenerating everything

```bash
python scripts/run_seed_sweep.py --jobs 6 --threads 4   # 1D training (~30-45 s/run)
python scripts/run_seed_sweep_2d.py --jobs 2            # 2D training (~108 s/run)
python scripts/filter_certificates.py
python scripts/filter_certificates2d.py
python scripts/exp7_degeneracy_2d.py
python scripts/exp8_subspace_certificate.py
python scripts/exp9_resolution_boundary.py
python scripts/discretization_study_2d.py --sizes 41 61 81 101 121
python scripts/go_nogo.py ; python scripts/leakage_audit.py
python scripts/multistate_report.py
python scripts/dense_reference_2d.py        # independent 2D reference (dense LAPACK)
python scripts/bench_count.py               # count/certificate/scan timings
python scripts/appendix_failures.py --variant soft_boundary
python scripts/appendix_failures.py --variant unnormalized
python scripts/residual_vs_error.py runs/appendix_failures/unnormalized
python scripts/lapack_reference_widths.py   # classical-solver widths quoted in Sec. 6.9
python scripts/directed_count_report.py     # directed-rounding counts (Sec. 6.1-6.3)
python scripts/continuous_bound.py          # variational bound + 2D brackets (Sec. 6.4-6.5)
python scripts/continuous_bound.py --alignment   # finite well, O(h) sign experiment
python paper/make_tables.py ; python paper/make_figures.py
python paper/make_cover_letter.py          # cover_letter.docx + .pdf from the .md
```

Note: run one sweep at a time and keep `--jobs` at 2–3; concurrent sweeps have
been observed to exhaust memory on the reference machine.

## Before submission

Everything on the old checklist is done; what remains is entered in the
submission system, not in the files:

- [x] author list, affiliation, e-mail in `main.tex`
- [x] repository link in the Program summary and in Code availability
- [x] abstract ≤ 250 words (209), keywords ≤ 7 (7), highlights ≤ 85 chars (5 bullets)
- [x] numbered references (`[n]`, order of appearance) per the Guide for Authors
- [x] CRediT, Funding, Data availability and generative-AI declarations
- [x] cover letter — `cover_letter.md` source, `cover_letter.pdf` (upload) and
      `.docx` built by `python paper/make_cover_letter.py`; no suggested referees
- [x] `declaration-of-interests.docx` — "nothing to declare" statement
- [ ] completing the declarations tool in the submission system
- [ ] choosing open access / licence option after acceptance

The upload list and the fields to fill in the system are in
[`submission/README-SUBMISSION.md`](submission/README-SUBMISSION.md); the
software archive is built with `python scripts/make_submission_archive.py`
(git-ignored, since it duplicates the tree).

Note: the bibliography is now numeric, so `main.bbl` must be regenerated after
any edit to `refs.bib` — run `pdflatex; bibtex main; pdflatex; pdflatex`, and
check that `main.bbl` is not empty (a zero-byte `.bbl` only shows up as one
natbib warning).
