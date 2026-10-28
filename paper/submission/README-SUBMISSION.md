# CPC submission package

Article type: **Computer Programs in Physics (CPiP)** — regular article.
Target: *Computer Physics Communications*.

## Files to upload at the "Attach Files" step

| File | Purpose |
|---|---|
| `paper/main.tex`, `paper/sections/*.tex`, `paper/tables/*.tex`, `paper/figures/`, `paper/refs.bib` | Editable LaTeX source (a PDF alone is not accepted as source) |
| `paper/main.pdf` | Compiled manuscript, 55 pages, for the reviewers |
| `paper/highlights.docx` (and `.txt`) | 5 highlights, each ≤ 85 characters |
| `paper/declaration-of-interests.docx` | Elsevier "Declaration of interests" document ("nothing to declare" box ticked) |
| `paper/cover_letter.pdf` (upload) / `.docx` (if an editable file is asked for) | Cover letter, 2 pages, addressed to the Editors; `.md` is the editable source |
| `paper/cover_letter.md` | Markdown source of the cover letter; regenerate the `.docx`/`.pdf` with `python paper/make_cover_letter.py` |
| `paper/submission/nn-schrodinger-certify.zip` | Program archive: source, 143-test suite, drivers, `runs/` evidence, README, MIT licence |

Rebuild the archive after any change with
`python scripts/make_submission_archive.py`
(it is not tracked by git, since it duplicates the tree).

### The "Program file(s)" upload

Editorial Manager asks for the program archive separately from the manuscript
files, because for a CPiP article the software is part of the submission and
is archived in the CPC Program Library. Upload
`nn-schrodinger-certify.zip` there — 816 files, 10.8 MB — built from the
repository root by `scripts/make_submission_archive.py`. It contains:

- `src/` — the certificate (`certify/`), the directed-rounding counter, the
  subspace certificate, the networks and the FD operators;
- `tests/` — the 143-test suite (`python tests/run_tests.py`);
- `scripts/` — the drivers that regenerate every table and figure;
- `runs/` — the JSON/CSV/npz evidence for every number quoted in the paper;
- `examples/certify_demo.py`, `README.md`, `LICENSE` (MIT), `requirements.txt`.

The "CPC Library link to program files" field in the Program Summary is left
for the Technical Editor to fill in; do not invent that URL.

## What must be entered in the submission system

- Title, author list and order: Chunyang Chen, Yuhao Peng (Peng corresponding,
  `pngyuo@cug.edu.cn`). Both are equal first authors.
- Abstract: 209 words (limit 250).
- Keywords: 7 (limit 7).
- Funding: none — the standard sentence is in the manuscript under *Funding*.
- Competing interests: nothing to declare. The declarations tool in the
  submission system must still be completed; `declaration-of-interests.docx`
  is the same statement in the form the system asks you to attach, and the
  manuscript carries the section "Declaration of competing interest" as well.
- Data availability: statement is in the manuscript; a link to
  <https://github.com/YuhaoPeng1103/PINN_equation> can be added to the
  data-linking step.
- Generative AI: declared in the manuscript (Claude, Anthropic, for drafting,
  editing and implementation).
- Suggested referees: none (stated in the cover letter).

## Compliance checks performed

- Abstract 209 words ≤ 250; keywords 7 ≤ 7; highlights 5 bullets, longest 78
  characters ≤ 85.
- References numbered `[n]` in order of appearance (`elsarticle-num`), as the
  Guide requires; 23 references, all with DOIs resolved on 2026-10-28.
- Program Summary uses the CPiP template fields: program title, CPC Library
  link (to be added by the Technical Editor), repository link, licence,
  language, supplementary material, nature of problem / solution method /
  additional comments each within 50–250 words, references.
- Licence MIT is on the journal's approved list; the archive contains the
  README with directory structure and a test run (`runs/`).
- Numbered sections, numbered equations, appendix, CRediT statement,
  declaration of competing interest, funding, data availability and the
  generative-AI declaration are all present.
- Build: `latexmk -pdf` exits 0, 0 overfull boxes, 0 undefined references.

## Not supplied (journal-side)

- CPC Library link to program files — assigned by the Technical Editor.
- Publishing agreement and licence choice — after acceptance.
