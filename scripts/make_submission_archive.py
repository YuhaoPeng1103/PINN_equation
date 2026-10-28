"""Build the archive uploaded to the CPC Program Library.

CPC (CPiP) requires the source code, a README describing the directory
structure, and sample input/output of at least one test run.  Everything the
repository already contains satisfies these; this script packages the tree
into one zip, skipping caches, the git directory and local web-page dumps.

Usage
-----
    python scripts/make_submission_archive.py [--out PATH]

The archive is written to ``paper/submission/nn-schrodinger-certify.zip`` by
default and is ignored by git (it duplicates the tree).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "paper" / "submission" / "nn-schrodinger-certify.zip"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "submission",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_NAMES = {"build.log"}


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if path.suffix in SKIP_SUFFIXES or path.name in SKIP_NAMES:
        return True
    # downloaded web pages kept for reference, not part of the software
    if path.name.startswith("Guide for authors") or "_files" in path.name:
        return True
    return False


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            current = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                full = current / name
                if _skip(full, ROOT):
                    continue
                arc = Path("nn-schrodinger-certify") / full.relative_to(ROOT)
                zf.write(full, arc.as_posix())
                count += 1
    # 路径含非 ASCII 字符，Windows 控制台默认 GBK 会报错，只打印文件名
    print(f"{count} files -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.out.resolve())


if __name__ == "__main__":
    main()
