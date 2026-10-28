"""Render paper/cover_letter.md into an upload-ready .docx and .pdf.

The Markdown source is the single source of truth.  Paragraphs are rejoined
before inline markup is parsed, because the source is hard-wrapped and both
emphasis and the quoted title span line breaks.

Usage (from the repository root)::

    python paper/make_cover_letter.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

PAPER = Path(__file__).resolve().parent
SRC = PAPER / "cover_letter.md"
DOCX = PAPER / "cover_letter.docx"
PDF = PAPER / "cover_letter.pdf"

SIGNATURE = "Yours sincerely,"


def read_blocks() -> list[str]:
    """Return the letter as blocks: one per paragraph, one per signature line.

    The heading is dropped.  Body paragraphs are hard-wrapped in the source and
    are rejoined; the signature block keeps its line structure, because the
    affiliation and e-mail must not run together on one line.
    """
    text = SRC.read_text(encoding="utf-8")
    raw = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    assert raw[0].lstrip("# ").lower() == "cover letter", raw[0]
    blocks: list[str] = []
    for block in raw[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if any("cug.edu.cn" in line for line in lines):
            blocks.extend(lines)
        else:
            blocks.append(" ".join(lines))
    return blocks


def add_runs(paragraph, line: str) -> None:
    """Add text to a paragraph, honouring **bold** and *italic*."""
    for token in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", line):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        else:
            paragraph.add_run(token)


def build_docx(blocks: list[str]) -> None:
    """Write the .docx; blocks already carry the signature line structure."""
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    salutation, body = blocks[0], blocks[1:]
    add_runs(doc.add_paragraph(), salutation)

    sig_at = next(i for i, b in enumerate(body) if b == SIGNATURE)
    for block in body[:sig_at]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, block)

    # signature block: no space between the four lines
    doc.add_paragraph()
    for block in body[sig_at + 1:]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        add_runs(p, block)

    doc.save(str(DOCX))


def build_pdf(blocks: list[str]) -> None:
    pandoc = shutil.which("pandoc") or r"D:/Anaconda/Library/bin/pandoc"
    salutation, body = blocks[0], blocks[1:]
    sig_at = next(i for i, b in enumerate(body) if b == SIGNATURE)
    # a trailing backslash is pandoc's hard line break: keeps the signature
    # lines on consecutive lines inside one paragraph
    md = "\n\n".join([salutation] + body[:sig_at] + [SIGNATURE]) + "\n\n"
    md += "\\\n".join(body[sig_at + 1:]) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "cover_letter.md"
        src.write_text(md, encoding="utf-8", newline="\n")
        cmd = [
            pandoc, str(src), "-o", str(PDF),
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "mainfont=Times New Roman",
            "-V", "lang=en-US",      # enables hyphenation; without it the
                                     # justified lines open large gaps
            "-V", "colorlinks=true",
            "-V", 'urlcolor=black',
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    blocks = read_blocks()
    build_docx(blocks)
    build_pdf(blocks)
    print(f"wrote {DOCX.name} ({DOCX.stat().st_size} bytes)")
    print(f"wrote {PDF.name} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
