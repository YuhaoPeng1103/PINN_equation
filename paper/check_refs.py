"""Static consistency check for the manuscript sources.

Reports duplicate labels, labels never referenced, and \\ref targets that no
label defines.  It is a development aid; the LaTeX log remains authoritative.
"""

import glob
import re
import sys

LABEL = re.compile(r"\\label\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]*)\}")


def main() -> int:
    files = sorted(glob.glob("sections/*.tex")) + ["main.tex"]
    text = "\n".join(open(f, encoding="utf-8").read() for f in files)
    labels = LABEL.findall(text)
    refs = set(REF.findall(text))
    problems = 0

    dups = sorted({lab for lab in labels if labels.count(lab) > 1})
    if dups:
        problems += len(dups)
        print("duplicate labels:", dups)

    unused = [lab for lab in labels if lab not in refs]
    print(f"{len(labels)} labels, {len(unused)} never referenced")
    for lab in unused:
        print("  unreferenced:", lab)

    missing = sorted(refs - set(labels))
    if missing:
        problems += len(missing)
        print("references with no label:")
        for lab in missing:
            print("  missing:", lab)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
