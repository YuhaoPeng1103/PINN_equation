"""打印项目结构（跳过缓存与依赖目录）。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".venv", "venv", "build", "dist", "runs"}
SKIP_PREFIXES = ("pip-", ".tmp_")


def walk(directory: Path, prefix: str = "") -> None:
    entries = sorted(
        (
            p
            for p in directory.iterdir()
            if p.name not in SKIP and not p.name.startswith(SKIP_PREFIXES) and not p.name.endswith(".egg-info")
        ),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    for index, entry in enumerate(entries):
        last = index == len(entries) - 1
        connector = "└── " if last else "├── "
        suffix = "/" if entry.is_dir() else ""
        print(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            walk(entry, prefix + ("    " if last else "│   "))


def main() -> int:
    print(f"{ROOT.name}/")
    walk(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
