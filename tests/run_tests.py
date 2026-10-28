"""测试入口：有 pytest 用 pytest，否则退回内置的 minipytest 运行器。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("未找到 pytest，使用内置 minipytest 运行器\n")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import minipytest

        return minipytest.main()
    return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
