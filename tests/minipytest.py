"""极简测试运行器：在无法安装 pytest 的受限环境里跑同一套测试文件。

设计原则：测试文件只使用 pytest 的一小部分 API——`pytest.approx`、`pytest.raises`、
`pytest.fixture`、`pytest.skip` / `pytest.importorskip`、`pytest.mark.skipif`。
本模块提供这些 API 的等价实现，从而让 `python tests/minipytest.py` 与 `pytest`
都能运行同一批测试（有 pytest 时优先用 pytest，它支持更丰富的功能）。

用法：
    python tests/minipytest.py            # 跑 tests/ 下所有 test_*.py
    python tests/minipytest.py -k numerov # 只跑名字匹配的测试
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (str(ROOT / "src"), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# pytest 兼容层
# ---------------------------------------------------------------------------
class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        try:
            return all(
                math.isclose(float(a), float(b), rel_tol=self.rel, abs_tol=self.abs)
                for a, b in zip(_flat(other), _flat(self.expected))
            )
        except Exception:
            return False

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


def _flat(value):
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flat(item))
        return out
    if hasattr(value, "tolist"):
        return _flat(value.tolist())
    return [value]


class _Raises:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"未抛出期望的异常 {self.expected.__name__}")
        return issubclass(exc_type, self.expected)


class _Skip(Exception):
    pass


class _Mark:
    def skipif(self, condition, reason=""):
        def decorator(func):
            func._skip = bool(condition)
            func._skip_reason = reason
            return func

        return decorator


class _Pytest:
    approx = staticmethod(lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs))
    raises = staticmethod(lambda exc: _Raises(exc))
    mark = _Mark()

    @staticmethod
    def skip(reason=""):
        raise _Skip(reason)

    @staticmethod
    def importorskip(name, *args, **kwargs):  # noqa: D401
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise _Skip(f"缺少模块 {name}") from exc

    @staticmethod
    def fixture(*args, **kwargs):
        scope = kwargs.get("scope", "function")

        def decorator(func):
            func._fixture_scope = scope
            return func

        return decorator


sys.modules.setdefault("pytest", _Pytest())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def _is_test(name: str) -> bool:
    return name.startswith("test_")


def _run_module(path: Path, keyword: str | None) -> tuple[int, int, int]:
    module = importlib.import_module(f"test_{path.stem[5:]}" if path.stem.startswith("test_") else path.stem)
    fixtures = {
        name: fn for name, fn in vars(module).items() if inspect.isfunction(fn) and hasattr(fn, "_fixture_scope")
    }
    cache: dict[str, object] = {}
    passed = failed = skipped = 0
    for name, func in list(vars(module).items()):
        if not (_is_test(name) and inspect.isfunction(func)):
            continue
        if keyword and keyword not in f"{path.stem}::{name}":
            continue
        if getattr(func, "_skip", False):
            skipped += 1
            continue
        try:
            params = inspect.signature(func).parameters
            kwargs = {}
            for pname in params:
                if pname in fixtures:
                    if pname not in cache or fixtures[pname]._fixture_scope == "function":
                        cache[pname] = fixtures[pname]()
                    kwargs[pname] = cache[pname]
            func(**kwargs)
            passed += 1
            print(f"  PASS {path.stem}::{name}")
        except _Skip as exc:
            skipped += 1
            print(f"  SKIP {path.stem}::{name} ({exc})")
        except Exception:
            failed += 1
            print(f"  FAIL {path.stem}::{name}")
            traceback.print_exc()
    return passed, failed, skipped


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    keyword = None
    if "-k" in argv:
        keyword = argv[argv.index("-k") + 1]
    files = sorted(p for p in TESTS.glob("test_*.py"))
    total_p = total_f = total_s = 0
    for path in files:
        print(f"[{path.name}]")
        try:
            p, f, s = _run_module(path, keyword)
        except _Skip as exc:
            # 整个模块在导入时被跳过（例如缺少 torch）
            print(f"  SKIP 整个模块（{exc}）")
            print()
            continue
        total_p += p
        total_f += f
        total_s += s
    print()
    print(f"{total_p} passed, {total_f} failed, {total_s} skipped")
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())