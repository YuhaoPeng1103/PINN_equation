"""深度学习框架的可用性探测。

项目**只用 PyTorch 一种后端**（HHL 论文用的是 PyTorch；其 QNN 分支虽然用
PennyLane，但经典部分完全可以在 torch 中复现）。如果环境里没有 torch，
数据管线与参考解依然可用，训练脚本会打印清晰的安装提示后退出。
"""

from __future__ import annotations

from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def require_torch(what: str = "训练") -> Any:
    """返回 torch 模块，若不可用则抛出带安装提示的 ImportError。"""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            f"{what} 需要 PyTorch。请安装：pip install torch "
            "（或 pip install -e .[torch]）。参考解与数据校验不依赖 torch。"
        ) from exc
    return torch


def describe_environment() -> dict[str, Any]:
    """打印环境信息，便于复现实验时定位问题。"""
    import platform

    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        info["numpy"] = "missing"
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = "missing"
    return info


__all__ = ["has_torch", "require_torch", "describe_environment"]
