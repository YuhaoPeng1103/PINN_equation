"""数据层：目标函数、网格、相似度度量。全部纯 numpy。"""

from .targets import (
    HHL_FUNCTIONS,
    HHL_GRID_POINTS,
    HHL_H,
    HHL_RANGE,
    TARGET_FUNCTIONS,
    TARGET_MODES,
    make_grid,
    make_target,
    similarity,
)

__all__ = [
    "HHL_FUNCTIONS",
    "HHL_GRID_POINTS",
    "HHL_H",
    "HHL_RANGE",
    "TARGET_FUNCTIONS",
    "TARGET_MODES",
    "make_grid",
    "make_target",
    "similarity",
]
