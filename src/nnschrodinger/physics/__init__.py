"""物理层：势函数、参考解、能量与约束算子。纯 numpy 实现，不依赖深度学习框架。"""

from .spectrum import (
    HARTREE_EV,
    HBAR,
    MASS,
    ReferenceSolution,
    finite_square_well,
    get_reference,
    harmonic_oscillator,
    hydrogen_radial,
    infinite_square_well,
    list_problems,
    to_hhl_normalization,
    unit_norm_psi,
)

__all__ = [
    "HARTREE_EV",
    "HBAR",
    "MASS",
    "ReferenceSolution",
    "finite_square_well",
    "get_reference",
    "harmonic_oscillator",
    "hydrogen_radial",
    "infinite_square_well",
    "list_problems",
    "to_hhl_normalization",
    "unit_norm_psi",
]
