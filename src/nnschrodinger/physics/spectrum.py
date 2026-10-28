"""一维定态薛定谔方程的参考解：解析式 + Numerov 数值解。

全项目统一约定
--------------
    Ĥψ = Eψ,    Ĥ = −½ d²/dx² + V(x),    ħ = m = 1

* **能量是 E（不是波数 k）。** 谐振子 E_n = n + ½，对应 k_n = sqrt(2n+1)。
  HHL 论文以 k 作“能量”（对 V = x²/2 有 k_n = sqrt(2n+1)），换算 E = k²/2：
  基态 k = 1 ↔ E = 0.5，第四个激发态 k = 3 ↔ E = 4.5。
* **波函数归一为 ∫|ψ|²dx = 1**（单位范数）。HHL 论文用 max|ψ| = 1，
  换算见 ``to_hhl_normalization``。
* **符号约定**：首尾样本中绝对值较大者取正（奇宇称态即右端为正）。

正交基函数与 wavefunction 的选取全部来自解析解：

| 问题 | 势 V(x) | 能量 | 波函数 | 域 |
|---|---|---|---|---|
| harmonic | ½x² | n + ½ | e^{−x²/2}·H_n(x) | ±3·sqrt(2n+1) |
| infinite_well | 0 (0<x<L) | n²π²/(2L²) | sqrt(2/L)·sin(nπx/L) | [0, L]（n ≥ 1） |
| finite_well | −V₀ (|x|<a) | Numerov | Numerov | [−8a, 8a] |
| hydrogen | −1/|x| | −1/(2n²) | R_{nl}(r) | [0, 20n] |

"参考解"分两类：解析解可直接写出来的（谐振子/无限深/氢原子）用解析式；
有限深方势阱没有闭式解，用 ``physics.numerov`` 的 Numerov + 打靶法给出数值参考解。
两者都在 ``physics.verify`` 中用**另一套方法**交叉验证（解析 ↔ Numerov）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .numerov import numerov_spectrum

# 物理常数（原子单位制 ħ = m_e = 1）
HBAR: float = 1.0
MASS: float = 1.0
# 1 Hartree = 27.211386 eV，用于把无量纲能量换算成 eV 展示
HARTREE_EV: float = 27.211386


@dataclass(frozen=True)
class ReferenceSolution:
    """某个问题的第 n 个本征态的参考解。

    Attributes
    ----------
    name   : 可读的问题名，如 "harmonic_oscillator"。
    n      : 0 基量子数（谐振子/氢原子）；对方势阱为“量子数 − 1”。
    energy : 无量纲能量 E（Hartree）。
    psi    : 波函数求值器 ``psi(x) -> ndarray``，满足 ∫|ψ|²dx = 1。
    domain : 训练/绘图使用的空间区间 (lo, hi)。
    """

    name: str
    n: int
    energy: float
    psi: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float]

    @property
    def energy_ev(self) -> float:
        return self.energy * HARTREE_EV

    def to_hhl_scale(self, x: np.ndarray) -> np.ndarray:
        """返回 max|ψ| = 1 约定下的波函数（HHL 论文的归一化）。"""
        vals = np.asarray(self.psi(np.asarray(x, dtype=float)), dtype=float)
        return to_hhl_normalization(vals)


# ---------------------------------------------------------------------------
# 归一化 / 符号
# ---------------------------------------------------------------------------
def unit_norm_psi(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """把 psi 的 L2 范数归一为 1：psi / sqrt(∫|psi|²dx)（等距网格梯形求和）。"""
    dx = float(np.asarray(x)[1] - np.asarray(x)[0])
    norm = math.sqrt(float(np.sum(np.asarray(psi) ** 2)) * dx)
    return np.asarray(psi) / norm


def to_hhl_normalization(psi_unit: np.ndarray) -> np.ndarray:
    """单位范数 → HHL 论文约定的 max|ψ| = 1。"""
    psi_unit = np.asarray(psi_unit)
    peak = float(np.max(np.abs(psi_unit)))
    if peak == 0.0:
        raise ValueError("zero wavefunction cannot be renormalized")
    return psi_unit / peak


def fix_sign(psi: np.ndarray) -> np.ndarray:
    """统一符号：首尾样本中绝对值较大者取正。"""
    psi = np.asarray(psi)
    ends = (abs(psi[0]), abs(psi[-1]))
    probe = 0 if ends[0] > ends[1] else len(psi) - 1
    return psi if psi[probe] >= 0 else -psi


# ---------------------------------------------------------------------------
# 精确解
# ---------------------------------------------------------------------------
def _hermite(n: int, x: np.ndarray) -> np.ndarray:
    """Hermite 多项式 H_n 的三项递推：H₀=1, H₁=2x, H_{k+1}=2xH_k − 2kH_{k−1}。"""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_prev, h_cur = np.ones_like(x), 2.0 * x
    for k in range(1, n):
        h_prev, h_cur = h_cur, 2.0 * x * h_cur - 2.0 * k * h_prev
    return h_cur


def harmonic_oscillator(n: int, domain: tuple[float, float] | None = None) -> ReferenceSolution:
    """谐振子 V(x) = ½ω²x²（ω = 1）：E_n = n + ½，ψ_n ∝ e^{−x²/2}·H_n(x)。

    默认在 ±3·sqrt(2n+1) 截断（已保证 ψ < 1e-4）。domain 参数仅用于扩展绘图/求值范围。
    """
    span = max(6.0, 3.0 * math.sqrt(2.0 * n + 1.0))

    def psi(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        return np.exp(-0.5 * xs**2) * _hermite(n, xs)

    grid = np.linspace(-span, span, 8001)
    scale = 1.0 / math.sqrt(float(np.sum(psi(grid) ** 2)) * float(grid[1] - grid[0]))

    def evaluate(xs: np.ndarray) -> np.ndarray:
        return scale * psi(xs)

    bounds = (-span, span) if domain is None else (float(domain[0]), float(domain[1]))
    return ReferenceSolution("harmonic_oscillator", n, n + 0.5, evaluate, bounds)


def infinite_square_well(n: int, L: float = 1.0) -> ReferenceSolution:
    """无限深方势阱 V = 0 (0<x<L)，边界外 ψ ≡ 0。参数 n 是 **1 基**量子数。

    E_n = n²π²/(2L²)，ψ_n = sqrt(2/L)·sin(nπx/L)。
    """
    if n < 1:
        raise ValueError("n must be >= 1 for the square well")

    def evaluate(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        inside = (xs >= 0.0) & (xs <= L)
        return np.where(inside, math.sqrt(2.0 / L) * np.sin(n * math.pi * xs / L), 0.0)

    return ReferenceSolution(
        "infinite_square_well", n - 1, n * n * math.pi**2 / (2 * L * L), evaluate, (0.0, float(L))
    )


def finite_square_well(depth: float = 50.0, half_width: float = 1.0, n_states: int = 1) -> ReferenceSolution:
    """有限深方势阱 V(x) = −depth (|x| < a)，0 (else)；返回第 n_states 个束缚态（1 基）。

    没有闭式解，用 Numerov + 打靶法在 [−8a, 8a] 上数值求解（见 physics.numerov）。
    这个“数值参考解”只用于测试/绘图，绝不参与训练。
    """
    a = float(half_width)
    lo, hi = -8.0 * a, 8.0 * a

    def potential(x: np.ndarray) -> np.ndarray:
        return np.where(np.abs(np.asarray(x, dtype=float)) < a, -float(depth), 0.0)

    spec = numerov_spectrum(potential, lo, hi, (-float(depth), 0.0), n_states, n_points=8001)
    energy = float(spec.energies[n_states - 1])
    values = fix_sign(spec.vectors[:, n_states - 1])

    def evaluate(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        inside = (xs >= lo) & (xs <= hi)
        return np.where(inside, np.interp(xs, spec.grid, values), 0.0)

    return ReferenceSolution("finite_square_well", n_states - 1, energy, evaluate, (lo, hi))


def _generalized_laguerre(k: int, alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """广义 Laguerre 多项式 L_k^α 的三项递推：
    (k+1)L_{k+1} = (2k+1+α−x)L_k − (k+α)L_{k−1}。"""

    def poly(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if k == 0:
            return np.ones_like(x)
        lm1, l0 = np.ones_like(x), 1.0 + alpha - x
        for j in range(1, k):
            lm1, l0 = l0, ((2 * j + 1 + alpha - x) * l0 - (j + alpha) * lm1) / (j + 1)
        return l0

    return poly


def hydrogen_radial(n: int, l: int, a0: float = 1.0) -> ReferenceSolution:
    """氢原子径向波函数 R_{nl}(r)，能量 E_n = −1/(2n²) Hartree（a0 = 1）。

    R_{nl}(r) = sqrt((2/(n a0))³·(n−l−1)!/(2n·(n+l)!))·e^{−ρ/2}·ρ^l·L^{2l+1}_{n−l−1}(ρ)，ρ = 2r/(n a0)。

    注意：R_{nl} 在 r → 0 处为 r^l 阶；HHL 论文用“点源发射”的 ψ/R 求值，
    对 l = 0 在原点奇异（∝ 1/r）——这是 docs/literature.md 中记录的一个重要边界案例。
    """
    if not (0 <= l < n):
        raise ValueError("require 0 <= l < n")
    pre = math.sqrt((2.0 / (n * a0)) ** 3 * math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l)))
    lag = _generalized_laguerre(n - l - 1, 2 * l + 1)

    def radial(r: np.ndarray) -> np.ndarray:
        rho = 2.0 * np.asarray(r, dtype=float) / (n * a0)
        return pre * np.exp(-rho / 2.0) * rho**l * lag(rho)

    rmax = 20.0 * n * a0
    grid = np.linspace(0.0, rmax, 8001)
    scale = 1.0 / math.sqrt(float(np.trapezoid(radial(grid) ** 2, grid)))

    def evaluate(xs: np.ndarray) -> np.ndarray:
        return scale * radial(np.abs(np.asarray(xs, dtype=float)))

    return ReferenceSolution(f"hydrogen_{n}{'spdf'[l]}", n - 1, -1.0 / (2.0 * n * n), evaluate, (0.0, rmax))


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------
def get_reference(problem: str, n: int = 0, **kwargs) -> ReferenceSolution:
    """一行式入口：``get_reference("harmonic", 2)``。

    对 infinite_well / finite_well，n 是 0 基序号，内部换成 1 基量子数，
    使所有问题对外统一使用“0 基 n”约定。
    """
    key = problem.lower()
    if key in ("harmonic", "harmonic_oscillator"):
        return harmonic_oscillator(n, **kwargs)
    if key in ("infinite_well", "infinite_square_well"):
        kwargs.pop("domain", None)
        return infinite_square_well(n + 1, **kwargs)
    if key in ("finite_well", "finite_square_well"):
        kwargs.pop("domain", None)
        return finite_square_well(n_states=n + 1, **kwargs)
    if key == "hydrogen":
        return hydrogen_radial(kwargs.pop("n_quantum", n + 1), kwargs.pop("l", 0), **kwargs)
    raise KeyError(f"unknown problem {problem!r}")


def list_problems() -> list[str]:
    return ["harmonic", "infinite_well", "finite_well", "hydrogen"]


__all__ = [
    "HBAR",
    "MASS",
    "HARTREE_EV",
    "ReferenceSolution",
    "get_reference",
    "list_problems",
    "harmonic_oscillator",
    "infinite_square_well",
    "finite_square_well",
    "hydrogen_radial",
    "unit_norm_psi",
    "to_hhl_normalization",
    "fix_sign",
]
