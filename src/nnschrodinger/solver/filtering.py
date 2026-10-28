"""证书引导的谱滤波：把神经本征函数投影到低阶正弦子空间。

动机
----
PINN 的能量/overlap 可以很好，但 H_h 残差被高频噪声主导（二阶差分把
波数 k 的分量放大 k²）。这些高频分量对能量只有二阶贡献，却让 Temple
区间变宽甚至不可认证。本模块在 **Dirichlet 边界**下把 ψ 的内部控制点
做 DST-I 展开，截断到前 K 个模式，再重新计算证书：

    ψ_filtered = Σ_{k=1}^{K} c_k sin(π k j/(M+1))

截断阶数 K **不靠参考解选择**：扫描一组 K，取使证书宽度最小的那个。
这是完全后验的准则——宽度本身就是可计算的误差条。

边界
----
* 只适用于 Dirichlet 边界（两端 ψ=0）；这是本项目所有势的设定。
* 滤波会改变 ψ；证书认证的是**滤波后**的函数，不是网络原始输出。
  论文里必须同时报告原始与滤波后的残差/宽度。
* 若某个 K 让证书识别到另一个本征态（state_index 改变），扫描结果会
  如实报告；不要把它当成“同一个态变准了”。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .certify import EigenpairCertificate, certify_eigenpair


def dst1(values: np.ndarray) -> np.ndarray:
    """正交 DST-I（自身即逆变换），输入长度 M，输出长度 M。

    变换矩阵 S_{jk} = sqrt(2/(M+1)) sin(π(j+1)(k+1)/(M+1))，S 对称正交。
    用长度为 2(M+1) 的奇延拓 + FFT 实现，避免构造 M×M 矩阵。
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    n = x.size
    if n == 0:
        return x.copy()
    extended = np.concatenate(([0.0], x, [0.0], -x[::-1]))
    spectrum = np.fft.fft(extended)
    return math.sqrt(2.0 / (n + 1.0)) * (-0.5 * np.imag(spectrum[1 : n + 1]))


def filter_dirichlet(psi: np.ndarray, x: np.ndarray, cutoff: int) -> np.ndarray:
    """保留前 ``cutoff`` 个正弦模式，返回与 ``psi`` 等长、端点为零的数组。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    if x.shape != psi.shape:
        raise ValueError(f"x 与 psi 形状不一致: {x.shape} vs {psi.shape}")
    m = x.size - 2
    if m <= 0:
        raise ValueError("至少需要 3 个网格点")
    cutoff = int(cutoff)
    if not (1 <= cutoff <= m):
        raise ValueError(f"cutoff 必须在 [1, {m}] 内，收到 {cutoff}")
    coefficients = dst1(psi[1:-1])
    coefficients[cutoff:] = 0.0
    out = np.zeros_like(psi)
    out[1:-1] = dst1(coefficients)
    out[0] = 0.0
    out[-1] = 0.0
    return out


def transported_angle_bound(
    psi: np.ndarray,
    x: np.ndarray,
    psi_filtered: np.ndarray,
    filtered_angle_bound: float,
) -> float:
    """把**滤波后**函数的夹角界搬回**原始**网络输出（初等三角不等式）。

    记 u=ψ/‖ψ‖、u_K=ψ_K/‖ψ_K‖、φ 为目标本征函数（‖φ‖=1）。则

        sin∠(u,φ) = ‖u − (u·φ)φ‖
                   ≤ ‖u − u_K‖ + ‖u_K − (u_K·φ)φ‖ + ‖(u_K·φ)φ − (u·φ)φ‖
                   ≤ 2‖u − u_K‖ + sin∠(u_K,φ).

    第二项由证书给出（Davis–Kahan），第一项在这里直接算出。这样即使原始
    ψ 自身不可认证，也能给出它到真实本征函数的严格上界——证书约束的是
    网络输出本身，而不只是我们构造的滤波函数。

    注意这是初等引理，不是新理论；它的价值在于让“认证对象是谁”可量化。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    psi = np.asarray(psi, dtype=float).reshape(-1)
    psi_filtered = np.asarray(psi_filtered, dtype=float).reshape(-1)
    if psi.shape != x.shape or psi_filtered.shape != x.shape:
        raise ValueError("psi / psi_filtered / x 形状必须一致")
    sin_filtered = math.sin(float(filtered_angle_bound))
    if sin_filtered < 0.0:
        raise ValueError(f"夹角界必须非负，收到 {filtered_angle_bound}")
    h = float(x[1] - x[0])
    a = psi[1:-1]
    b = psi_filtered[1:-1]
    na = float(np.sqrt(h * np.sum(a**2)))
    nb = float(np.sqrt(h * np.sum(b**2)))
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("ψ 或 ψ_filtered 的范数为零，夹角无定义")
    distance = float(np.sqrt(h * np.sum((a / na - b / nb) ** 2)))
    return float(math.asin(min(1.0, 2.0 * distance + sin_filtered)))


def default_cutoffs(n_interior: int, max_cutoff: int | None = None) -> list[int]:
    """一组覆盖低阶到高阶的截断候选（去重、升序）。

    高端一直扫到 1500：有限深阱的势有跳变，基态在正弦基下收敛慢，最优
    截断落在 600；只扫到 600 会让“最优值出现在候选边界”看起来像扫描
    没扫够。实测 800/1000/1500 的宽度都比 600 大，说明 600 确实是极小值。
    """
    base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80,
            100, 150, 200, 300, 400, 600, 800, 1000, 1500]
    limit = n_interior if max_cutoff is None else min(n_interior, int(max_cutoff))
    return sorted({k for k in base if k <= limit})


@dataclass(frozen=True)
class GuidedFilterResult:
    """一次证书引导滤波的结果。"""

    cutoff: int
    psi: np.ndarray
    certificate: EigenpairCertificate
    scan: tuple[dict, ...]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "cutoff": self.cutoff,
            "certificate": self.certificate.to_dict(),
            "width": self.certificate.width,
            "scan": list(self.scan),
            "reason": self.reason,
        }


def guided_filter(
    psi: np.ndarray,
    x: np.ndarray,
    potential: np.ndarray,
    cutoffs: list[int] | None = None,
    max_cutoff: int | None = None,
) -> GuidedFilterResult:
    """扫描截断阶数，返回证书宽度最小的滤波结果。

    选择准则：在 ``valid=True`` 的候选中取 ``temple_upper − temple_lower``
    最小者；若全部无效，取残差最小者并保留原始 ψ（reason 说明原因）。
    整个过程不读取参考本征值或参考波函数。

    **状态一致性约束**：先对原始 ψ 做一次证书（``isolate=True``）得到
    ``state_index``；扫描时只接受认证到**同一个**本征态的候选。否则把
    激发态过度滤波成基态正弦波，会得到一个更窄但“答非所问”的证书。
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    n_interior = x.size - 2
    candidates = cutoffs or default_cutoffs(n_interior, max_cutoff)
    reference = certify_eigenpair(psi, x, potential, isolate=True)
    target_index = reference.state_index

    scan: list[dict] = []
    best_valid: tuple[float, int, np.ndarray, EigenpairCertificate] | None = None
    best_any: tuple[float, int, np.ndarray, EigenpairCertificate] | None = None
    for cutoff in candidates:
        filtered = filter_dirichlet(psi, x, cutoff)
        cert = certify_eigenpair(filtered, x, potential, isolate=True)
        same_state = target_index is None or cert.state_index == target_index
        entry = {
            "cutoff": int(cutoff),
            "energy": cert.energy,
            "residual_norm": cert.residual_norm,
            "width": cert.width,
            "valid": cert.valid,
            "state_index": cert.state_index,
            "same_state": same_state,
            "reason": cert.reason,
        }
        scan.append(entry)
        if cert.valid and cert.width is not None and same_state:
            if best_valid is None or cert.width < best_valid[0]:
                best_valid = (cert.width, int(cutoff), filtered, cert)
        elif same_state and (
            best_any is None or cert.residual_norm < best_any[0]
        ):
            best_any = (cert.residual_norm, int(cutoff), filtered, cert)

    if best_valid is not None:
        width, cutoff, filtered, cert = best_valid
        return GuidedFilterResult(
            cutoff=cutoff,
            psi=filtered,
            certificate=cert,
            scan=tuple(scan),
            reason="取证书宽度最小的截断阶数",
        )
    if best_any is not None:
        _, cutoff, filtered, cert = best_any
        return GuidedFilterResult(
            cutoff=cutoff,
            psi=filtered,
            certificate=cert,
            scan=tuple(scan),
            reason="所有截断阶数均不可认证；返回残差最小者",
        )
    # 没有候选（极短网格）：退回原始 ψ
    cert = certify_eigenpair(psi, x, potential, isolate=False)
    return GuidedFilterResult(0, psi.copy(), cert, tuple(scan), "没有可用的截断候选")


__all__ = [
    "GuidedFilterResult",
    "default_cutoffs",
    "dst1",
    "filter_dirichlet",
    "guided_filter",
    "transported_angle_bound",
]
