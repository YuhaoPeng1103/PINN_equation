# 多态证书（n = 0, 1, 2）

| 势 | n | runs | 原始可认证 | 证书态一致 | 滤波后包含**目标态** | 中位 E 误差 | 中位 overlap | 中位滤波宽 | 最大滤波宽 | max‖⟨φ|ψ⟩‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| 有限深阱 | 0 | 3 | 3/3 | 3/3 | 3/3 | 0.000679 | 0.999994 | 0.652 | 0.729 | — |
| 有限深阱 | 1 | 3 | 3/3 | 3/3 | 3/3 | 0.00036 | 0.999986 | 0.426 | 0.567 | 1.14e-16 |
| 有限深阱 | 2 | 3 | 3/3 | 3/3 | 3/3 | 0.000666 | 0.999973 | 0.523 | 0.683 | 5.68e-17 |
| 谐振子 | 0 | 3 | 2/3 | 3/3 | 3/3 | 4.89e-06 | 0.999998 | 3.03e-05 | 0.000926 | — |
| 谐振子 | 1 | 3 | 2/3 | 2/3 | 2/3 | 3.7e-06 | 0.999997 | 3.55e-05 | 0.0286 | 1.71e-16 |
| 谐振子 | 2 | 3 | 3/3 | 2/3 | 2/3 | 1.18e-06 | 0.999998 | 4.74e-05 | 0.0239 | 1.28e-16 |
| 无限深阱 | 0 | 3 | 2/3 | 3/3 | 3/3 | 7.11e-14 | 1 | 9.87e-12 | 9.87e-12 | — |
| 无限深阱 | 1 | 3 | 3/3 | 3/3 | 3/3 | 4.23e-10 | 1 | 7.17e-10 | 7.78e-10 | 1.14e-16 |
| 无限深阱 | 2 | 3 | 3/3 | 3/3 | 3/3 | 1.25e-09 | 1 | 2.64e-09 | 1.9e-08 | 8.53e-17 |

> 激发态组的 deflation 参考是**同一 seed 学出来的低能态**（n=1 投 ψ₀；n=2 链式投 ψ₀ 与 ψ₁），不是解析解；``max‖⟨φ|ψ⟩‖`` 是投影后再取内积的残差，理想值为 0。滤波宽度与基态组同口径（Temple 区间半宽），跨势不可直接比较——无限深阱的定义域只有 [0,1]，同样的相对精度对应更小的绝对宽度。

## 逐 run

| 势 | n | run | 证书态 | E 误差 | overlap | 滤波宽 | 包含 | ⟨φ|ψ⟩ |
|---|---|---|---|---|---|---|---|---|
| 有限深阱 | 0 | runs\seeds\certs5_rayleigh\finite_well_0\seed0\wavefunction.npz | 0 | 0.000128 | 0.999996 | 0.374 | ✓ | — |
| 有限深阱 | 0 | runs\seeds\certs5_rayleigh\finite_well_0\seed1\wavefunction.npz | 0 | 0.000771 | 0.999993 | 0.729 | ✓ | — |
| 有限深阱 | 0 | runs\seeds\certs5_rayleigh\finite_well_0\seed2\wavefunction.npz | 0 | 0.000679 | 0.999994 | 0.652 | ✓ | — |
| 有限深阱 | 1 | runs\seeds\certs5_deflated\finite_well_1\seed0\wavefunction.npz | 1 | 9.33e-05 | 0.999987 | 0.356 | ✓ | 1.14e-16 |
| 有限深阱 | 1 | runs\seeds\certs5_deflated\finite_well_1\seed1\wavefunction.npz | 1 | 0.00036 | 0.999986 | 0.426 | ✓ | 5.68e-17 |
| 有限深阱 | 1 | runs\seeds\certs5_deflated\finite_well_1\seed2\wavefunction.npz | 1 | 0.000664 | 0.999985 | 0.567 | ✓ | 5.68e-17 |
| 有限深阱 | 2 | runs\seeds\certs5_deflated2\finite_well_2\seed0\wavefunction.npz | 2 | 0.000666 | 0.999973 | 0.523 | ✓ | 5.68e-17 |
| 有限深阱 | 2 | runs\seeds\certs5_deflated2\finite_well_2\seed1\wavefunction.npz | 2 | 0.000335 | 0.999973 | 0.5 | ✓ | 5.68e-17 |
| 有限深阱 | 2 | runs\seeds\certs5_deflated2\finite_well_2\seed2\wavefunction.npz | 2 | 0.00109 | 0.999973 | 0.683 | ✓ | 2.13e-17 |
| 谐振子 | 0 | runs\seeds\certs5_rayleigh\harmonic_0\seed0\wavefunction.npz | 0 | 1.61e-06 | 1 | 3.03e-05 | ✓ | — |
| 谐振子 | 0 | runs\seeds\certs5_rayleigh\harmonic_0\seed1\wavefunction.npz | 0 | 4.89e-06 | 0.999998 | 2.68e-05 | ✓ | — |
| 谐振子 | 0 | runs\seeds\certs5_rayleigh\harmonic_0\seed2\wavefunction.npz | 0 | 7.37e-05 | 0.999988 | 0.000926 | ✓ | — |
| 谐振子 | 1 | runs\seeds\certs5_deflated\harmonic_1\seed0\wavefunction.npz | 1 | 3.7e-06 | 0.999998 | 3.55e-05 | ✓ | -8.53e-17 |
| 谐振子 | 1 | runs\seeds\certs5_deflated\harmonic_1\seed1\wavefunction.npz | 1 | 3.6e-06 | 0.999997 | 3.44e-05 | ✓ | -1.71e-16 |
| 谐振子 | 1 | runs\seeds\certs5_deflated\harmonic_1\seed2\wavefunction.npz | 2 | 0.0112 | 0.109172 | 0.0286 | ✓ | 7.46e-17 |
| 谐振子 | 2 | runs\seeds\certs5_deflated2\harmonic_2\seed0\wavefunction.npz | 2 | 5.91e-07 | 0.999998 | 4.74e-05 | ✓ | 1.28e-16 |
| 谐振子 | 2 | runs\seeds\certs5_deflated2\harmonic_2\seed1\wavefunction.npz | 2 | 1.18e-06 | 0.999998 | 4.45e-05 | ✓ | 6.13e-17 |
| 谐振子 | 2 | runs\seeds\certs5_deflated2\harmonic_2\seed2\wavefunction.npz | 1 | 0.012 | 0.109235 | 0.0239 | ✓ | 8.53e-17 |
| 无限深阱 | 0 | runs\seeds\certs5_rayleigh\infinite_well_0\seed0\wavefunction.npz | 0 | 1.84e-13 | 1 | 9.87e-12 | ✓ | — |
| 无限深阱 | 0 | runs\seeds\certs5_rayleigh\infinite_well_0\seed1\wavefunction.npz | 0 | 3.73e-14 | 1 | 9.87e-12 | ✓ | — |
| 无限深阱 | 0 | runs\seeds\certs5_rayleigh\infinite_well_0\seed2\wavefunction.npz | 0 | 7.11e-14 | 1 | 9.87e-12 | ✓ | — |
| 无限深阱 | 1 | runs\seeds\certs5_deflated\infinite_well_1\seed0\wavefunction.npz | 1 | 4.61e-10 | 1 | 7.78e-10 | ✓ | -1.14e-16 |
| 无限深阱 | 1 | runs\seeds\certs5_deflated\infinite_well_1\seed1\wavefunction.npz | 1 | 4.23e-10 | 1 | 7.17e-10 | ✓ | 0 |
| 无限深阱 | 1 | runs\seeds\certs5_deflated\infinite_well_1\seed2\wavefunction.npz | 1 | 2.13e-11 | 1 | 7.36e-11 | ✓ | 1.14e-16 |
| 无限深阱 | 2 | runs\seeds\certs5_deflated2\infinite_well_2\seed0\wavefunction.npz | 2 | 9.26e-09 | 1 | 1.9e-08 | ✓ | 8.53e-17 |
| 无限深阱 | 2 | runs\seeds\certs5_deflated2\infinite_well_2\seed1\wavefunction.npz | 2 | 8.25e-11 | 1 | 3.04e-10 | ✓ | 8.53e-17 |
| 无限深阱 | 2 | runs\seeds\certs5_deflated2\infinite_well_2\seed2\wavefunction.npz | 2 | 1.25e-09 | 1 | 2.64e-09 | ✓ | 2.84e-17 |
