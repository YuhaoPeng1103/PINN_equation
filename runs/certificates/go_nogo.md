# A 线 go/no-go 汇总

* 无泄漏 run：12
* 滤波后证书有效且状态一致：12
* 区间包含同一离散算子的参考本征值：12
* 中位滤波后宽度：0.000499584837749506
* 中位 宽度/|E_filt−λ_h|：38.014183336963264
* 夹角界覆盖真实 arccos(overlap)：12/12（比较对象为离散本征向量）

## 判定

* 包含率 12/12 = 100%
* 包含率满足 Go 门槛（≥95%）。
* 紧度未达 Go 门槛：中位 宽度/误差 = 38.0（最大 2.92e+03；Go 要求 <10，No-go 是 >1e3）。
* 注意：误差本身接近舍入下限的 run（3 个）会让比值失去意义，应看绝对宽度与包含性。

## 逐 run

| run | potential | state | cert_state | certified | contains | width | width/|ΔE| | angle | angle(ψ,φ_h) | arccos(overlap_analytic) | rel L2 change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| runs\seeds\certs5_rayleigh\finite_well_0\seed0\wavefunction.npz | finite_well | 0 | 0 | ✓ | ✓ | 0.374 | 2.92e+03 | 0.56 | 0.00104 | 0.00273 | 6.48e-05 |
| runs\seeds\certs5_rayleigh\finite_well_0\seed1\wavefunction.npz | finite_well | 0 | 0 | ✓ | ✓ | 0.729 | 946 | 0.835 | 0.00297 | 0.00373 | 0.000236 |
| runs\seeds\certs5_rayleigh\finite_well_0\seed2\wavefunction.npz | finite_well | 0 | 0 | ✓ | ✓ | 0.652 | 960 | 0.777 | 0.00205 | 0.00352 | 0.000134 |
| runs\seeds\certs5_rayleigh\harmonic_0\seed0\wavefunction.npz | harmonic | 0 | 0 | ✓ | ✓ | 3.03e-05 | 18.8 | 0.00389 | 0.000549 | 0.000874 | 0.000681 |
| runs\seeds\certs5_rayleigh\harmonic_0\seed1\wavefunction.npz | harmonic | 0 | 0 | ✓ | ✓ | 2.68e-05 | 5.47 | 0.00366 | 0.00175 | 0.00177 | 0.000248 |
| runs\seeds\certs5_rayleigh\harmonic_0\seed2\wavefunction.npz | harmonic | 0 | 0 | ✓ | ✓ | 0.000926 | 12.6 | 0.0215 | 0.00466 | 0.00492 | 0.00159 |
| runs\seeds\certs5_rayleigh\infinite_well_0\seed0\wavefunction.npz | infinite_well | 0 | 0 | ✓ | ✓ | 9.87e-12 | 53.7 | 7.28e-10 | 9.5e-13 | 2.96e-05 | 2.96e-05 |
| runs\seeds\certs5_rayleigh\infinite_well_0\seed1\wavefunction.npz | infinite_well | 0 | 0 | ✓ | ✓ | 9.87e-12 | 265 | 7.48e-10 | 9.5e-13 | 5.76e-06 | 5.76e-06 |
| runs\seeds\certs5_rayleigh\infinite_well_0\seed2\wavefunction.npz | infinite_well | 0 | 0 | ✓ | ✓ | 9.87e-12 | 139 | 7.55e-10 | 9.5e-13 | 3.77e-06 | 3.77e-06 |
| runs\seeds\certs\learned0\seed0\wavefunction.npz | — | 0 | 0 | ✓ | ✓ | 7.28e-05 | 17.4 | 0.00603 | 0.0013 | 0.0052 | 0.00503 |
| runs\seeds\certs\learned0\seed1\wavefunction.npz | — | 0 | 0 | ✓ | ✓ | 0.00147 | 22.3 | 0.0271 | 0.00288 | 0.0049 | 0.00543 |
| runs\seeds\certs\learned0\seed2\wavefunction.npz | — | 0 | 0 | ✓ | ✓ | 0.00171 | 9.8 | 0.0292 | 0.00767 | 0.00918 | 0.00506 |
