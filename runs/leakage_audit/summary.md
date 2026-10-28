# 泄漏对照

| 组 | n | 原始可认证 | 滤波后包含 | 中位 E 误差 | 中位 overlap | 中位滤波宽度 | 最大滤波宽度 |
|---|---|---|---|---|---|---|---|
| deflated | 3 | 0/3 | 3/3 | 0.109 | 0.9446 | 0.218 | 0.486 |
| learned | 9 | 5/9 | 9/9 | 7.22e-06 | 0.999998 | 4.89e-05 | 2.72 |
| rayleigh | 9 | 7/9 | 9/9 | 4.89e-06 | 0.999998 | 3.03e-05 | 0.729 |
| reference | 9 | 8/9 | 9/9 | 6.42e-06 | 0.999997 | 2.59e-05 | 0.628 |

## 逐 run

| 组 | run | 势 | 目标态 | 证书态 | E 误差 | overlap | 原始 r | 原始有效 | 原始宽 | K | 滤波宽 | 包含 | rel L2 改变 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned | runs\seeds\certs5\finite_well_0\seed0\wavefunction.npz | finite_well | 0 | 0 | 0.00743 | 0.999946 | 3.15 | ✗ | — | 200 | 1.87 | ✓ | 0.00137 |
| learned | runs\seeds\certs5\finite_well_0\seed1\wavefunction.npz | finite_well | 0 | 0 | 0.00419 | 0.99998 | 1.73 | ✓ | 3.95 | 150 | 2.07 | ✓ | 0.0025 |
| learned | runs\seeds\certs5\finite_well_0\seed2\wavefunction.npz | finite_well | 0 | 0 | 0.00576 | 0.999973 | 3.66 | ✗ | — | 80 | 2.72 | ✓ | 0.00703 |
| learned | runs\seeds\certs5\harmonic_0\seed0\wavefunction.npz | harmonic | 0 | 0 | 1.67e-06 | 1 | 0.159 | ✓ | 0.0505 | 18 | 1.6e-05 | ✓ | 6.58e-05 |
| learned | runs\seeds\certs5\harmonic_0\seed1\wavefunction.npz | harmonic | 0 | 0 | 7.22e-06 | 0.999998 | 0.956 | ✓ | 1.83 | 15 | 4.89e-05 | ✓ | 0.00062 |
| learned | runs\seeds\certs5\harmonic_0\seed2\wavefunction.npz | harmonic | 0 | 0 | 5.62e-05 | 0.99999 | 1.19 | ✗ | — | 15 | 0.000425 | ✓ | 0.000981 |
| learned | runs\seeds\certs5\infinite_well_0\seed0\wavefunction.npz | infinite_well | 0 | 0 | 1.34e-13 | 1 | 7.65 | ✗ | — | 1 | 9.87e-12 | ✓ | 4.89e-05 |
| learned | runs\seeds\certs5\infinite_well_0\seed1\wavefunction.npz | infinite_well | 0 | 0 | 9.33e-14 | 1 | 0.527 | ✓ | 0.297 | 1 | 9.87e-12 | ✓ | 1.34e-05 |
| learned | runs\seeds\certs5\infinite_well_0\seed2\wavefunction.npz | infinite_well | 0 | 0 | 1.33e-14 | 1 | 0.192 | ✓ | 0.0392 | 1 | 9.87e-12 | ✓ | 4.3e-06 |
| reference | runs\seeds\certs5_refenergy\finite_well_0\seed0\wavefunction.npz | finite_well | 0 | 0 | 9.32e-05 | 0.999996 | 0.539 | ✓ | 0.385 | 800 | 0.337 | ✓ | 1.92e-05 |
| reference | runs\seeds\certs5_refenergy\finite_well_0\seed1\wavefunction.npz | finite_well | 0 | 0 | 6.94e-05 | 0.999996 | 0.582 | ✓ | 0.448 | 600 | 0.345 | ✓ | 5.59e-05 |
| reference | runs\seeds\certs5_refenergy\finite_well_0\seed2\wavefunction.npz | finite_well | 0 | 0 | 0.000875 | 0.999993 | 0.883 | ✓ | 1.03 | 600 | 0.628 | ✓ | 8.76e-05 |
| reference | runs\seeds\certs5_refenergy\harmonic_0\seed0\wavefunction.npz | harmonic | 0 | 0 | 1.57e-06 | 1 | 0.178 | ✓ | 0.0637 | 18 | 1.38e-05 | ✓ | 0.000138 |
| reference | runs\seeds\certs5_refenergy\harmonic_0\seed1\wavefunction.npz | harmonic | 0 | 0 | 6.42e-06 | 0.999997 | 0.372 | ✓ | 0.277 | 18 | 2.59e-05 | ✓ | 0.000271 |
| reference | runs\seeds\certs5_refenergy\harmonic_0\seed2\wavefunction.npz | harmonic | 0 | 0 | 5.34e-05 | 0.999989 | 1.87 | ✗ | — | 15 | 0.000471 | ✓ | 0.00169 |
| reference | runs\seeds\certs5_refenergy\infinite_well_0\seed0\wavefunction.npz | infinite_well | 0 | 0 | 7.9e-14 | 1 | 3.14 | ✓ | 10.6 | 1 | 9.87e-12 | ✓ | 2.21e-05 |
| reference | runs\seeds\certs5_refenergy\infinite_well_0\seed1\wavefunction.npz | infinite_well | 0 | 0 | 1.47e-13 | 1 | 1.76 | ✓ | 3.32 | 1 | 9.87e-12 | ✓ | 1.01e-05 |
| reference | runs\seeds\certs5_refenergy\infinite_well_0\seed2\wavefunction.npz | infinite_well | 0 | 0 | 1.4e-13 | 1 | 0.172 | ✓ | 0.0317 | 1 | 9.87e-12 | ✓ | 3.87e-06 |
| rayleigh | runs\seeds\certs5_rayleigh\finite_well_0\seed0\wavefunction.npz | finite_well | 0 | 0 | 0.000128 | 0.999996 | 0.646 | ✓ | 0.553 | 600 | 0.374 | ✓ | 6.48e-05 |
| rayleigh | runs\seeds\certs5_rayleigh\finite_well_0\seed1\wavefunction.npz | finite_well | 0 | 0 | 0.000771 | 0.999993 | 1.44 | ✓ | 2.76 | 400 | 0.729 | ✓ | 0.000236 |
| rayleigh | runs\seeds\certs5_rayleigh\finite_well_0\seed2\wavefunction.npz | finite_well | 0 | 0 | 0.000679 | 0.999994 | 1.12 | ✓ | 1.65 | 600 | 0.652 | ✓ | 0.000134 |
| rayleigh | runs\seeds\certs5_rayleigh\harmonic_0\seed0\wavefunction.npz | harmonic | 0 | 0 | 1.61e-06 | 1 | 0.57 | ✓ | 0.65 | 15 | 3.03e-05 | ✓ | 0.000681 |
| rayleigh | runs\seeds\certs5_rayleigh\harmonic_0\seed1\wavefunction.npz | harmonic | 0 | 0 | 4.89e-06 | 0.999998 | 0.268 | ✓ | 0.143 | 15 | 2.68e-05 | ✓ | 0.000248 |
| rayleigh | runs\seeds\certs5_rayleigh\harmonic_0\seed2\wavefunction.npz | harmonic | 0 | 0 | 7.37e-05 | 0.999988 | 1.88 | ✗ | — | 15 | 0.000926 | ✓ | 0.00159 |
| rayleigh | runs\seeds\certs5_rayleigh\infinite_well_0\seed0\wavefunction.npz | infinite_well | 0 | 0 | 1.84e-13 | 1 | 4.48 | ✗ | — | 1 | 9.87e-12 | ✓ | 2.96e-05 |
| rayleigh | runs\seeds\certs5_rayleigh\infinite_well_0\seed1\wavefunction.npz | infinite_well | 0 | 0 | 3.73e-14 | 1 | 0.0235 | ✓ | 0.000588 | 1 | 9.87e-12 | ✓ | 5.76e-06 |
| rayleigh | runs\seeds\certs5_rayleigh\infinite_well_0\seed2\wavefunction.npz | infinite_well | 0 | 0 | 7.11e-14 | 1 | 0.0241 | ✓ | 0.000622 | 1 | 9.87e-12 | ✓ | 3.77e-06 |
| deflated | runs\seeds\certs\state1_deflated\seed0\wavefunction.npz | — | 1 | 1 | 0.109 | 0.9446 | 6.1 | ✗ | — | 18 | 0.218 | ✓ | 0.00348 |
| deflated | runs\seeds\certs\state1_deflated\seed1\wavefunction.npz | — | 1 | 1 | 0.259 | 0.863552 | 15.7 | ✗ | — | 15 | 0.486 | ✓ | 0.00604 |
| deflated | runs\seeds\certs\state1_deflated\seed2\wavefunction.npz | — | 1 | 1 | 0.0848 | 0.95737 | 6.45 | ✗ | — | 15 | 0.172 | ✓ | 0.00268 |
