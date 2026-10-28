# 实验设计与预期

三个实验对应 HHL 论文里“量子线路求解薛定谔方程”这套工作的三组核心主张，
以及本仓库补充的一组工程结论。每个实验都有明确的**可证伪指标**。

---

## 实验一：正则化网络 + 能量插值

**脚本**：`scripts/exp1_regularized_network.py`

| 项目 | 设定 |
|---|---|
| 输入 | `(x, E)`，E 作为第二通道 |
| 输出 | `ψ(x; E)`，归一化到 max\|ψ\| = 1 |
| 训练能量 | E ∈ {0.5, 1.5, 2.5, 3.5}（谐振子前 4 态） |
| 训练域 | x ∈ [−6, 6]，每步 2500 点 |
| 损失 | 监督平方误差 + L2 |

**要验证的主张**：网络在训练集未覆盖的能量（E = 1.0, 2.0, 3.0）上
能否给出正确的本征函数？

**对照基线（关键）**：对同样的“插值能量”，经典做法是把相邻两个已知本征函数
做线性插值。如果网络不显著优于这个基线，就不存在“网络学会了能量→波函数”的证据，
它只是做了平滑插值。**报告必须同时给出两者**。

**输出**：`runs/exp1_regularized.json`，字段 `energies[E].max_abs_err` 与 `is_interpolated`。

---

## 实验二：非正则化 PINN（无监督）

**脚本**：`scripts/exp2_unregularized_pinn.py`

| 项目 | 设定 |
|---|---|
| 输入 | 只用 `x`（能量是标量超参，不进入网络） |
| 损失 | 残差 + 100·归一化 + 10·Rayleigh 商（默认） |
| 网络 | 4 层 tanh MLP + 傅里叶特征 |
| 早停 | 固定网格上的**加权总损失**（与训练目标同口径） |

**对照配置**（`scripts/run_seed_sweep.py` 统一扫描，种子 0/1/2）：

| 配置 | 残差中的 E | Rayleigh 项 | 采样 |
|---|---|---|---|
| `residual_only` | 固定为参考值 | 无 | — |
| `residual_learned` | **可学习**（初值 1.0） | 无 | — |
| `residual_learned_e15` | 可学习（初值 1.5） | 无 | — |
| `rayleigh_learned` | **可学习**（初值 1.0） | 有 | 固定等距网格 |
| `rayleigh_random` | 固定为参考值 | 有 | 每步随机点 |
| `rayleigh_grid` | 固定为参考值 | 有 | 固定等距网格 |

> “固定 E”等价于把答案告诉网络，只能检验优化是否收敛，不能检验
> “残差损失是否偏爱基态”。真正的全无监督对照是
> `residual_learned`（无 Rayleigh）vs `rayleigh_learned`（有 Rayleigh）。
> 早期版本没有这个对照，因而得出了过强的结论，详见工程记录第 13 条。
>
> 四个关键配置（`residual_only`、`residual_learned`、`residual_learned_e15`、
> `rayleigh_learned`）跑 10000 epochs；`rayleigh_random` / `rayleigh_grid` /
> `rayleigh_random_gradbalance` 跑 5000，以便在同一预算下比较采样方式。
> 激发态 n=0/1/2/3 跑 5000 epochs（width=128、depth=4）。

**要验证的主张**：完全没有标签时，物理约束足以把网络推回本征态。

**必须警惕的失败模式**：Rayleigh 商收敛到正确值，但波函数包含与真解正交的污染分量。
这是因为一阶微扰下，正交污染对能量泛函的贡献是二次的——**能量准 ≠ 波形对**。
因此评测同时报告 `overlap`、`inf_norm`、`residual_rms`。

**与实验一的对照**：同一个本征态问题，有监督 vs 无监督的收敛步数与最终精度。

**输出**：`runs/exp2_pinn.json`。

---

## 实验三：数据同化（物理 + 少量观测）

**脚本**：`scripts/exp3_data_assimilation.py`

| 项目 | 设定 |
|---|---|
| 问题 | 无限深方势阱基态（E = π²/2） |
| 观测 | k ∈ {0, 16, 64} 个随机点的 ψ，带 σ = 0.01 高斯噪声 |
| 损失 | 残差 + 归一化 + 边界 + λ·监督 |

**要验证的主张**：少量观测（相对 2000 个配点）能显著加速收敛并消除多解性；
k = 0 时退化为实验二，作为基线。

**预期趋势**：k 增大 → `overlap` ↑、`energy_abs_err` ↓、`inf_norm` ↓。
若趋势不成立，说明物理项与观测项权重失衡（正是 `grad_balance_norm` 要解决的问题）。

**输出**：`runs/exp3_data_assimilation_k{k}.json`。

---

## 实验四：后验证书（A 线）

**脚本**：`src/nnschrodinger/solver/certify.py`、`scripts/certify_saved.py`；
训练入口复用 `scripts/run_seed_sweep.py --groups certs`。

| 项目 | 设定 |
|---|---|
| 证书对象 | 离散 H_h（二阶中心差分 + Dirichlet 边界） |
| 能量界 | Weinstein（无条件）+ Temple（单态区间，Sturm 计数验证） |
| 波函数界 | Davis–Kahan：`sin θ ≤ ‖r‖/γ`，γ 由单态区间给出 |
| 区间来源 | `isolating_interval` 自动二分相邻本征值；不依赖 Numerov/参考谱 |
| 参考值 | `scipy.linalg.eigh_tridiagonal`（独立于 Sturm） |
| 对照 | `learned0`（无泄漏）vs `state1_deflated`（解析基 deflation，有泄漏） |

**要验证的主张**：证书能（a）包含同一离散算子的参考本征值；
（b）在无真解时识别“能量好但波形是杂波”的解；（c）宽度与真实误差同量级
或只大 1–2 个数量级，而不是完全无用。

**判定标准**：见 [`certification.md`](certification.md) §5.3。若大量 run
无法给出紧证书，则 A 线降级为“评测协议/负结果”，主攻 B（泄漏审计）。

**输出**：`runs/seeds/certs/**/wavefunction.npz` +
`runs/certificates/summary.{json,md}`。

---

## 实验五：证书同口径训练（原型）

**脚本**：`scripts/exp5_certificate_training.py`

实验四的早期观察是：PINN 的连续残差/能量可以很好，但证书用的离散残差
‖H_hψ − Eψ‖ 很大（边界层 + 高频噪声被 1/h² 放大），Temple 区间宽到无用。
本实验把**证书实际使用的量**直接放进训练目标，检验证书能否随之变紧：

| 损失项 | 公式 | 默认权重 |
|---|---|---|
| 离散残差 | mean[(H_hψ − Eψ)²]，H_h 为 2001 点二阶差分 + Dirichlet | 1 |
| 归一化 | (h·Σψ² − 1)² | 100 |
| 边界 | ψ(−6)² + ψ(6)² | 100 |
| Rayleigh | h·ΣψH_hψ / h·Σψ²（可选） | 10 |

网格固定为 2001 点，与证书完全一致；早停只看误差型项
（残差 + 归一化 + 边界），不把能量目标混进验证。支持
`--potential harmonic|infinite_well|finite_well` 与 `--state k`
（激发态默认关闭 Rayleigh 项，因为它会把训练拉回基态）。训练后保存
`wavefunction.npz`，再用 `certify_saved.py` 与同规模 exp2 基线比较宽度。

**要验证的主张**：在能量误差/overlap 相当的前提下，同口径训练给出的
Temple 宽度显著小于 exp2（autograd 残差 + 随机配点）。若成立，
“证书驱动训练/早停”就是一个可写进论文的方法组件；若不成立，
说明宽证书的根源不是训练口径，而是网络本身的高频分量，A 线需要重新定位。

**早期结果（3000 epoch，128×4，谐振子基态，修正能量项之后）**：
E=0.500014、overlap=0.999998、离散残差 0.361、证书有效、宽度 0.260。
比 exp2 的“不可认证”好，但宽度仍是 |η − λ_h|（2.1e-5）的 ~1.3e4 倍。
（修正前同一配置为 E=0.500047、残差 0.328、宽度 0.216；差异来自能量项
使用 Dirichlet 延拓，见工程记录 18。）

**再加一步后验滤波**（`scripts/filter_certificates.py`）：把 ψ 投影到
前 K 个正弦模式并重新认证，K 由“证书宽度最小”后验选出（K=15）：
残差 0.361 → **0.00511**，宽度 0.260 → **5.22e-5**，夹角界 0.369 →
**0.00511 rad**，区间仍包含独立参考本征值；滤波造成的相对 L2 改变
8.8e-5。对不可认证的 exp2 小模型同样有效（残差 1.12 → 0.0107，
宽度 2.3e-4）。详见 [certification.md](certification.md) §5.6。

**长训练（15000 epoch）**：原始残差 0.0421、宽度 3.5e-3；滤波（K=18）后
残差 1.9e-4、宽度 **7.6e-8**、夹角界 1.9e-4。宽度已小于 H_h 与连续
算子之间的离散偏差（~1e-6），认证对象必须写成 H_h。注意这个 run 用的是
修正前的能量项，ψ 与证书仍有效，但重跑待做。

**输出**：`runs/certificate_training/exp5_certificate_training.json` +
`wavefunction.npz`；滤波对比表在 `runs/certificates_filtered/summary.md`。

---

## 统一评测指标

| 指标 | 阈值（本项目认为“可用”） | 说明 |
|---|---|---|
| energy_abs_err | < 1e-2（Hartree） | 比化学精度（~1.6e-3）略宽 |
| overlap | > 0.99 | 波形整体一致 |
| inf_norm | < 0.05 | 无局部伪影 |
| residual_rms | < 1e-2 | 确实满足方程 |

指标实现见 `src/nnschrodinger/solver/evaluate.py`，陷阱测试见
`tests/test_evaluate.py::test_orthogonal_noise_is_caught`。

---

## 当前状态

| 实验 | 代码 | 是否已运行 | 结果 |
|---|---|---|---|
| 参考解交叉校验 | ✅ | ✅ | 10/10 通过 |
| 单元测试 | ✅ | ✅ | 100/100 通过（含 torch 路径） |
| 实验一 | ✅ | ✅ | 多种子均值见 `runs/seeds/summary.md` |
| 实验二 | ✅ | 🔄 重新运行中 | 旧单次结果因损失未加权而作废（工程记录 13） |
| 实验三 | ✅ | 🔄 多种子重跑中 | k=64 的单次结果 5.58e-3 |
| 实验五（证书训练） | ✅ | ✅ 原型 | 3000ep: r 0.361 / 宽 0.260 → 滤波 5.22e-5；15000ep: r 0.042 / 宽 3.5e-3 → 滤波 7.6e-8 |

实测细节与解读见 **[results.md](results.md)**。