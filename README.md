# nnschrodinger — 用神经网络求解与求值薛定谔方程

围绕 **PINN（物理信息神经网络）求解定态薛定谔方程** 组织的完整项目：
参考解 → 网络/损失/训练 → 评测 → 可复现实验，每一层都有测试与证据。

```
  物理真值层            求解器层               实验层
  ─────────            ────────               ──────
  解析解 + Numerov  →   PINN / (x,E) 网络  →   实验一 有监督回归
  交叉校验 10/10        Rayleigh/残差损失      实验二 无监督 PINN
  纯 numpy，无依赖      经典基线（Numerov）    实验三 数据同化
```

> 详细脉络见 **[docs/project-overview.md](docs/project-overview.md)**。

---

## 这个项目解决什么

一维定态薛定谔方程

```
    Ĥψ = Eψ,      Ĥ = −½ d²/dx² + V(x)     （ħ = m = 1）
```

三个层次的问题：

1. **“真值是什么”** —— 给出可信的参考本征对，并用第二种方法交叉验证；
2. **“网络怎么解”** —— 无监督 PINN、`(x,E)` 正则化回归、数据同化三种训练范式；
3. **“解得好不好”** —— 同时报告能量、波形相似度、最坏点偏差和方程残差。

## 快速开始

```bash
pip install -e .          # 只装 numpy 核心（不需要 torch）

# ① 参考解交叉校验（纯 numpy）
python -m nnschrodinger.physics.verify
# 输出：10/10 passed

# ② 单元测试（有 pytest 用 pytest；没有则用内置运行器）
python tests/run_tests.py
# 输出：143 passed, 0 failed, 0 skipped

# ③ 证书 API 的最小示例（不需要 torch、不需要训练）
python examples/certify_demo.py
# 1D 谐振子：解析试探函数的认证区间与角度界；
# 2D 方盒简并对：单态证书拒绝（谱隙低于分辨率），簇证书接受
```

需要训练网络时：

```bash
pip install -e .[torch]   # 或 pip install torch
python scripts/exp1_regularized_network.py   # 实验一
python scripts/exp2_unregularized_pinn.py    # 实验二
python scripts/exp3_data_assimilation.py     # 实验三

# 多种子复现（种子 0/1/2，并行；每进程 4 线程）
python scripts/run_seed_sweep.py --jobs 6 --threads 4
python scripts/aggregate_seeds.py            # 输出 runs/seeds/summary.{json,md}

# 经典数值基线（Numerov，与网络指标同口径）
python scripts/classical_baseline.py

# 二维（实验六）：多种子扫描 → 滤波 + 认证汇总
python scripts/run_seed_sweep_2d.py --jobs 2
python scripts/filter_certificates2d.py
# 二维简并：单态拒绝 / 子空间证书 / 离散化 gap
python scripts/exp7_degeneracy_2d.py
python scripts/exp8_subspace_certificate.py
python scripts/exp9_resolution_boundary.py   # 单态/簇证书的分辨率边界
python scripts/discretization_study_2d.py --sizes 41 61 81 101 121
python scripts/bench_count.py                # 计数/证书耗时（runs/_bench/count_timing.json）
python scripts/dense_reference_2d.py         # 二维独立参考（稠密 LAPACK，非 ARPACK）
python scripts/appendix_failures.py --variant soft_boundary   # 附录 A 失败模式复现
python scripts/appendix_failures.py --variant unnormalized
python scripts/residual_vs_error.py runs/appendix_failures/unnormalized   # 残差 vs 真实误差
python scripts/lapack_reference_widths.py    # 经典 LAPACK 本征向量的证书宽度（§6.9 对照）
```

## 当前验证状态

| 检查项 | 结果 | 证据 |
|---|---|---|
| 参考解交叉校验（解析 ⟷ Numerov） | **10 / 10 通过** | [runs/physics_verification.txt](runs/physics_verification.txt) |
| 单元测试（含 torch 相关） | **143 / 143 通过** | [runs/tests_output_torch.txt](runs/tests_output_torch.txt) |
| 实验一 有监督回归 | 已运行 | [runs/exp1_regularized.json](runs/exp1_regularized.json) |
| 实验二 无监督 PINN | 已运行 | [runs/exp2_pinn.json](runs/exp2_pinn.json) |
| 实验三 数据同化 | 已运行 | [runs/exp3_data_assimilation_k*.json](runs/) |

### 关键实测结果

> ⚠️ 2026-10-27：发现实验二的 `Trainer` 未应用损失权重、早停只看未归一化残差，
> 下表中的实验二数字（及基于它们的“25 倍”结论）**已作废**。修复后的 3 种子
> 结果正在重跑，完成后替换。实验一/三不受影响。

| 实验 | 指标 | 数值 |
|---|---|---|
| 实验二 PINN（仅残差损失） | 能量误差 / overlap | 1.05e-1 / 0.9859 |
| 实验二 PINN（+ Rayleigh 商） | 能量误差 / overlap | 4.16e-3 / 0.99994 |
| 实验二 PINN（**+ 分层能量网格**） | 能量误差 / overlap | **8.73e-4 / 0.99990** |
| 实验三 数据同化 k=0 → 64 | 能量误差 | 2.78e-2 → **5.58e-3** |
| 实验二 激发态 n=1（正交约束） | 相对误差 / overlap | **0.74% / 0.9965** |
| 实验一 训练能量 → 插值能量 | 波形误差 | 1.4e-2 → 0.39~1.57 |
| 实验五 证书同口径训练（原型） | 离散残差 / Temple 宽度 | 0.361 / 0.260（有效） |
| 实验五 + 证书引导谱滤波 | 离散残差 / Temple 宽度 | **0.0051 / 5.2e-5** |
| 实验五 多势基态（3 势 × 3 seed，Rayleigh 商残差） | 区间包含参考值 | **12 / 12**（中位宽度 5.0e-4） |
| 实验五 激发态 n=1 / n=2（可学习 deflation） | 落到目标态且证书确认 | **8 / 9 · 8 / 9**（此前 n=1 为 0/6） |
| 实验六 二维（3 势 × 3 seed，81×81） | 滤波后证书有效 / 包含 λ_h | **9 / 9 · 9 / 9**（宽度 4.6e-5 ~ 1.4e-3） |
| 实验六 二维简并拒绝（方阱激发态） | 精确本征向量（r≈4e-12） | **正确拒绝**（谱隙 7e-9 < 分辨率 1.3e-5） |
| 实验八 二维简并**子空间**证书（81×81） | 精确对：验证重数 / 主子角界 | **2 / 8.2e-13**（真实 2.3e-16）；单成员仍拒绝 |
| 实验八 神经网络的对（4 seed，deflation） | 接受的张成对 / 塌缩对 | **3 对接受 5.6–6.3e-3**（真实 2.4–3.0e-4）；塌缩对拒绝（ρ_F=16.3 > δ=6.1） |
| 实验九 单态证书的分辨率边界（λ(x²−y²) 劈裂） | 单态转折点 / 簇证书 | 裂距/floor 从 0.95 → 1.01 时由**拒绝**变**接受**（转折点就在计数分辨率）；簇证书全程 δ=7.397、界 ~8e-13 |
| 定向舍入计数（3 势 × 121 shift，N=2001） | 认证 / 与浮点及稠密谱一致 | **361 / 363**（拒绝恰在 shift 等于本征值处） |
| 定向舍入计数（二维，81×81，可分势） | 认证 shift / 分歧 | **25 / 25 · 0**（不可分势明确拒绝） |
| 连续算子单侧界（6 个问题 × 2 档网格） | 参考值落在 [λ_h, R(ψ_h)] | 粗网格 **5/5**；细网格 **4/5**（有限深阱的反例见下） |
| 二维独立参考（稠密 LAPACK `eigh` + Rayleigh 商） | 与 ARPACK 之差 / 包含 | **≤3.1e-13 / 9 / 9**（ARPACK 只提议区间） |
| 附录 A 失败模式复现（方阱，N=41，3000 epoch） | 软边界 / 未归一化 / 硬掩码对照 | 5.0e-2、5.72 / ‖ψ‖²=6.0e-5 / 1.2e-6、1.2e-2（`runs/appendix_failures/`） |

三条最有价值的发现：

1. **分项损失的权重必须真正进入优化目标**。实验二旧代码把 `normalization=100` 按 1 生效，仅残差配置收敛到 ψ→0（评测时 E=8.6）；修复后旧“21% / 25 倍”结论撤回，新结果见 [docs/results.md](docs/results.md)。
2. **物理约束 + 少量观测确实互补**。64 个带 1% 噪声的观测点（相对 2000 个配点）就把相对能量误差压到 0.11%。
3. **`(x, E)` 双输入不能让网络"学会"能量→波函数映射**。训练能量上误差 1.4e-2，插值能量上大 1~2 个数量级；容量消融（width 64→256）**没有改善**，排除了"网络太小"。与 HHL 论文"可插值"的主张不一致。
4. **Rayleigh 商用随机配点估计会带来 1.9% 噪声**（2500 点），换成等距网格后降到 1e-8。能量项必须用确定性网格——否则能量曲线永远在抖，还会被误判成"训练不稳定"。
5. **后验证书能给出不依赖真解的误差条，但“紧”只对光滑势成立**：12 个无泄漏基态 run 的滤波后区间包含离散参考本征值 12/12，中位宽度 5.0e-4；谐振子的宽度是真实误差的 5.5–19 倍（可用），有限深阱是 946–2920 倍（不可用）。残差里的 E 用 **Rayleigh 商**而非可学习标量，可以在**不泄漏**的前提下把有限深阱的宽度从 1.87–2.72 降到 0.37–0.73。详见 [docs/certification.md](docs/certification.md)。


6. **离散认证不等于连续认证**。由 min–max，P1/Q1 插值后的采样函数给连续基态能量一个**不需要插值常数**的上界；
   但反方向没有免费的午餐：有限深阱在 N=4001 时跳点落在节点上，离散能量被推到连续值**之上**（高 3.67e-3），
   而同一张网格上把那个节点从井外改成井内（0 → −50）就把它翻到真值下方 3.73e-3——整个误差就是这一个格点，
   不是“更细的网格”能消掉的。box 相反，`sinθ ≤ θ` 给出严格的两侧界。证据：
   [runs/continuous_bound/alignment.json](runs/continuous_bound/alignment.json)。

完整报告见 **[docs/results.md](docs/results.md)**。

## 已实现的问题

| 问题 | 势能 | 能量 | 参考解来源 |
|---|---|---|---|
| `harmonic` | ½x² | n + ½ | 解析（Hermite 递推） |
| `infinite_well` | 0 (0<x<L) | n²π²/(2L²) | 解析（正弦） |
| `finite_well` | −V₀ (\|x\|<a) | 无闭式解 | Numerov + 打靶 |
| `hydrogen` | −1/\|x\| | −1/(2n²) | 解析（Laguerre） |

```python
from nnschrodinger.physics import get_reference

ref = get_reference("harmonic", 2)
print(ref.energy)        # 2.5
print(ref.psi([0.0, 1.0]))  # 归一化波函数（∫|ψ|²dx = 1）
```

## 项目结构

```
src/nnschrodinger/
├── physics/    参考解、Numerov、Jacobi、势函数、交叉校验（纯 numpy）
├── data/       HHL 论文风格的 10 个目标函数与相似度度量
└── solver/     采样 / 网络 / 损失 / 训练 / 评测
scripts/        实验入口 + 证书 / 滤波 / 多种子扫描 / 汇总 / 经典基线
tests/          单元测试（143 项，含子空间证书）
docs/           项目脉络、结构、文献、实验、工程记录
runs/           运行证据
```

完整地图见 **[docs/structure.md](docs/structure.md)**。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/project-overview.md](docs/project-overview.md) | ★ 研究问题 → 代码 → 证据的完整脉络 |
| [docs/structure.md](docs/structure.md) | 目录职责与数据流 |
| [docs/literature.md](docs/literature.md) | 三条技术主线的文献骨架 |
| [docs/experiments.md](docs/experiments.md) | 三个实验的设定、指标与预期 |
| [docs/engineering-log.md](docs/engineering-log.md) | 踩过的坑与修复方式 |
| [HANDOFF.md](HANDOFF.md) | ★★ **交接文档：换设备继续做只需读这一个文件** |
| [docs/results.md](docs/results.md) | ★ 三个实验的实测结果与解读 |
| [paper/main.pdf](paper/main.pdf) | ★★ 论文稿（CPC **正刊**稿，54 页、24 refs、143 项测试）：`cd paper && latexmk -pdf main.tex` |

## 论文稿

`paper/` 下的表格与图**全部由脚本从 `runs/*.json` 重新生成**，不手工转录：

```bash
python paper/make_tables.py     # → paper/tables/*.tex
python paper/make_figures.py    # → paper/figures/*.png + figures/data/*.csv
cd paper && latexmk -pdf main.tex
```

投稿前待补（详见 HANDOFF.md）：作者/单位、CPC 库与仓库链接。`refs.bib`
中 24 条文献的 DOI（`tancik2020` 无 DOI）已于 2026-10-28 全部对 Crossref 解析通过
（原先 `mishra2023` 与 `rump1999` 两条 DOI 记错，已改正）。

## 设计原则

1. **参考解必须能被第二种方法复现**——否则所有精度结论作废；
2. **指标必须能识别“能量准但波形错”**——否则评测器在说谎；
3. **没有运行过的实验就写“未运行”**——不编造数字；
4. **优化与正确性分开**——性能优化必须有独立测试，对不上就退回已验证实现。

## 许可证

MIT（见 `LICENSE`；版权行目前写的是 “The authors”，投稿前替换为实际作者）