# 交接文档：PINN 求解薛定谔方程

> 这份文档的目标是：**换一台设备，只读这一个文件就能继续干**。
> 它包含：项目是什么、现在到哪一步、怎么跑起来、有哪些实测结论、踩过哪些坑、下一步做什么。

| 项 | 值 |
|---|---|
| 文档生成时间 | 2026-10-28 |
| 源码规模 | 73 个 Python 文件 / 13,646 行（`src` + `scripts` + `tests`）；9 个 Markdown 文档 / 约 3.9k 行（含本文件，不含 `runs/`） |
| 验证过的环境 | Python 3.11.9 · torch 2.12.0+cpu · numpy 2.4.6 · Windows（CPU） |
| 测试状态 | **143 / 143 通过** |
| 参考解校验 | **10 / 10 通过** |
| 实验状态 | 实验一/二/三、实验五（1D 证书）、实验六（**2D 证书**）已跑完 |

---

## 0. 三十秒摘要

这是一个**用 PINN 求解定态薛定谔方程（1D/2D）并用后验证书认证结果**的项目。
成稿论文在 `paper/main.pdf`（54 页）：主张不是“更准的求解器”，而是
**一套把神经网络本征对变成可审计陈述的协议**，以及这套协议在什么条件下失效。
定位见 §7。

五条有实测数据的结论：

1. **“残差损失是否足以定基态”需要可学习的 E 才能回答**——旧实验把真实 E 喂给残差项、且损失权重未生效，E=0.6047 只是欠收敛的非本征态；修复后 `residual_learned` 的三个种子分别停在 n=2/4/2（E=3.98±1.26），残差损失本身不选基态；加 Rayleigh 项的 `rayleigh_learned` 回到 E≈0.502、overlap≈0.99998。
2. **Rayleigh 商用随机配点估计会带 1.9% 噪声**——换成等距网格后降到 1e-8，能量误差再降 4.8 倍。这是"能量曲线抖动"的真正原因，不是训练不稳定。
3. **`(x, E)` 双输入不能插值**——训练能量误差 1.8e-2，插值能量 0.25~1.79；网络加宽到 256 也无改善，容量不是原因。
4. **正交投影只能作用于 ψ，不能作用于 ψ′**——否则导数和动能项被削掉一半，能量算成 1.25 而不是 1.5。
5. **后验证书能给出不依赖真解的误差条，但“紧”只对光滑势成立**——12 个无泄漏基态 run 包含参考值 12/12，中位宽度/误差 38（自定 Go 门槛 <10）；谐振子滤波后宽度是真实误差的 5.5–19 倍（可用），有限深阱 946–2920 倍（不可用）；原始输出 7/12 可认证。第一版“无泄漏”配置其实是泄漏的（参考 E 进了残差），同 epoch 对照显示泄漏会把有限深阱的证书收紧 3–4 倍。
6. **两个方法性修复让“无泄漏”与“多态”同时成立（2026-10-28）**——(a) 残差里的 E 换成当前 ψ 的 **Rayleigh 商**后，有限深阱的滤波宽度从 1.87–2.72 降到 **0.37–0.73**（无泄漏地拿回泄漏的收益），有限深阱 3/3 原始可认证；(b) 用**同一 seed 学出来的低能态**做硬正交（不读解析解）求激发态：n=1 有 **8/9** 落到目标态且被证书确认（此前 0/6），n=2 用链式 deflation（同时投 ψ₀、ψ₁）也是 **8/9**；两次失败都是 `harmonic` 的 seed2——它在 n=1 跳到 ψ₂，错误沿链传播到 n=2，证书如实报 `state_index`。
7. **证书可以“搬回”网络原始输出（初等引理，非新定理）**——`sin∠(ψ,φ) ≤ 2‖u−u_K‖ + sin∠(ψ_K,φ)`，把滤波位移与滤波后证书相加，就能给未滤波的 ψ 一个严格夹角上界。谐振子 5.3e-3、无限深阱 5.9e-5、有限深阱 0.560（界成立但无信息量）。这回答了“你认证的是自己构造的函数，不是网络”。

8. **二维协议已跑通（2026-10-28）**——稀疏 LU 的 Sylvester 惯性计数
   （`permc_spec="NATURAL"` + 禁用主元交换 + 显式检查置换矩阵）替代一维
   Sturm 计数；3 势 × 3 seed、81×81，9/9 原始与滤波证书有效、9/9 包含
   独立求出的 λ_h，滤波宽度 4.6e-5 ~ 1.4e-3。**简并态被正确拒绝**：方阱
   第一激发态用 ARPACK 精确本征向量（残差 4e-12）输入，证书仍返回
   `valid=False`（谱隙 7.1e-9 < 双精度分辨率 1.3e-5）。二维网络对 H_h 的
   误差 1.7e-6 ~ 4.6e-5，比 H_h 自身的离散化 gap（N=81 时 3.5e-4）小一个
   量级；方阱外推谱 9.8696042046 与 π² 差 2e-7。

9. **简并子空间证书（2026-10-28）**——单态证书在简并簇上必须拒绝；新增
   `certify_subspace.py` 把认证对象换成 m 维不变子空间：重数由已验证的
   Sturm/Sylvester 计数**测出**，簇区间 (a,b) 满足 N(b)−N(a)=m，Rayleigh–Ritz
   给出 θ_i、ρ_i，主子角界 sinΘ ≤ ρ_F/δ（δ 为簇边距）。方阱 81×81 上，精确
   简并对被认证（重数 2，界 8e-13，真实 2e-16）；单成员仍拒绝。四个独立训练
   的 seed 中三个塌缩到同一代表元、一个落到正交伙伴；证书接受含伙伴的对
   （界 5.6–6.3e-3，真实 2.4–3.0e-4），拒绝塌缩对（ρ_F=16.3 > δ=6.1），对
   “第二方向主要是噪声”的对给出有效但松两个量级的界。对基态+seed0 同时
   deflation 的伙伴与独立发现的伙伴重叠 0.99995。11 项新测试锁定重数、
   包含性、基无关性与拒绝路径；加上定向舍入计数与连续算子界共 20 项（11 + 9），总计 143 项。

10. **分辨率边界（2026-10-28）**——实验九用 $V_\lambda=\lambda(x^2-y^2)$
   对方阱做受控劈裂（裂距 0.076λ，平均能量不动），十三个 λ 上同时跑单态与
   簇证书。单态证书的转折点在**计数分辨率**上：裂距 7.6e-6（比 round-off
   高七个数量级，ARPACK 轻易分辨）仍被拒绝；裂距/floor = 0.95 拒绝、1.01
   接受，转折点就在 floor=1.28e-5 处（常规提议内缩 10%，谱隙 = 0.9×裂距；
   贴到 floor 时退回二分兜底，谱隙 ≈ 裂距）。簇证书全程接受、δ=7.397 不变。
   结论：“简并态”不是正确的类别，**可分辨性**才是；子空间证书把这个
   障碍整体换掉了。驱动 `scripts/exp9_resolution_boundary.py`，
   输出 `runs/resolution2d/boundary.{json,md}`，论文 §6.7 Table 8。
> ⚠️ **2026-10-27 更正**：结论 1 的旧证据（E=0.6047、误差降 25 倍）已作废——
> 实验二的 `Trainer` 当时没有应用损失权重，且残差项直接用了真实 E。正确对照
> 是把 E 设为可学习参数的 `residual_learned`；修复后的多种子结果见
> `runs/seeds/summary.md`。结论 2 的“随机 Rayleigh 估计有 1.9% 噪声”是
> 估计量本身的性质，仍然成立，但“它是能量误差主因”的旧推断需要等新扫描确认。
> 详见工程记录第 13 条。

**核心不等式都是经典的**（Temple/Weinstein/Davis–Kahan、Sturm 计数），但
2026-10-28 增加了一个新的**构造**：把认证对象从单个特征对换成 m 维不变子空间，
用已验证的惯性计数**测出重数**而不是假定它，给出簇的双侧围道与主子角界
（`certify_subspace.py`，论文 §4），从而把“简并态只能拒绝”变成“简并态作为子空间
可以认证”，并进一步用受控劈裂（实验九）把边界定位到计数分辨率上。它的价值不在
不等式本身，而在于让谱簇成为可计算、可拒绝、可验收的对象。

论文按 **CPC 正刊（regular article）** 起草（`paper/main.tex`，含 Program
summary 块）：贡献是协议 + 子空间构造 + 审计，而不是新不等式，这一点在摘要
与 §1 里写明。**明确的边界**（论文里作为 limitation 写，不回避）：证书只对
$H_h$ 严格；计数是浮点 + 护栏而非区间算术；$H_h$ 装不出来时（高维）不适用；
与其它 PINN 实现的系统对比未做。

---

## 1. 项目结构

```
基于神经网络的薛定谔方程求值问题/
├── README.md                     入口：快速开始 + 关键结论
├── HANDOFF.md                    ★ 本文件
├── pyproject.toml                打包配置（torch 为可选 extra）
├── requirements.txt
├── Makefile                      make verify / test / lint
│
├── src/nnschrodinger/
│   ├── physics/                  纯 numpy，不依赖 torch
│   │   ├── spectrum.py           参考解：解析式 + Numerov（谐振子/方势阱/氢原子）
│   │   ├── numerov.py            Numerov 积分 + 打靶求本征值
│   │   ├── eigen.py              稠密 Jacobi 特征分解（无 scipy 的替代品）
│   │   ├── potentials.py         势函数库
│   │   ├── operators.py          梯度/拉普拉斯 numpy 实现
│   │   └── verify.py             ★ 交叉校验：解析 ⟷ Numerov
│   ├── data/targets.py           HHL 论文的 10 个目标函数 + 相似度度量
│   └── solver/                   部分依赖 torch
│       ├── backends.py           torch 可用性探测
│       ├── inputs.py             采样：配点 / (x,E) 成对 / 按跳变点分段
│       ├── networks.py           MLP(+傅里叶特征) / HHL 网络 / 数值基线
│       ├── losses.py             ★ 物理项/数据项/正则项，含投影版本
│       ├── train.py              ★ 训练器：早停 + 梯度均衡
│       ├── evaluate.py           ★ 指标：能量误差/overlap/inf-norm/残差
│       ├── certify.py            ★ 后验证书：Sturm 计数 + Weinstein/Temple/夹角界
│       ├── certify2d.py          ★ 二维证书：Sylvester 惯性计数 + 同样的界
│       ├── certify_subspace.py   ★ 子空间（簇）证书：验证重数 + 主子角界
│       ├── filtering.py          ★ 证书引导的 DST 谱滤波
│       └── filtering2d.py        二维滤波
│
├── scripts/
│   ├── exp1_regularized_network.py   实验一：有监督 (x,E) 回归 + 能量插值
│   ├── exp2_unregularized_pinn.py    实验二：无监督 PINN（支持激发态 deflation、可学习 E）
│   ├── exp3_data_assimilation.py     实验三：物理 + 少量观测
│   ├── exp5_certificate_training.py  实验五：证书同口径训练
│   ├── exp6_certificate_training_2d.py 实验六：二维证书同口径训练（支持 deflation）
│   ├── exp7_degeneracy_2d.py         实验七：二维简并的拒绝
│   ├── exp8_subspace_certificate.py  实验八：二维简并子空间的认证
│   ├── exp9_resolution_boundary.py   实验九：单态/簇证书的分辨率边界
│   ├── certify_saved.py              批量对已保存波函数算证书
│   ├── filter_certificates.py        证书引导滤波 + 重新认证
│   ├── run_seed_sweep.py             多种子并行扫描（子进程 + 线程限制）
│   ├── _seed_worker.py               单次 (实验, 配置, 种子) worker
│   ├── aggregate_seeds.py            均值 ± 标准差 → summary.{json,md}
│   ├── classical_baseline.py         Numerov 经典基线（与网络同口径）
│   └── tree.py                       打印项目结构
│
├── tests/                        143 项测试
│   ├── run_tests.py              统一入口（有 pytest 用 pytest）
│   ├── minipytest.py             内置极简运行器（无 pytest 时用）
│   └── test_*.py                 16 个测试文件（含 test_certify2d / test_certify_subspace）
│
├── docs/
│   ├── project-overview.md       ★ 项目脉络：研究问题 → 代码 → 证据
│   ├── results.md                ★ 全部实测结果与解读
│   ├── engineering-log.md        ★ 36 个踩坑记录（重要）
│   ├── structure.md              目录职责与数据流
│   ├── experiments.md            实验设计与预期
│   ├── certification.md          ★ 后验证书：数学、协议、判决标准
│   └── literature.md             文献骨架
│
├── examples/
│   └── certify_demo.py         最小可运行示例（无 torch）：1D 认证 + 2D 拒绝/簇认证
│
└── runs/                         运行证据
    ├── environment.txt           环境快照
    ├── physics_verification.txt  参考解交叉校验表
    ├── tests_output_torch.txt    完整测试输出
    ├── classical_baseline.json   Numerov 基线指标
    ├── seeds/                    多种子原始结果 + manifest + summary.{json,md}
    ├── exp1_regularized.json     旧单次结果
    ├── exp2_pinn.json            旧单次结果（因损失未加权而作废，见工程记录 13）
    ├── exp2_states/              旧激发态结果（同上）
    ├── exp2_gradbalance/         旧梯度均衡对照（同上）
    └── exp3_data_assimilation_k{0,16,64}.json
```

---

## 2. 新设备上如何跑起来

### 2.1 环境

```bash
# 只需要 numpy 就能跑参考解与校验（不需要 torch）
pip install -e .

# 需要训练网络时
pip install -e .[torch]   # 或 pip install torch
```

已验证组合：**Python 3.11.9 + torch 2.12.0+cpu + numpy 2.4.6**。
torch 2.x 应都能用；numpy 需 ≥1.24（用到了 `np.trapezoid`）。

> **踩坑提醒**：本机有两套 Python。装 torch 的是 `D:\Python-3.11`，
> 而某些工具默认的 3.12 没有 torch。新设备上务必确认 `python -c "import torch"` 能过。

### 2.2 三条命令验证环境

```bash
# 1) 参考解交叉校验 -> 应输出 10/10 passed
python -m nnschrodinger.physics.verify

# 2) 单元测试 -> 应输出 143 passed
python tests/run_tests.py

# 3) 三个实验（需 torch）
python scripts/exp1_regularized_network.py --epochs 3000
python scripts/exp2_unregularized_pinn.py --epochs 5000 --state 0
python scripts/exp3_data_assimilation.py --epochs 2000 --n-obs 0 16 64

# 4) 多种子复现（种子 0/1/2，6 进程 × 4 线程）与汇总
python scripts/run_seed_sweep.py --jobs 6 --threads 4
python scripts/aggregate_seeds.py
python scripts/classical_baseline.py
```

**耗时参考**（CPU）：实验二基态约 8 分钟；激发态约 8 分钟；实验三每个 k 约 2 分钟；实验一约 2 分钟。

### 2.3 注意：pytest 可能装不上

如果 `pip install pytest` 因为沙箱限制失败，用内置运行器即可：

```bash
python tests/minipytest.py            # 跑全部
python tests/minipytest.py -k losses  # 只跑匹配的
```

`tests/run_tests.py` 会自动探测：有 pytest 用 pytest，没有就退回 minipytest。

---

## 3. 实测结果（全部有 JSON 落盘）

### 3.1 参考解交叉校验：10/10

| 问题 | n | E_ref | E_numeric | \|ΔE\| | max\|Δψ\| |
|---|---|---|---|---|---|
| 谐振子 | 0 | 0.5 | 0.50000000 | 1.38e-12 | 1.16e-8 |
| 谐振子 | 1 | 1.5 | 1.50000000 | 1.64e-12 | 9.88e-8 |
| 谐振子 | 2 | 2.5 | 2.50000000 | 7.77e-14 | 8.17e-9 |
| 谐振子 | 3 | 3.5 | 3.50000000 | 9.84e-12 | 1.29e-9 |
| 无限深阱 | 0-2 | 4.93/19.74/44.41 | — | 4.9e-3~4.4e-2 | 1.1e-3~3.3e-3 |
| 有限深阱 | 0-2 | −48.98/−45.92/−40.85 | — | ~1e-12 | ~1e-12 |

无限深阱的对照是"深 2000 的有限阱"，两者差约 E/V₀ —— 是**物理差异**不是数值误差。

### 3.2 实验一：正则化网络 + 能量插值

| E | 类型 | max\|ψ_model − ψ_ref\| |
|---|---|---|
| 0.50 | 训练 | 1.42e-2 |
| 1.50 | 训练 | 1.43e-2 |
| 2.50 | 训练 | 1.49e-2 |
| 3.50 | 训练 | 1.52e-2 |
| 1.00 | 插值 | 3.95e-1 |
| 2.00 | 插值 | 1.57 |
| 3.00 | 插值 | 3.85e-1 |

**容量消融**（同一设置，只改网络规模）：

| 配置 | 训练误差 | 插值误差范围 |
|---|---|---|
| width=64, depth=3 | 1.52e-2 | 0.39 ~ 1.57 |
| width=128, depth=4 | 1.70e-1 | 0.10 ~ 1.40 |
| width=256, depth=5 | 1.52e-2 | 0.23 ~ 1.90 |

结论：**加宽网络无改善**，排除容量解释。插值失败是方法层面问题。

### 3.3 实验二：无监督 PINN（谐振子）

> ⚠️ **本节旧表已作废**（损失权重未生效 + 早停只看未归一化残差）。
> 修复后的 3 种子结果见 `runs/seeds/summary.md`，正在重跑；不要引用下表。

| 配置（旧，作废） | 能量 | 绝对误差 | overlap |
|---|---|---|---|
| 仅残差损失 | 0.6047 | 1.05e-1 | 0.9859 |
| + Rayleigh 商（随机配点） | 0.5042 | 4.16e-3 | 0.99994 |
| + 梯度均衡 | 0.5037 | 3.71e-3 | 0.99992 |
| **+ 分层能量网格（最终）** | **0.50087** | **8.73e-4** | **0.99990** |

**激发态**（正交约束 / deflation；旧表同样作废，见工程记录 14）：

| 态（旧，作废） | E_ref | E_model | 相对误差 | overlap |
|---|---|---|---|---|
| n=0 | 0.5 | 0.500873 | 0.17% | 0.999901 |
| n=1 | 1.5 | 1.511106 | 0.74% | 0.996496 |

### 3.4 实验三：数据同化（无限深阱，E_ref = π²/2 ≈ 4.9348）

| 观测点数 k | 能量误差 | overlap |
|---|---|---|
| 0（纯 PINN） | 2.78e-2 | 0.9999996 |
| 16 | 1.93e-2 | 0.9999993 |
| 64 | **5.58e-3** | 0.9999994 |

单调下降，64 个带 1% 噪声的观测点把相对误差压到 0.11%。

---

## 4. 关键 bug 与教训（务必先读）

完整记录见 [docs/engineering-log.md](docs/engineering-log.md)，这里只列最要命的六条：

### 4.0 分项损失的权重可能被静默忽略

`Trainer` 以前对 `{"terms": {...}}` 直接做 `sum(terms)`，完全不看
`LossWeights`。实验二因此把 `normalization=100` 当成 1 用，残差项靠 ψ→0
就能把损失压到最低；早停又只看未归一化残差，进一步锁死平凡解。
修复：`TrainConfig.term_weights` + 验证损失改成与训练目标同口径的加权和。
**新增回归测试**：`test_term_weights_are_applied`。详见工程记录第 13 条。

### 4.1 Rayleigh 商不能用随机配点估计

`Losses.energy` 用 `mean(...)` 当积分。样本若是每步重采的随机点，这个 mean 就是
蒙特卡洛估计，噪声 ∝ 1/√N：

| N | 估计值标准差 |
|---|---|
| 2500 随机点 | **1.9%** |
| 2500 等距网格 | **1e-8** |

**做法**：能量项传固定的等距网格（并 `requires_grad_(True)`）；残差项可以继续随机采样。

### 4.2 正交投影只能作用于 ψ，不能作用于 ψ′

约束 `⟨ψ|φ_k⟩ = 0` 限制的是 ψ 的**取值**，其切空间基是 `φ_k′` 不是 `φ_k`。
用 `φ_k` 去投影 `ψ′` 会减掉本不属于该空间的分量：实测把中点处的 `|dψ/dx|`
从 1.062 削到 0.531（正好一半），Rayleigh 商从 1.5 变成 1.25。

**做法**：先求导、只对波函数本身投影 —— 见 `Losses.energy_projected`。
有回归测试锁住：`test_energy_projected_preserves_derivative`。

### 4.3 `_prepare` 不能无脑 detach

曾经在 `_prepare` 里对所有输入做 `x.detach().requires_grad_(True)`，
这会把已经参与上游计算的张量（非叶子）从计算图上切断，autograd 拿不到 `dψ/dx`，
能量算成 6.38（真值 1.5）。

**正确写法**：只在 `x.is_leaf` 时处理，非叶子原样返回。

### 4.4 跳过的测试等于没有测试

装上 torch 之前，所有 torch 相关测试都被 `importorskip` 跳过，计数显示 48/48 全绿，
但**训练管线一次都没执行过**。装上 torch 后立刻崩了 5 个测试。

**教训**：必须在一个装齐依赖的环境里跑一遍才算数。

### 4.5 投影矩阵的网格必须与传入张量的网格一致

正交约束的 `project` 是按固定 `energy_grid` 的行索引构造的矩阵，
但旧代码把**随机配点**的残差喂给了它——形状同为 `(2500,1)`，
语义却完全不对，投影静默失效。修复：正交分支的残差项也使用 `energy_grid`。
详见工程记录第 14 条。

---

## 5. 代码里容易踩的地方

| 位置 | 注意点 |
|---|---|
| `Losses.energy` | 必须传确定性等距网格，且开启 `requires_grad` |
| `Losses.energy_projected` | 只投影 ψ，不投影 ψ′ |
| `Losses._prepare` | 只对叶子做处理，非叶子原样返回 |
| `Trainer.fit` | `step_fn` 返回 `{"loss": ...}` 或 `{"terms": {...}}`；后者必须用 `TrainConfig.term_weights` 传权重，否则权重静默失效 |
| `spectrum.get_reference` | 能量是 **E** 不是波数 k；k ↔ E 换算是 E = k²/2 |
| `spectrum.to_hhl_normalization` | 论文用 max\|ψ\|=1，本项目物理口径是 ∫\|ψ\|²=1 |
| `exp1.harmonic()` | 峰值必须取整条曲线的最大值，不能取"传入这批点的最大值" |
| `exp2 --orthogonalize` | 激发态几乎必需；**训练残差、能量、归一化、评估**都必须与投影基用同一张 `energy_grid` |

---

## 6. 下一步做什么（按优先级）

### P0 · 把结论做成可发表的复现报告（基本完成）

当前结论是**观察**，还不是**证据链**。需要补：

- [x] 种子重复（3 个 seed）与均值 ± 标准差：`run_seed_sweep.py` + `aggregate_seeds.py`
- [x] 经典数值基线并与网络指标同口径：`classical_baseline.py`
- [x] 超参敏感性：`exp2_hparams`（lr ×2、energy 权重 ×2、epochs）
- [x] 把所有结果并进 `docs/results.md` / README
- [x] **Rayleigh 商残差**替代可学习 E（修好“无泄漏”这句话，有限深阱宽度 3–4×）
- [x] **可学习 deflation** 求第一激发态（8/9），以及链式 deflation 求 n=2
- [x] **证书搬运**：把滤波后的夹角界搬回网络原始输出（初等引理 + 单测）
- [x] **二维协议**：惯性计数 + 9 run 扫描 + 简并拒绝 + 离散化 gap
- [x] 图（`paper/make_figures.py`：Pillow 热图 + pgfplots 矢量图，绕开 matplotlib ABI）
- [x] 论文稿（`paper/main.pdf`，45 页，`latexmk -pdf main.tex` 零错误零 overfull）
- [x] 简并**子空间**证书 + 实验八/九（分辨率边界）+ `cover_letter.md`
- [ ] git 仓库与代码发布（当前目录不是 git 仓库；`LICENSE`（MIT）已就位，版权行为占位 “The authors”）

> 2026-10-27 的扫描同时暴露了实验二损失未加权的 bug，旧单次 exp2 结果作废；
> 修复后的多种子结果才是可引用数字。

> **论文状态（2026-10-28 凌晨）**：`paper/main.tex` 是 CPC **正刊
> （regular article）** 稿：摘要、program summary 框、8 节正文（含 §4 子空间
> 证书与 §7.3 “为什么不直接解 H_h”）、附录 A（算法 1 + 三次可复现失败）、
> 24 refs，共 **45 页**。所有表由 `paper/make_tables.py` 从 `runs/*.json`
> 生成，所有图由 `paper/make_figures.py` 生成，正文数字与 JSON 同源。构建：
> `cd paper && latexmk -pdf main.tex`（45 页、0 error、0 overfull、0
> undefined）。投稿材料另有 `paper/cover_letter.md`（草稿）、
> `paper/README.md`（生成/复现说明）、`examples/certify_demo.py`（无 torch
> 的最小 API 示例）与 `paper/check_refs.py`（标签/引用静态检查）。计时数字（计数、证书、引导扫描、随网格的伸缩）由
> `scripts/bench_count.py` 实测写入 `runs/_bench/count_timing.json`，正文引用该文件
> （2026-10-28 补测**完整证书**耗时：1D 76 ms、2D 193 ms，均已写入该 JSON）。
> **一维网络的真实规模是 64×3**：`run_seed_sweep.py` 经 `_seed_worker.py` 调用，
> worker 的 `--width/--depth` 默认是 64/3，只有直接跑 `exp5_certificate_training.py`
> 才用 128×4；几份文档与论文初稿曾按 128×4 写，已全部改正（逐 run 的 JSON 为证）。
> 二维参考不再用 ARPACK 值（那正是提议区间的同一批数，属循环论证）：
> `scripts/dense_reference_2d.py` 对 6241×6241 全矩阵做稠密 LAPACK `eigh`
> 并取 Rayleigh 商，9/9 包含、与 ARPACK 差 ≤3.1e-13；据此全稿的包含性
> 计数为 **39/39**（主扫描 18 + 泄漏审计 21，其中 9 个两处重合）。附录 A
> 的三次失败里，傅里叶特征与软边界/未归一化两项均有 `scripts/appendix_failures.py`
> 可复现（前者的数据在 `runs/appendix_fourier2d/`）。§6.7 的
> 转折点已用加细的 λ 扫描收紧到 **split/floor = 0.95（拒绝）→ 1.01（接受）**，
> 并把与 Cancès et al. (Math. Comp. 2020, 簇本征值界) 的关系写进 §4 与 §7.4。
> **2026-10-28 夜间订正**（逐条对 JSON 核过）：一维网络 64×3（不是 128×4）；
> 1D/2D 训练墙钟 23–39 s / 93–103 s；线程数 4/2（不是单线程）；SciPy 1.17；
> $\varepsilon\|H_h\|$ 为 7e-12–2e-9（原写 1e-9）；滤波 L2 改变 4e-6–1.6e-3 限定为
> “主扫描 18 个 run”（泄漏审计最大 7e-3）；§7.3 补实测：同一矩阵 shift-invert
> 求解 1.7 ms/19.7 ms，证书 74 ms/193 ms（证书慢约 10 倍，原文“差不多快”不准确）。
> 附录 A 三条失败全部有落盘 run（`runs/appendix_failures/`、`runs/appendix_fourier2d/`），
> 其中“塌缩到 ψ≡0”改写为“未归一化损失下尺度不被固定，场停在 ‖ψ‖²=6e-5”。
> 另有：引言两句“我们反复观察到”原无落盘依据（其中“残差比误差小四个数量级”
> 在该谱隙下根本不可能），已换成可核实例——`infinite_well_0/seed1` 原始场
> |η−λ_h|=2.6e-7、残差 0.53、角界 0.56 rad、区间宽 0.297（真误差的 1.1e6 倍）；
> 未归一化损失残差 2.9e-4 而到单位本征函数的 L2 距离 0.99（3414 倍）。
> 后者由新脚本 `scripts/residual_vs_error.py` 从落盘场重算，不重训。见工程记录 30。
> 另修一处**出处泄漏**：二维 `summary.json` 的 `discrete_ref` /
> `filtered_contains_ref` 其实是提出区间的那个 ARPACK 值，散点图与 `contains λ_h`
> 行一直沿用它。现改为稠密参考（`runs/reference2d/dense_ref.json`），逐 run 结果
> 落盘 `runs/reference2d/filtered_containment.json`（9/9）；二维宽度/误差比
> harmonic 25–26、anharmonic 31–36、box 694–794，§5 紧度段与结论按此重写
> （box 与无穷深阱的分母都在舍入底噪上，唯一真正宽的是有限深阱）。两张表头把
> 角界误写成 `sinΘ`，改回 `Θ`。见工程记录 31。
> 另外热图（图 2）的 anharmonic "参考"原来是用**零势**做 eigsh 得到的，即方阱
> 基态（λ=9.868）去比 anharmonic 网络（λ=1.0977）；现三幅参考都取各自 run 的势，
> ARPACK 值与稠密值差 ≤3.1e-13，题注相应改为"仅显示、不进入误差估计"。见工程记录 32。
> 最后一遍数字核对另修：二维引导扫描实为 62/30 候选（6 s/12 s），基准里的 16 s 是
> 81 候选满扫描；201×201 计数 0.85 s；§3.2 的"塌缩到零"改为"尺度不被确定"；
> 摘要/结论的"39/39 磁盘上全部"改为"我们报告的 39 个"并注明多态链 27 个的
> 性质（含有其认证序号所指的本征值）。见工程记录 33。
> **独立复核（第二双眼睛）**又修 10 处：滤波降残差的最小比值（1.2 倍，不是"一个数量级"）、
> §7.1 把"四到六次计数"限定到二维（一维证书约 180 次计数）、结论里 box 的比值归到
> 7–8e2、训练时间改为 manifest 墙钟 30–45 s / 99–109 s、§7.3 一维快 43 倍、
> leakage 的"四个变体"改为三个能量变体、子空间独立对界 5.6–6.3e-3、证书 0.074 s、
> 投影残差 ≤2e-16；并新增 `scripts/lapack_reference_widths.py`，把 §6.9 引用的
> 经典求解器证书宽度落到 `runs/reference1d/lapack_certificates.json`。
> 论文现 54 页。见工程记录 34。
> **作者块已填（2026-10-28）**：Chunyang Chen、Yuhao Peng，共同第一作者，单位
> School of Mathematics and Physics, China University of Geosciences (Wuhan)，
> 彭宇豪为通讯作者（pngyuo@cug.edu.cn），`\fntext` 注明同等贡献。
> **仓库已公开（2026-10-29）**：`https://github.com/YuhaoPeng1103/PINN_equation`
> （public，main 分支，初次提交 799 个文件 / 22 MB，`runs/` 证据与 `paper/main.pdf`
> 都在其中）。program summary 的 `Developer's repository link` 与 Code availability
> 已填该 URL 并重新编译。CPC Library 链接由 CPC 分配，保留 `(to be assigned)`。
> **投稿前仍需**：(1) 参考文献 18 条 DOI 已于 2026-10-28 全部对 Crossref 解析通过
> （其中 `mishra2023` 原记 drab096、`rump1999` 原记 …7186-3_5，均已改正；
> `tancik2020` 无 DOI，按 NeurIPS 33, 7462–7473 著录）；(4) 若改投 MLST，需补连续算子层面
> 的 $O(h^2)$ 显式界与系统基线对比，当前稿件明确声明不做。

> **运行状态（2026-10-28）**：磁盘结果：`exp1` 3/3；`exp2` 20/21
> （`rayleigh_random` / `rayleigh_random_gradbalance` 未跑，二者都是
> E=参考值的泄漏对照，不是主结果）；`exp3` 9/9；`states` 12/12；
> `exp2_hparams` 15/15；`certs` 6/6；`certs5` 18/18；`certs5_refenergy`
> 9/9；`certs5_rayleigh` 9/9；`certs5_deflated` 9/9；`certs5_deflated2` 9/9。
>
> **A 线主协议已改为 Rayleigh 商残差**（`--rayleigh-energy`）：
> 12 个无泄漏基态 run（9 个 `certs5_rayleigh` + 3 个 `certs/learned0`）
> 包含 12/12，中位宽度 5.0e-4，中位宽度/误差 38（谐振子 5.5–19、
> 无限深阱数值下限、有限深阱 946–2920）；原始可认证 7/12，滤波后 12/12。
> **仍按 §5.3 是窄义 Go、广义 No-go（紧度不够）。**
>
> **多态（C 线）**：`certs5_deflated`（n=1）用同 seed 学出的基态做硬正交，
> 8/9 落到 n=1 且证书确认（此前 0/6）；`certs5_deflated2`（n=2，链式
> deflation）8/9 落到 n=2。两次失败都是 `harmonic` 的 seed2：它在 n=1
> 跳到 ψ₂，错误沿链传播到 n=2。27 个多态 run 中 25 个包含目标态参考值。
> 汇总脚本：`scripts/multistate_report.py` → `runs/multistate/summary.md`。
>
> 详见 [certification.md](certification.md) §5.6.1、§5.7–5.9 与
> `runs/certificates/go_nogo.md`、`runs/leakage_audit/summary.md`。
>
> 可学习 E 在深阱上不收敛的问题（E 停在 −9.8，真值 −48.97）已由
> Rayleigh 商残差修复（工程记录 21→22）。

### P0.5 · 2D 原型（想投 MLST/CPC 正刊的硬门槛）——**2026-10-28 完成**

> **结论**：计划里“`splu` 不给惯性”的判断是错的。`splu` 确实会做行列
> 交换，但 `permc_spec="NATURAL"` + `diag_pivot_thresh=0` + `SymmetricMode`
> 下 `perm_r/perm_c` 都是恒等（必须显式断言），`U` 对角元符号即惯性。
> 实现见 `certify2d.count_eigenvalues_below_2d`，稠密/惯性/ARPACK 三方
> 比对单测在 `tests/test_certify2d.py`。新增一道**分辨率下限**
> `1e-9·scale`——否则扰动重试会给简并态发假证书（工程记录 27）。
>
> 产物：`filter_certificates2d.py`、`run_seed_sweep_2d.py`、
> `exp7_degeneracy_2d.py`、`discretization_study_2d.py`、
> `runs/certificates_filtered2d/`、`runs/degeneracy2d/`、
> `runs/discretization2d/`。9 个 run 约 93–103 s/run（6000 epoch，81×81，
> jobs=2）。文档见 `docs/certification.md` §5.10。

<details><summary>原始计划（保留）</summary>


一维 + 经典定理的协议，审稿人会直接问“为什么不直接用谱方法”。要把
“协议”升级成“结果”，最小增量是 **2D**：

1. `H_h = −½(Δ_x + Δ_y)h + V`（五点差分），N=81×81 → 内部 79²=6241 个
   未知量；稠密 LDLᵀ 取惯性的成本 O(n³)≈2e11 太贵，改用 **稀疏 LDL**
   （`scipy.sparse.linalg.splu` 不给惯性；可行路线是 shift-invert Lanczos
   取邻近谱 + 区间计数用稀疏 `ldl` 的 `scipy.linalg.ldl` 仅对小规模，
   或用 `sksparse.cholmod` 的 inertia）。
2. 证书：Temple 的隔离条件需要 γ = 到其余谱的距离下界；2D 里用
   shift-invert 的 Ritz 值 + 残差做**单侧**界，再配区间计数。
3. 势：2D 谐振子（可分，有解析谱）→ 2D 有限深方阱（不可分）→
   2D 双阱（近简并、能展示“证书拒绝认证”）。
4. 训练：把 exp5 网格目标搬到 2D（2001 → 81×81），其余不变。

成本估计：网格 + 计数 1–2 天；训练与滤波 1 天；文档 0.5 天。**没有这一
步，MLST/CPC 正刊基本不可能**；有它 + 连续算子界 + 系统基线，才有资格
竞争。

</details>

### P0.6 · 定向舍入的计数 + 连续算子的界——**2026-10-28 完成**

> **结论**：§5.3（原“局限”）里剩下的两条都被补上了，代价很小。
> 此外二维可分势的 λ_h 也变成了**严格区间**（`certified_separable_brackets_2d`
> 在二维计数上二分，宽度 1e-11–2.5e-9），Table 2 里只有 anharmonic 的 λ_h
> 仍是稀疏分解值——论文的表格脚注已按此写明。
>
> **(a) 定向舍入计数**（`src/nnschrodinger/solver/certified_count.py`）。
> 区间算术 = IEEE 正确舍入 + 每步 `nextafter` 外推一个 ulp，二十行。
> 关键发现是**行列式递推在区间里不能用**：相对宽度在第 35–45 步就超过 1
> （363 个扫描 shift 上无一例外），因为 d_k 在谱的体量尺度上剧烈抵消；
> 换成 **pivot 比值递推** q_k=(a_k−σ)−b²/q_{k−1} 后，区间一直很窄。
> 一维：N=2001、3 个势、121 shift，**361/363 认证**，两个拒绝都落在扫描最后
> 一个 shift 上（该 shift 在双精度里恰好等于本征值，正是必须拒绝的情形），
> 拒绝报告出问题的 pivot 下标 999。代价 5 ms vs 浮点 0.4 ms（13 倍）。
> 二维：带状的区间 LDLᵀ **不可行**（361 未知量下 10 个 shift 拒绝 6 个，首次
> 拒绝在第 174 步时相对枢轴宽度已达 0.93，且随 N 恶化）；可分势改用
> Kronecker 和 H_h = A_x⊗I + I⊗A_y，计数退化成两两求和比较，
> 81×81 上 25 个 shift 全部认证、零分歧、18 ms/次；不可分势**明确拒绝**并给出理由。
> 拒绝永远不会被四舍五入成整数——这是整个实现的意义。
>
> **(b) 连续算子的单侧界**（`src/nnschrodinger/solver/continuous_bound.py`）。
> 由 min–max，λ₁(H) ≤ R(u) 对任何 u∈H¹₀ 成立；P1/Q1 插值后的采样函数是
> 符合的试探函数，所以**不需要参考解、不需要网格算子、不需要插值常数**。
> 势能项用 Gauss 求积，低次多项式时精确（谐振子/box/anharmonic/把跳点作为单元端点的
> 有限深阱）；代码同时用两种阶的 Gauss 规则交叉检查，**两个严格区间不相交**才拒绝
> （固定 1e-11 相对阈值在二维会误判，因为 padding 随项数增长）。
> 结果（`runs/continuous_bound/bound.json`，论文表 `tab_continuous`）：
> 网格加密时 R(ψ_h)−λ_h 的收缩比 6.23–6.26（二维，预测 6.25）和 56–64（一维，
> 预测 64）；有参考值的五例里，粗网格 **5/5** 落在 [λ_h,R] 内，细网格 **4/5**。
>
> **(c) 附带的反例**：有限深阱在 N=4001 时跳点正好落在节点上，节点采样把 −50
> 换成 0，λ_h 被抬高 3.67e-3 到连续值**之上**；N=1000–1011 上离散能量
> 六次在上、六次在下，误差高达 1.4e-2 且不单调衰减
> （`runs/continuous_bound/alignment.json`，`scripts/continuous_bound.py --alignment`）。
> **归因实验**：同一张网格只改那一个落在跳点上的节点的取值（0 → −50），
> λ_h 从 +3.67e-3 变到 −3.73e-3，跨过真值 7.4e-3——整个误差就是这一个格点，
> 两种约定没有哪个更准。这十二个符号画成论文 Figure 2
> （`paper/figures/data/finite_well_alignment.csv`，`python paper/make_figures.py` 生成）。
> 所以“离散证书”≠“连续证书”，当格式不是变分相容时
> 尤其如此。box 则可以用 sinθ≤θ 得到严格的下界
> λ_j^{(h)} = (2/h²)sin²(jπh/2) ≤ j²π²/2，上端由 R(ψ_h) 给出，所以从
> **bracket 下端点**到 R 两端都严格（表里是中点，比下端点高 ≤2.5e-9，
> 而边距 8e-4，小五个数量级）。
>
> 测试：`tests/test_certified_count.py`（11 项）、`tests/test_continuous_bound.py`
> （9 项），含“在等于本征值的 shift 上必须拒绝”“不可分势必须拒绝”
> “未告知的 kink 必须被求积检查抓到”“非方网格上 Q1 界等于两个一维界之和”。
> 论文见 `paper/sections/05_certified.tex`（第 6 节）。

### P1 · 实验一若要真正对标 HHL，得换网络

现在的网络是普通 MLP。HHL 用的是角度嵌入 + 纠缠层的量子线路式结构。
要断言"HHL 的插值主张不成立"，必须按其原始结构重建，否则只能说自己这套配置不行。

### P2 · 激发态扩展到 n=2, 3（扫描中）

`states` 组已加入 n=0/1/2/3（n≥1 用正交约束），种子 0/1/2、5000 epochs、
width=128、depth=4。正交残差也改用与投影基一致的固定网格（工程记录 14）。
结果汇总后应给出"态序号 vs 精度"的均值 ± 标准差。

### P3 · 谐振子的边界处理（旧解释作废，待重测）

旧记录说“谐振子 ±6 截断 + Dirichlet 导致 8.3e-3”，但实验二的 `boundary`
权重是 0，根本没有加 Dirichlet；8.3e-3 来自已作废的随机采样单次运行。
修好权重后，谐振子网格配置在 2000 epochs 已达到 1.9e-3 相对误差。
是否还需要把域拓到 ±8，等新扫描的 `rayleigh_learned` 结果再判断。

### 不建议做的

- 多维/氢原子三维扩展：一维还没做透
- PennyLane 量子线路：成本高、不影响现有结论
- 继续优化数值性能：已够用（参考解生成从 3.2s/批 优化到 3.6ms/批）

---

## 7. 诚实声明：创新点在哪

### 严格意义上的学术创新：**基本没有**（A 线组合协议除外）

| 用的方法 | 来源 |
|---|---|
| PINN / 残差损失 | Raissi et al. 2019 |
| Rayleigh 商 / 变分原理 | 教科书（Rayleigh–Ritz） |
| 序列变分 / deflation | 教科书（Hylleraas–Undheim） |
| `(x, E)` 双输入 | HHL 论文 |
| Numerov + 打靶 | 教科书 |
| Jacobi 特征分解 | 教科书（Golub & Van Loan §8.5） |
| 傅里叶特征 | Tancik et al. 2020 |
| 数值基线 | 教科书 |
| Sturm 计数 / Sylvester 惯性 | 教科书（Golub & Van Loan） |
| Kato–Temple / Weinstein / Davis–Kahan | 数值分析教科书 |
| DST 截断 | 谱方法教科书 |

**贡献（2026-10-28 定稿，按论文顺序）**——没有一条是发明新定理，全部是
“把现成工具用对地方 + 弄清它在什么条件下失效”：

1. **协议**：把后验界（Temple/Weinstein/Davis–Kahan）的**假设**变成可计算的
   验证（Sturm/惯性计数），从而能用在谱未知的网络输出上；并让证书宽度反过来
   选滤波截断阶数（`guided_filter`）。
2. **子空间扩展**：简并态单态证书必然拒绝，簇证书（验证重数 + 主子角界）
   把“拒绝”换成一个可认证的对象；实验九把两者的边界量化到计数分辨率。
3. **定向舍入计数**：一维 pivot 比值递推（行列式递推在区间里第 35–45 步失效，
   这是实测的性质不是实现问题）；二维可分势用 Kronecker 和、不可分势明确拒绝。
4. **连续算子的单侧界**：min–max + P1/Q1 插值，不需要参考解也不需要插值常数；
   并给出有限深阱的反例说明下界为什么不对称（同一个格点翻符号 7.4e-3）。
5. **审计**：泄漏审计、状态指标、包含率/紧度统计、分辨率边界——这部分是
   “评测协议 / 失败模式检测”，不是“更高精度的求解器”；宽度对非光滑势仍
   保守 3 个数量级，论文里明说了。

**不声称的**：没有更快的求解器（比 shift-invert Lanczos 慢 10–40 倍）；
没有对高维矩阵-free 情形的验证；没有超越经典后验界理论的新估计式。

### 真正的产出：可复现的**纠正性结论**

1. 残差损失在多本征态共存时不唯一（**旧量化 1.05e-1 → 4.16e-3 已作废**；
   新结论以 `residual_learned` vs `rayleigh_learned` 的 3 种子结果为准）
2. Rayleigh 商的随机估计带 1.9% 噪声（估计量性质，仍成立；量化：1.9% → 1e-8）
3. `(x,E)` 插值不成立且容量无关（量化：训练 1.4e-2 vs 插值 0.39~1.57；宽度 64→256 无改善）
4. 正交投影误作用于导数会让能量偏差 20%（量化：1.25 vs 1.5）

### 定位建议（更新）

- **论文已成形**：`paper/main.pdf`，54 页，零 overfull、零未定义引用，
  143 项测试全绿。目标期刊：CPC 正刊（定位为“评测协议 + 软件”，
  符合 CPC 的 software-and-validation 体裁）。
- **仓库**：<https://github.com/YuhaoPeng1103/PINN_equation>，已 `git init` 并推送
  （MIT，LICENSE 持有人 Chunyang Chen and Yuhao Peng）。作者块已填好
  （Chen & Peng，共同第一作者，Peng 通讯，pngyuo@cug.edu.cn）。
- **投稿前仍需人决定的**：cover letter 的签名与投稿系统里填写的推荐审稿人
  （作者已表示不推荐，信里写的是 "We do not suggest referees"）；
  是否给仓库打 Zenodo DOI 冻结版本（推荐，但非必需）。
- **如果还要加强**：多跑种子把包含率的 Wilson 区间收窄（见 §6 的统计功效段，
  需要约 70 次无失败运行才能声称 ≥0.95）。

---

## 8. 交给下一个人的话

1. **先跑 `python tests/run_tests.py`**。143 项全绿说明环境没问题，再动代码。
2. **读 `docs/engineering-log.md` 第 4 节和第 11 节**。那两条最容易重新踩进去。
3. **改 `Losses` 之前先看回归测试**。那里钉死了几个数值上的坑。
4. **实验结果是单次运行的**。如果你要引用这些数字，先补种子重复。
5. **不要相信"Skipped"**。torch 没装时测试会跳过，看起来全绿但管线没跑过。

---

## 附：常用命令速查

```bash
# 环境自检
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

# 参考解与测试
python -m nnschrodinger.physics.verify
python tests/run_tests.py
python tests/minipytest.py -k losses        # 只跑某类

# 实验
python scripts/exp1_regularized_network.py --epochs 3000
python scripts/exp2_unregularized_pinn.py --epochs 5000 --state 0
python scripts/exp2_unregularized_pinn.py --epochs 4000 --state 1 --orthogonalize
python scripts/exp3_data_assimilation.py --n-obs 0 16 64

# 结构
python scripts/tree.py
```
