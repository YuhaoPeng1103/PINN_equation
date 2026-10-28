# 项目结构与模块地图

> 生成方式：手写维护（`scripts/tree.py` 可打印实时结构）。

```
基于神经网络的薛定谔方程求值问题/
├── README.md                      # 入口：这是什么、怎么跑、结论在哪
├── pyproject.toml                 # 打包/依赖（torch 为可选 extra）
├── requirements.txt
├── Makefile                       # make verify / test / lint
├── LICENSE                        # MIT（版权行为占位 “The authors”）
├── .gitignore
│
├── docs/
│   ├── project-overview.md        # ★ 项目脉络：研究问题 → 代码 → 证据
│   ├── structure.md               # ★ 本文件：目录与模块职责
│   ├── literature.md              # 文献脉络与研究现状
│   ├── experiments.md             # 三个实验的设计与预期
│   ├── certification.md           # ★ 后验证书：数学、Sturm 验证、实验协议
│   ├── results.md                 # ★ 实测结果、多种子统计与解读
│   └── engineering-log.md         # 工程问题记录（踩坑与修复）
│
├── src/nnschrodinger/
│   ├── __init__.py
│   │
│   ├── physics/                   # 【纯 numpy，无 torch 依赖】
│   │   ├── spectrum.py            # 参考解：谐振子/方势阱/氢原子（解析式 + Numerov）
│   │   ├── numerov.py             # Numerov 积分 + 打靶求本征值（核心数值工具）
│   │   ├── eigen.py               # 稠密/三对角 Jacobi 特征分解（无 scipy 的替代品）
│   │   ├── potentials.py          # 势函数库（含跳变点、深阱标记）
│   │   ├── operators.py           # 梯度/拉普拉斯/Ĥ 的 numpy 实现（测试用）
│   │   └── verify.py              # ★ 交叉校验：解析 ⟷ Numerov，10/10 通过
│   │
│   ├── data/                      # 【纯 numpy】
│   │   └── targets.py             # HHL 论文的 10 个目标函数 + 相似度度量
│   │
│   └── solver/                    # 【部分依赖 torch】
│       ├── backends.py            # torch 可用性探测（无 torch 时给出清晰提示）
│       ├── inputs.py              # 采样：配点 / (x,E) 成对 / 按跳变点分段
│       ├── networks.py            # MLP(+傅里叶特征) / HHL 网络 / 谱方法基线
│       ├── losses.py              # 物理项 / 数据项 / 正则项，可独立测试
│       ├── train.py               # 训练器：分项权重 + 早停/best checkpoint + 非有限检测
│       ├── evaluate.py            # ★ 指标：能量误差 / overlap / inf-norm / 残差
│       ├── certify.py             # ★ 后验证书：Weinstein/Temple/夹角界 + Sturm 计数
│       ├── filtering.py           # ★ 证书引导的 DST 谱滤波 + 三角函数把界搬回原始 ψ
│       ├── certify2d.py           # ★ 二维证书：稀疏 LU 的 Sylvester 惯性计数 + 分辨率下限
│       ├── certify_subspace.py    # ★ 簇/子空间证书：验证重数 + 主子角界（sinΘ ≤ ρ_F/δ）
│       ├── certified_count.py     # ★ 定向舍入计数：一维 pivot 比值、二维 Kronecker 和、认证 bracket
│       ├── continuous_bound.py    # ★ 连续算子单侧界：P1/Q1 插值的 Rayleigh 商 + 求积交叉检查
│       └── filtering2d.py         # ★ 二维 DST 滤波 + 二维搬运界
│
├── scripts/
│   ├── exp1_regularized_network.py   # 实验一：有监督 (x,E) 回归 + 能量插值
│   ├── exp2_unregularized_pinn.py    # 实验二：无监督 PINN（含残差/能量/正交对照）
│   ├── exp3_data_assimilation.py     # 实验三：物理 + 少量观测
│   ├── exp5_certificate_training.py  # 实验五：证书同口径训练（离散残差 + 边界）
│   ├── certify_saved.py              # 对保存的 wavefunction.npz 批量算证书
│   ├── filter_certificates.py        # 证书引导滤波 + 重新认证 + 把界搬回原始 ψ
│   ├── go_nogo.py                    # A 线 go/no-go 判定（无泄漏基态 run）
│   ├── leakage_audit.py              # 泄漏对照汇总（rayleigh/learned/reference/deflated）
│   ├── multistate_report.py          # 多态（基态 + 激发态）证书表
│   ├── exp6_certificate_training_2d.py  # 实验六：二维证书同口径训练（硬 Dirichlet 掩码）
│   ├── filter_certificates2d.py      # 二维滤波 + 认证 + 与 ARPACK 参考比对
│   ├── run_seed_sweep_2d.py          # 二维多种子扫描（子进程，建议 --jobs 2）
│   ├── exp7_degeneracy_2d.py         # 二维简并态的证书拒绝实验
│   ├── exp8_subspace_certificate.py  # 二维简并子空间的簇证书（含独立 run 配对判定）
│   ├── exp9_resolution_boundary.py   # 单态证书的分辨率边界（受控劈裂 λ(x²−y²)）
│   ├── discretization_study_2d.py    # 二维 λ_h(h) 收敛与外推 λ*
│   ├── bench_count.py                # 计数/证书/引导扫描的耗时基准（论文中计时数字的来源）
│   ├── dense_reference_2d.py         # 二维独立参考：稠密 LAPACK eigh + Rayleigh 商（非 ARPACK）
│   ├── appendix_failures.py          # 复现附录 A 的两种训练失败（软边界惩罚 / 未归一化残差）
│   ├── residual_vs_error.py          # 从落盘场重算“残差 vs 真实 L2 距离”（附录 A / 引言引用）
│   ├── lapack_reference_widths.py    # 经典 LAPACK 本征向量的证书宽度（§6.9 对照，runs/reference1d/）
│   ├── classical_baseline.py         # 经典 Numerov 基线（与网络指标同口径）
│   ├── run_seed_sweep.py             # 多种子并行扫描（子进程 + 线程限制）
│   ├── _seed_worker.py               # 单次 (实验, 配置, 种子) 的 worker
│   ├── aggregate_seeds.py            # 汇总均值 ± 标准差 → summary.{json,md}
│   └── tree.py                       # 打印项目结构
│
├── tests/
│   ├── run_tests.py               # 统一入口（有 pytest 用 pytest）
│   ├── minipytest.py              # 内置极简运行器（受限环境无 pytest 时用）
│   ├── conftest.py                # sys.path 与 requires_torch 标记
│   ├── test_physics_spectrum.py   # 解析解：能量、归一化、正交性、边界
│   ├── test_numerov.py            # Numerov：与解析谱逐项比对
│   ├── test_eigen.py              # Jacobi 与 numpy.linalg.eigh 逐元素比对
│   ├── test_verify.py             # 交叉校验器本身
│   ├── test_potentials.py         # 势函数库
│   ├── test_data_targets.py       # 目标函数与相似度
│   ├── test_inputs.py             # 采样器
│   ├── test_evaluate.py           # ★ 评测器：杂波必须被识别出来
│   ├── test_networks.py           # 网络形状/可二阶导/谱方法基线
│   ├── test_losses.py             # 精确解上每项损失为 0
│   ├── test_train.py              # 训练循环冒烟测试
│   ├── test_certify.py            # ★ 后验证书：Sturm 计数、Temple 包含性、拒绝非单态区间
│   ├── test_filtering.py          # ★ DST 正交性、截断正确性、带噪状态的恢复
│   ├── test_certify2d.py          # ★ 二维惯性计数 vs 稠密谱、简并拒绝、二维 DST/搬运界
│   ├── test_certify_subspace.py   # ★ 簇证书：重数、基无关性、秩亏/漏态/塌缩拒绝
│   ├── test_certified_count.py    # ★ 定向舍入计数：对稠密谱、单调性、简并处与不可分势必须拒绝
│   ├── test_continuous_bound.py   # ★ 连续界：超过真值、O(h²) 收缩、kink 求积检查、非方网格可加性
│   └── test_seed_aggregation.py   # 多种子汇总：均值/标准差/目录发现
│
├── paper/                         # CPC 投稿稿（main.tex + sections/ + 生成表图）
├── examples/
│   └── certify_demo.py            # 最小可运行示例（无 torch）：1D 认证 + 2D 拒绝/簇认证
│
└── runs/                          # 运行证据
    ├── environment.txt            # 环境快照（含 torch 是否可用）
    ├── physics_verification.txt   # 参考解交叉校验表
    ├── tests_output_torch.txt     # 完整测试输出（含 torch）
    ├── classical_baseline.json    # Numerov 基线指标
    └── seeds/                     # 多种子原始结果 + manifest + summary.{json,md}
```

## 数据流

```
  problems  ──►  potentials.py (V(x))
                    │
                    ├──► spectrum.py / numerov.py ──► 参考解 (E_ref, ψ_ref)
                    │                                      │
                    │        verify.py 交叉校验 ◄──────────┘
                    │
                    ├──► inputs.py ──► 采样 (配点 / (x,E) 对)
                    │
                    └──► networks.py ──► 网络 ψ_θ(x)
                              │
                              ├──► losses.py ──► train.py ──► 训练好的 ψ_θ
                              │
                              └──► evaluate.py ◄── 参考解
                                        │
                                        ▼
                               runs/*.json （指标报告）
```

## 关键接口速查

| 想做什么 | 调用 |
|---|---|
| 取某个问题的精确解 | `physics.get_reference("harmonic", 2)` |
| 数值求解任意一维势 | `physics.numerov.numerov_spectrum(V, lo, hi, window, k)` |
| 验证参考解是否可信 | `python -m nnschrodinger.physics.verify` |
| 评估一个波函数 | `solver.evaluate.evaluate_model(psi, x, ref.psi, E_ref, V)` |
| 给一个波函数打后验证书 | `solver.certify.certify_eigenpair(psi, x, V, isolate=True)` |
| 数离散谱中有几个本征值小于 σ | `solver.certify.count_eigenvalues_below(x, V, sigma)` |
| 建一个 PINN | `solver.networks.build_mlp(fourier=True)` + `solver.losses.Losses` |
| 训练 | `solver.train.Trainer(net, step_fn, TrainConfig(...)).fit()` |
| 二维证书 | `solver.certify2d.certify_eigenpair_2d(psi, grid, isolate=True)` |
| 二维惯性计数 | `solver.certify2d.count_eigenvalues_below_2d(grid, sigma)` |
| 二维滤波 | `solver.filtering2d.guided_filter_2d(psi, grid)` |
| 二维簇/子空间证书 | `solver.certify_subspace.certify_subspace_2d(psis, grid)` |
| 一维簇/子空间证书 | `solver.certify_subspace.certify_subspace_1d(psis, x, V)` |
| 定向舍入的 Sturm 计数 | `solver.certified_count.certified_sturm_count_1d(x, V, sigma)` |
| 定向舍入的可分二维计数 | `solver.certified_count.certified_separable_count_2d(grid, sigma)` |
| 二维本征值的严格区间（可分离势） | `solver.certified_count.certified_separable_brackets_2d(grid, k, hint=...)` |
| 认证的本征值 bracket | `solver.certified_count.certified_eigenvalue_brackets_1d(x, V, k=0)` |
| 连续算子的上界（一维/二维） | `solver.continuous_bound.p1_rayleigh_bound(x, V, psi)` / `q1_rayleigh_bound_2d(x, y, V, psi)` |
| box 的严格下界 | `solver.continuous_bound.box_lower_bound_1d(h, j)` |
