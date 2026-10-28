# 文献脉络：神经网络求解薛定谔方程

> 说明：以下文献线索来自本项目的设计依据与 2025–2026 年的检索。
> **引用信息以各自发表版本为准**；标注“检索线索”的条目在正式写作前需核对原文，
> 本项目不伪造作者、页码或 DOI。

## 1. 三条主线

### 主线 A：把方程“写进损失函数”（PINN）

| 工作 | 核心思想 | 对项目的启发 |
|---|---|---|
| Raissi, Perdikaris & Karniadakis (2019), *Physics-Informed Neural Networks* | 用自动微分把 PDE 残差作为损失项 | 项目里 `losses.residual` 的来源 |
| Markham & Tabrizi (2020), *Deep Learning for the Schrödinger Equation* | 用能量泛函（Rayleigh 商）+ 边界项预测基态 | `losses.energy` 与边界处理 |
| 后续 PINN 稳定性研究（2021–2023） | 多损失项梯度失衡是主要失败原因 | `LossWeights` + `grad_balance_norm` |

**项目里的落点**：`solver/losses.py`、`scripts/exp2_unregularized_pinn.py`。

### 主线 B：把“量子力学结构”写进网络

| 工作 | 核心思想 | 对项目的启发 |
|---|---|---|
| HHL 论文（2021），*计算薛定谔方程* 一系量子线路 + 经典 DNN | `(x, E)` 双输入；正则化 / 非正则化两种网络；与量子线路对照 | **项目的主骨架**：实验一/二的设计、`(x,E)` 输入、max\|ψ\|=1 归一化 |
| 变分蒙特卡洛 + 神经网络波函数（NNQS） | 波函数本身就是网络，用采样估能量 | `energy` 项与“网络即波函数”的视角 |
| 特征向量延拓（Eigenvector Continuation） | 用少量精确解张成子空间，插值出新参数下的解 | 实验一“插值能量”的**经典对照**；说明网络的插值能力并非魔法 |

**项目里的落点**：`solver/inputs.py`（`(x,E)` 与 HHL 输入映射）、`solver/networks.py`（HHL 网络）、
`data/targets.py`（论文的 10 个目标函数）、`scripts/exp1_regularized_network.py`。

### 主线 C：和经典数值方法比什么

| 方法 | 优点 | 缺点 | 项目中的角色 |
|---|---|---|---|
| Numerov + 打靶 | O(N)、精度高、一维标准解法 | 只到一维、高维失效 | **参考解与基线**（`physics/numerov.py`） |
| 有限差分 + 特征分解 | 通用、可直接给全部本征对 | O(N³)（稠密）或需专门的三对角算法 | `physics/eigen.py`（Jacobi） |
| 谱方法（Galerkin） | 光滑问题上指数收敛 | 需选取合适基函数 | 序列中的理想对照 |

**项目里的落点**：`physics/eigen.py`、`solver/networks.py::SpectralSolver`、`docs/experiments.md`。

## 2. 2025–2026 年这个领域的实际困难（检索线索）

这一节是 2026-10 做的文献检索摘要，用来回答“现在做 PINN 求解薛定谔方程，
真正的痛点是什么”。**不是**完整的综述；正式引用前需核对原文。

| 困难 | 典型表现 | 与本项目的关系 |
|---|---|---|
| **谱偏置 / 高激发态** | 基态容易，n≥2 的能量和波形迅速变差；课程学习、傅里叶特征、多尺度网络被用来缓解 | 实验二只做到 n=0..3；`FourierFeatures` 是现有缓解手段 |
| **简并 / 正交性** | 逐个态 deflation 需要参考态或软正交惩罚；简并子空间容易塌缩 | 现有 `orthogonalize` 用**解析参考态**做投影，是已知泄漏；证书会暴露这一点 |
| **损失平衡 / 失败模式** | 残差、归一化、能量项量纲差异大；梯度失衡导致“能量好、波形是杂波” | `grad_balance_norm` + `Trainer.term_weights`；证书宽度是新的失败模式指标 |
| **高维扩展** | 3D/多体问题的配点数爆炸；量子电路混合模型（QCPINN 等）是 2026 年的热点 | 本项目只做 1D，明确声明边界 |
| **可复现性 / 评测口径** | 不同论文的“误差”定义不一致；缺少专门的 Schrödinger 本征值 benchmark | 本项目的 `evaluate.py` + `physics/verify.py` + 多种子扫描 |
| **没有后验误差条** | 绝大多数工作只报告与参考解的误差，无法在无真解时判断解是否可信 | **本项目的 A 线**：Weinstein/Temple/Davis–Kahan + Sturm 计数 |

检索中反复出现的具体线索（仅作定位用）：

* 高激发态/谱偏置：课程学习与多尺度 PINN 的 2025–2026 年工作；有工作明确指出
  经典 PINN 在 n=2、3 上明显吃力（检索线索：arXiv:2609.22189 附近的 PIQNN 系）。
* 简并/正交：Rayleigh–Ritz 子空间与软正交约束的 2024–2026 年工作
  （检索线索：NJP 2024 的软正交、Physica Scripta 2026 的 Rayleigh–Ritz 子空间）。
* 失败模式/评测：2025–2026 年出现 PINN failure atlas / survey 类工作
  （检索线索：TechRxiv 2025、Zenodo 2026），说明“报喜不报忧”已是社区问题。
* 后验误差界：现有 PINN 误差界主要集中在**正问题**（forward PDE），
  例如用残差构造能量范数界（检索线索：arXiv:2606.12050；Hillebrecht & Unger 2022）。
  **没有找到**针对本征值问题、且能验证单态区间的神经本征对证书。

## 3. 后验证书的经典数学（A 线的理论依据）

这些是经典结果，不是本项目发明的：

* **Weinstein 界**：`dist(η, spec(H)) ≤ ‖Hψ − ηψ‖`；
* **Temple 界（Kato–Temple）**：若 [a,b] 内恰有一个本征值 λ、a<η<b 且
  `‖r‖² < (b−η)(η−a)`，则
  `η − ‖r‖²/(b−η) ≤ λ ≤ η + ‖r‖²/(η−a)`
  （下界用上邻居距离、上界用下邻居距离；分母写反会给出无效的“界”）；
* **Davis–Kahan**：`sin θ ≤ ‖r‖/γ`，γ 为 η 到其余谱的距离下界；
* **Kato–Temple / 谱扰动**：Temple 界的标准来源；
* **Sturm 序列 / Sylvester 惯性定律**：对称三对角矩阵的负主元个数 =
  小于位移的本征值个数。本项目用它**验证**区间，而不是假设区间。

本项目的贡献定位是“把经典界做成神经本征对的可计算协议”，详见
[`certification.md`](certification.md)。不要写成“提出了新的谱界”。

## 4. HHL 论文的关键设定（复现时必须对齐的细节）

| 设定 | 论文做法 | 本项目实现 |
|---|---|---|
| 能量表示 | 用波数 k 当“能量”，对 V=x²/2 有 k_n=√(2n+1) | 统一用 E，并在 `spectrum.py` 文档里给出 k ↔ E 换算 |
| 波函数归一化 | max\|ψ\| = 1 | `to_hhl_normalization`，同时保留物理的 ∫\|ψ\|²=1 |
| 输入 | `(x, E)` 双输入，含一个把 x 映射到远端的技巧 | `inputs.hhl_input_map` |
| 训练域 | x ∈ [−5, 5]，2500 点 | `data/targets.py::HHL_RANGE / HHL_GRID_POINTS` |
| 目标函数 | 10 个连续非线性函数（f(0)=0，\|f\|≤H≈100） | `data/targets.py::HHL_FUNCTIONS` |
| 网络规模 | hidden=10、嵌入层 3、纠缠层 4、输出 2、Adam η=0.05 | `networks.py::build_hhl_network` 的默认参数 |

## 5. 这个项目**没有**主张的事

* 没有主张神经网络能超过经典数值方法。相反，文档中反复强调：一维、光滑势上，
  Numerov/谱方法又快又准，神经网络的**价值在别处**（参数化、数据同化、
  “物理约束 + 少量测量”的融合、以及可微链路的端到端优化）。
* 没有复现论文的量子线路部分。`build_hhl_network` 是经典等价结构；
  接入 PennyLane 只需替换前向计算。
* 没有声称 Temple/Weinstein 界是新结果；贡献是可计算性、验证协议与实验。
* 没有声称证书对**连续** Hamiltonian 严格：证书对象是离散 H_h，
  连续扩展需要额外的 O(h²) 截断误差界（见 `certification.md` §4）。
* 没有跑三维/多体问题；所有结论限定在一维、四个势、n≤3。

## 6. 引用与出处提示

本项目引用的概念（按出现位置）：

* **Numerov 方法**：`physics/numerov.py` 模块文档（Hairer, Nørsett & Wanner）。
* **Jacobi 特征分解**：`physics/eigen.py`（Golub & Van Loan §8.5）。
* **Rayleigh 商 / 变分原理**：`solver/losses.py::energy`。
* **谱偏置（spectral bias）**：`solver/networks.py::FourierFeatures`。
* **Temple/Weinstein/Davis–Kahan**：`solver/certify.py` 模块文档；
  正式引用请用 Kato、Temple、Weinstein、Davis–Kahan 的原始文献。
* **HHL 论文的目标函数与 (x,E) 约定**：`data/targets.py`、`solver/inputs.py`。

> 若要把本项目的实现写成论文/报告，请以上述原始文献的正式引用为准。
> 2025–2026 的检索线索（arXiv/期刊编号）必须逐条核对后再引用，避免
> “看似精确、实则杜撰”的引用。
