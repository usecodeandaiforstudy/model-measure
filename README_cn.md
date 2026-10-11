> English version: [README.md](README.md)

# Capacity Lens

**一套用一个证明和一个测量来回答同一个问题的工具：这个任务至少需要多少表征容量，以及模型小到什么程度任务就装不下了？**

它的做法是把一个训练好的网络当作**速率受限的通信信道**来读。从参照模型的错误率出发，推出任务的信息需求（`R_IB`）与能满足该需求的最小码本（`K_min`）——这一侧是**可证下界**，所以它只能**否证**可行性，永远不会把自己当成承诺；再从一列更小的学生模型里定位**实测墙**（`K_wall`），即精度真正开始崩塌的规模——这一侧是**测量**，项目对它依赖的每一样东西都写明了。

头号规则，也是唯一属于"证明"的部分：

```
        log K  <  R_IB        ⇒        这个配置装不下这个任务
      （供给）    （需求）              无需训练，也不可能被实验推翻
```

| | |
| --- | --- |
| **版本** | `capacity_lens` 0.6.1 —— 10 个模块、37 个导出名，只依赖 PyTorch |
| **测试** | 94 条测试、0 失败；有 pytest 或用零依赖 runner 都能跑 |
| **证据** | 32 个实验脚本、`docs/` 里 21 份文档（每份都有中文孪生版）、33 行台账、26 份记录数据 —— 全部可在**离线**复算，不需要 GPU、不需要联网 |
| **复现** | `python tests/run_all.py` 与 `python tools/reproduce_all.py` |

---

## 1. 思想

### 1.1 为什么常用数字答不了这个问题

参数量、标称位宽、FLOPs、码本大小，说的都是模型**能装多少**。它们不说明**任务需要多少**，因此也就无法说明"换小一点的模型还灵不灵"。Capacity Lens 把两侧都折算成同一个单位（nats）下的信息量，然后只在唯一安全的方向上做比较。

### 1.2 两侧，以及为什么只有一侧可证

| 侧 | 量 | 怎么得到 | 认识论地位 |
| --- | --- | --- | --- |
| **需求** | `R_IB = ln C − h_b(ε) − ε·ln(C−1)` | 由 Fano 不等式，从参照模型在 `C` 类任务上的测试错误率 `ε` 反解 | 任务信息需求的**下界**（前提是这个参照模型确实会做这个任务） |
| **需求** | `K_min = max(2, ceil(e^{R_IB}))` | 使 `ln K` 达到 `R_IB` 的最小码本 | 由上界推出的码本地板 |
| **供给** | `log K`（或 `d·log q`，或拓扑的 min/sum） | 从候选结构的接线方式读出 | 候选能通过的信息量**上界** |
| **判定** | `log K < R_IB` | 算术 | **可证**：供给低于需求意味着任务装不下。方向安全，任何实验都推翻不了 |
| **测量** | `K_wall`＝学生保留参照精度 ≥90% 的最小 `K` | 训练一列学生模型 | **协议相对**：依赖训练配方、预算、学生族、随机种子 |

这个不对称正是全部要点所在。可证排除可以放心据此行动；而墙这个数字必须连同它的测量条件一起引用。

### 1.3 "可证"买到了什么、没买到什么

- 它**确实**意味着：*只要*参照模型真的会做这个任务，那么任何可证容量不达标的架构都做不了它——无论训练技巧、数据、运气如何。
- 它**不**意味着反向成立。这里没有任何东西说某个配置*能*成功、或*能有多好*。项目里每一条判据都是下界；它们只能否证，从不确认。
- 它**在能说话之前保持沉默**。在 `C` 类任务上 `K ≥ C` 永远无法排除；而排除 `K = 2` 就需要参照精度 ≥ 0.577（`K = 4` 需 ≥ 0.807，`K = 8` 需 ≥ 0.966）。低于这些阈值时读数是"**可靠但空**"——冻结特征或廉价见证者正落在这一类。
- 它是**事后**的：`R_IB` 从一个已经work的模型反解而来，所以它校准的是判断，而不是事先预测"该建多大"。

### 1.4 见证者相对性

这是本仓库里实践上最重要的一条发现：

> **需求读数只和它读自的那个模型一样好。** 参照模型偏弱会压低 `R_IB`（造成虚假的"可行"）；而见证者的强度**实质性地决定**这条规则能做出多少真排除。

廉价与冻结的见证者依然**可靠**——它们从不产生假排除——但会漏掉排除，因为它们证不出多少东西。这也是为什么本项目从不把"跑了规则、什么都没被排除"当成好消息。

### 1.5 同一套标准也适用于项目自己的主张

仓库里每个战役都预注册判据、用固定的判定词表给出结论（`SUPPORT` / `PARTIAL` / `MIXED` / `REFUTED` / `UNDECIDABLE` / `RESTRICTED` / `USABLE`），并把**被证伪的预期**同等清楚地留在记录里。失败的结论与成立的结论一样显眼，包括项目自己那些**被取代的旧结论**（见台账与 §9 的否证归档）。

---

## 2. 架构

### 2.1 两层，各司其职

```
capacity_lens/            库：纯函数 + 可插入的层。不含数据集、不做实验、不写任何文件。
                          可单独安装（发行包里只有这个包）。
└── 被谁引用 ──►  examples/quickstart.py        自包含的 30 秒演示
                  experiments/*.py              证据：跑过各个战役的 32 个脚本
                  tests/*.py                    库的不变量 + 每一条预注册判据
                  tools/reproduce_all.py        用记录数据复算全部已发布判定
                  tools/check_wheel.py          校验发行包只装库、不夹带其它东西
                  .github/workflows/tests.yml   每次推送跑上面全部关卡（外加 --selftest）
```

库从不依赖实验；实验只依赖库（外加 torch，其中两个还要 torchvision）。这个方向正是证据可核查的原因：§3 里每一条主张都由一个任何人可重跑的脚本产生，而且其中大多数可以用**零训练**从记录的 jsonl 复算出来。

### 2.2 管线

```
                       ┌─────────────────────── 需求侧（可证）───────────────────────┐
   任务 + 参照模型       │  参照误差 ε  ──Fano──►  R_IB  ──ceil(e^R_IB)──►  K_min      │
   （"见证者"）         │                                                              │
                       └──────────────────────────────────────────────────────────────┘
                                                     │
   候选架构 ──接线──► 容量上界                          │   比较：log K < R_IB  ⇒  否证
   （串行 min / 旁路 sum / MoE 两层 / KV）             │        （无需训练）
                                                     ▼
                       ┌────────────────────── 实践侧（测量）──────────────────────┐
   一列学生模型         │  K = 2, 4, 8, …  ──训练──►  保留率(acc_K / acc_ref)  ──►  K_wall（含容差带）  │
   蒸馏见证者           │  永不高于可证侧的承诺，永远高于 K_min                                        │
                       └──────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                  preflight 判定（五项必检 + 墙几何；
                                  任何一项缺失都让判定降级，而不是悄悄给个数）
```

### 2.3 库暴露了什么

| 模块 | 提供什么 |
| --- | --- |
| `bottleneck.py` | 可插入瓶颈：`VQBottleneck`（离散码本，`metric="euclid"` 或 `"cosine"`）与 `GaussBottleneck`（连续、信息正则化的信道） |
| `capacity.py` | 纯函数的容量上界：`min_capacity_serial`、`sum_capacity_parallel`、`conditional_capacity`（MoE 两层）、`kv_storage_capacity`、`entropy_of`、`mutual_info_of` |
| `training.py` | 最小训练/评估环（自动处理 VQ 承诺损失与可选蒸馏），以及 `task_info_fano`（需求侧核心的 Fano 反解） |
| `distill.py` | `DistillCapacity`：`min_codebook_size(R_IB)` —— 蒸馏容量地板 |
| `quant.py` | `QuantCapacity`：`min_bits(R_IB, d)` —— 每维量化地板 |
| `preflight.py` | 判定机制：`g0_gate`、`distill_plateau_ok`、`distill_gap_k_star`、`measured_wall`、`k_wall_band`、`looseness`、`excluded_set`、`coverage`、`false_exclusions`、`ratio_verdict`、`preflight_report` |
| `geometry.py` | 可分性度量与 `wall_geometry_probe`（离散墙/弥散墙，零训练） |
| `monitor.py` | 训练诊断：码本信息量 `H(Z)`、利用率，以及**为什么刻意不提供"过拟合"捷径** |
| `fisher.py` | `FisherEffectiveDim`（梯度 Gram 矩阵的参与率）与 `arc_length_step` |
| `__init__.py` | 37 个公开名字，以及 `__version__` |

### 2.4 证据纪律

本仓库每个实验脚本都遵守同一形状，而这个形状正是主张可审计的原因：

- **docstring 里的预注册** —— 设计、网格、判据、判定规则，以及铁律（判据先于数据；不重掷种子；失败如实报告、不抹平）。
- **判据写成纯函数** —— 判定逻辑与训练分离，并在 `tests/test_experiment_criteria.py` 里被单测覆盖。
- **`--selftest`** —— 用夹具数据跑这些判据，不训练。
- **接力模式** —— 昂贵的单元会落检查点，所以运行可以续跑而不是重算。
- **`--verdict-only`** —— 从记录的 jsonl 重建判定，**零训练**。
- **追加式产物** —— `<topic>.jsonl`、`<topic>_report.txt`、`<topic>/<fingerprint>/` 检查点，因此修订是可见的，而不是互相覆盖。
- **台账条目** —— `docs/ledger.md` 记录判定、数字，以及（重要的）**产生该判定的判据自身被记录的缺陷**。

---

## 3. 现在能做到什么程度，每条主张立到什么份上

下表每一行都有已记录的战役支撑；最后一列点名的笔记载有设计、数字与边界。

| # | 能力 | 已达到的状态 | 真正要命的边界 | 细节 |
| --- | --- | --- | --- | --- |
| 1 | **可证排除**（`log K < R_IB` ⇒ 装不下） | 在所有检验过的战役里**零假排除**：4 个类数 × 3 种训练配方 × 2 个数据集（合成格点 + CIFAR-10） | 只是必要条件；见证者弱时它沉默（不提供信息） | `witness-strength.md`、`cifar10-anchor.md` |
| 2 | **需求读数** `R_IB`、`K_min` | 在贝叶斯误差可算的族上与蒙特卡洛真值吻合（格点、随机中心混合、16 维混合） | 事后量；给定见证者的下界；`K ≥ C` 永不可排除 | `cross-task-comparability.md` |
| 3 | **实测墙** `K_wall`（含容差带） | 在细阶梯上墙按 **`C^0.603`** 增长（R² = 0.988，n = 4 个类数），而需求按 **`C^0.58`** 增长 —— 即**墙与需求几乎同速** | 它是阈值穿越：必须带容差带引用（`rel_tol` 0.05/0.10/0.15）；在平坦曲线上单个数字可能差 4 倍 | `wall-remeasurement.md` |
| 4 | **松紧** `ln K_wall / R_IB` | 在 38 个场景的普查中 `K_wall > K_min` **恒成立**，幅度 1.5–48× | 它不是常数、也不能由任务预测；它正是"可证界少承诺了多少"的诚实度量 | `bound-looseness.md` |
| 5 | **墙的配方依赖** | 墙是**协议相对**的，且配方效应是**任务 × 预算 × 调度**的相互作用：预算够时退火把赤字打平（0.12 → −0.008），预算不够时反而**欠训练**（一道墙被推向错误方向，8 → 32）；在 CIFAR-10 的记录档上固定学习率直接训不动（保留率 **+0.612 vs +0.117**），**3/3 种子** | 不存在"最佳配方"这种主张 —— 只有"调度与预算一起选" | `budget-and-recipe.md` §6–§7、`wall-remeasurement.md` §7 |
| 6 | **预算依赖** | **作为定律被否证**：保留赤字不是预算的幂律、也不单调（0.121 → 0.069 → 0.109），存在内部最优 | 早先"预算 ×3 赤字减半"的读法不成立为规则（其单个配对的数字仍然成立） | `budget-and-recipe.md` |
| 7 | **跨任务可比性** | 在同一个合成族内读数能**正确排序**（0 次实质倒置）；跨族则**不能** —— 偏移可达 1.4 nats；而且即便族内，不变性也**以两个见证者质量可比为前提** | 永不跨任务/跨族比较读数；要比就比"一个读数 vs 一个码本大小" | `cross-task-comparability.md` §5–§6 |
| 8 | **见证者强度决定覆盖** | SUPPORT：廉价见证者依然可靠但会漏排除；"跳过见证者"的代价被量化了 | 弱见证者给出低的 `R_IB`，让人误以为"可行" | `witness-strength.md` |
| 9 | **可达的尺度轴** | 样本数：MIXED（受数据限制而非算力）。宽度：SUPPORT 但边际且脆弱。分辨率：SUPPORT —— 读数不变（饱和时 Δ = 0.000，见证者欠训练时 0.198） | 离线可达的只有这三条轴；自然多样性与大规模预训练模型不可达 | `scale-axes.md` |
| 10 | **蒸馏选型**（`K_min`、`K*` 定位） | `K_min` 只需见证者精度；在离散墙任务上可定位最小可行 `K*`，经验裕度 1.7–3.0 | 真实任务（CIFAR-10）上排除成立 3/3，但 `K*` 定位**不迁移** —— 弥散墙只提供排除 | `cifar10-anchor.md` |
| 11 | **量化地板** `q_min = 2^(R_IB/d)` | 在合成任务上 3 种子验证（预测悬崖 4，实测 3.81–3.92）；QAT 消除 PTQ 校准伪影 | 只有 `d` 小且任务信息量大时才有分辨力；在 MNIST/CIFAR 规模退化为 ≈1 bit | `mnist_demo.py`、`match-judgment-gaps.md` |
| 12 | **接线规则**（串行 `min`、旁路 `sum`） | 在 MLP 上 VERIFIED；在真实 Transformer 上，残差旁路意味着 FFN 分支内的 VQ **根本不是供给**（K = 2 仍保留 97%；去掉旁路后同一个 VQ 崩到 0.101） | 把串行 `min` 规则照字面套到逐层 `log K` 会**误杀**有效配置；先读拓扑 | `transformer.md` |
| 13 | **Transformer 排除管线** | SUPPORT：可证排除与定位墙在池化串行路径上端到端跑通，**并且**确认了结构性边界 | 在残差流内部，容量读法不适用 | `transformer.md` |
| 14 | **MoE / KV 约定** | MoE 两层 `conditional_capacity` VERIFIED（route64，6 种子）。KV 存储层 VERIFIED（每 token 码本不变而存储 ×8）；KV **流量层**是乐观地板（实测常数高 1.5–4×） | 流量层的锚点必须是**实测的稠密墙**，而不是 Fano 下界 | `kv-memory.md` |
| 15 | **连续瓶颈** | PARTIAL（3 条判据中 2 条）：两层读法可迁移，但严格界**高估可用容量 ≈5.3×** | 高斯信道要用可用性修正，而不是原始界 | `continuous-bottleneck.md` |
| 16 | **预训练/冻结见证者** | MIXED–REFUTED：预训练模型没能替代训练过的见证者（读数依然可靠，但失去了牙齿） | 见证者要训练、要健康（见 G0 门） | `pretrained-witness.md` |
| 17 | **族内摊销** | MIXED–REFUTED（3 条中 1 条）：族内标定在数量上可行，但达不到判定级；排除是**阈值函数**，阈值余量小时必须实测 | 余量小时就测；规则预测的是"能否排除"，不是幅度 | `family-amortisation.md` |
| 18 | **墙几何探针** | 零训练地把任务的墙分类为离散/弥散；阈值由剂量响应实验标定，并在 MNIST 上做过前瞻验证 | 对未过 G0 前提的欠训练参照模型，表征约定偏乐观 —— 先过前提再探 | `output-mapping-discreteness.md` |
| 19 | **训练期监控** | 训练中给出 `H(Z)` 与码本利用率（单位 nats）；**刻意不提供**"过拟合"捷径 | `H(Z)` 是 `I(X;T)` 的上界，会高估任务信息（5.36 vs 4.16）—— 要与任务比较容量，走 Fano 方向 | `monitor.py` docstring |

### 3.1 五步法

给定一个任务与一个候选网络 —— **只有第 ④ 步是硬判断**，其余是它的前提与边界：

| 步 | 内容 | 工具 | 状态 |
| --- | --- | --- | --- |
| ① | 读接线（串行 / 旁路 / 注意力 / MoE），决定用哪个容量公式 | §3.2 的接线表 | 串行、旁路、MoE VERIFIED；注意力 CANDIDATE |
| ② | 测需求：由见证者误差得 `R_IB`，再得 `K_min` | `task_info_fano`、`DistillCapacity` | VERIFIED，含真实任务 |
| ③ | 由接线算供给 | `min_capacity_serial` / `sum_capacity_parallel` / `conditional_capacity` / `kv_storage_capacity` | 串行、旁路、MoE VERIFIED |
| ④ | **判定排除**（`log K < R_IB` ⇒ 装不下），并报告 `K_wall` 与容差带用于选型 | `preflight_report` | 排除：零假排除；墙：协议相对 |
| ⑤ | 什么都没被排除时，在**同一任务内**给训练配置排序 | `opt_demo` | VERIFIED（两个任务上 regret 0） |

### 3.2 接线表（第 ① 步的字典）

| 架构 | 存储层（装得下吗） | 流量层（单个样本能流过吗） | 依据 |
| --- | --- | --- | --- |
| 纯串行 | 各瓶颈取 `min` | `min` | VERIFIED |
| 旁路 / 残差 / 并行 | `sum`（每条支路各自计入） | ≥ 各路径中最大者 | MLP 上 VERIFIED；后果在 Transformer 上确认 |
| MoE | shared + `n_experts·log K_e` | shared + `top_k·log K_e` | VERIFIED（route64，6 种子） |
| 注意力 KV | `seq_len × 每 token` 信息量 | 由读出侧的流量层决定 | 存储层 VERIFIED；流量层是乐观地板 |

---

## 4. 它做不到什么

直说，因为一个工具正是由它拒绝做的事定义的：

- **不能证明可行。** 所有判据都是下界；它否证与测量，从不承诺某个配置会work。
- **不能预测性能**、悬崖可定位性、训练难度 —— 任何信息论/几何/检测论量都不行：那条路线被测过并失败了。
- **不能跨任务比较架构优劣。** 训练难度只在同一任务内可比。
- **不能事前定尺寸。** `R_IB` 是事后的，读自一个已经work的模型。
- **见证者弱或码本塌掉时说不出任何东西** —— preflight 门会拒绝给出数字，而不是给出一个误导性的数字。
- **不能回答"模型是否过拟合"**（靠 `H(Z)` 不行）；该判据作为不可靠已被删除。
- **没有大规模验证。** 可达证据是合成任务族 + MNIST + CIFAR-10；自然多样性与大规模预训练模型超出本环境可测范围。

---

**每份文档都有中英两版**：中文版文件名 = 英文名 + `_cn`（`README.md` → `README_cn.md`，
`docs/transformer.md` → `docs/transformer_cn.md` ✓），且孪生版必须与英文版**同标题、同数字、同文件名引用、同代码** ✓ ——
`python tools/check_bilingual.py` 就是机械校验这件事的 ✓，CI 会跑它 ✓。`LICENSE`、`CITATION.cff`、`.gitignore`、
`pyproject.toml` 与记录证据**有意豁免**：它们要么文件名被工具读取、要么内容不是散文 ✓。

## 5. 安装、快速开始、测试、复现

```bash
pip install -e .                          # 库本体（只依赖 torch）
pip install -e ".[experiments]"           # 追加 torchvision，供 CIFAR 实验用
pip wheel . --no-deps -w dist/            # 构建发行包：只含 capacity_lens/ 与许可证
```

**快速开始 —— 30 秒、无需数据集、无需 GPU：**

```bash
python examples/quickstart.py
```

```
[1] reference: acc=0.977 (err=0.023)  H(Z_t)=5.17 nats
    requirement R_IB = 3.11 nats  ->  K_min = ceil(e^R_IB) = 23
[2] certified exclusion (log K < R_IB): [2, 4, 8, 16]
        student K=  32: acc=0.764  relative gap=+0.22
        student K=  64: acc=0.978  relative gap=-0.00
[3] practical wall K_wall = 64 | looseness = ln(K_wall)/R_IB = 1.34
    measured: succeeds=[64, 128, 256] fails=[2, 4, 8, 16] | coverage=1.00 | false exclusions=[]
[4] preflight: VERDICT USABLE: K*=64; practical wall K_wall=64 (certified bound sits x1.3 lower)
```

怎么读它：边界是真的，因为**每一个被排除的 K 都真的失败了**（`false exclusions=[]`）；可证侧是保守的，因为墙（64）高于地板（23）—— 这个差就是 `looseness` 度量的东西。

**测试与离线复现：**

```bash
python tests/run_all.py                  # 94 条测试，零依赖
python tests/run_all.py capacity         # 只跑某个模块
pytest tests/                            # 装了 pytest 时跑同一批测试
python tools/reproduce_all.py            # 从 experiments/data/*.jsonl 重建全部已发布判定
python tools/reproduce_all.py --check    # ... 并与已提交快照 diff
python tools/check_wheel.py --selftest   # wheel 内容检查器自证"能说不"
python tools/check_bilingual.py          # 每份文档都有与之一致的中文孪生版
```

**持续集成**（`.github/workflows/tests.yml`）在每次推送时跑同一批关卡，**不需要 GPU、数据集或联网**：测试套件跑 Python 3.9/3.11/3.13 ✓、每个声明了 `--selftest` 的实验脚本各跑一次 ✓、离线判定复算（判定重建不出来、或与已提交快照不一致 ⇒ 构建失败 ✓）、构建 wheel 后跑 `tools/check_wheel.py` 与公开示例 ✓、以及 `tools/check_bilingual.py`（**每份文档都必须有与之一致的中文孪生版，缺一份就红** ✓）。

`tools/reproduce_all.py` 在**进程内**导入每个实验模块（绝不用子进程——复现工具不该依赖沙箱模式），调用它的 `verdict_only()`，把**复算**判定与**记录**判定并排打印，对没有离线入口的模块做分类而不是假设它有，并写出 `tools/reproduce_all_snapshot.json` 供两次运行 diff。当前状态：**12 个可复算模块全部复算成功、11 个只暴露判据函数、9 个是有记录可查的早期代**。无需 GPU、无需数据集、无需联网。

**数据集**从不由库下载；手工放一次即可，之后全程离线安全（数据已存在时 `download=True` 立即返回）：

```
experiments/data/
├── MNIST/raw/                    train-images-idx3-ubyte, train-labels-idx1-ubyte,
│                                 t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
└── cifar-10-batches-py/          data_batch_1 … data_batch_5, test_batch, batches.meta
```

**运行证据脚本：**

| 档 | 预算 | 出判定吗 |
| --- | --- | --- |
| 默认 | 全量数据、完整 epoch | 出 |
| `--fast` | 20–30% 子集、更少 epoch | 出（判据逻辑相同） |
| `--quick` | 最小预算 | 不出（只做冒烟测试） |

当前代脚本还支持 `--fresh`（重算而非用接力缓存）、`--verdict-only`（离线复算判定）、`--selftest`（夹具数据跑判据）；最早的三套用 `--only` 选单个子实验
（`mnist_demo.py`：`p1,p2,p3,p4,quant,qat,distill,distill-mnist,fisher`；`cifar_demo.py`：`c1,c2,c3,c3-cifar,c4`；
`transformer_demo.py`：`t1,t2`），例如 `python experiments/mnist_demo.py --fast --only distill`。
最早的九个脚本（`cifar_demo`、`cp3_demo`、`cp3_dyn`、`cp3_dyn_sens`、`cp3_multilevel`、`cph_post`、`mnist_demo`、`opt_demo`、`transformer_demo`）早于这些约定，**刻意保持原样**：它们记录的产物正是这批代码产生的，改写它们会让数字与产生它的代码脱钩。见 `CONTRIBUTING.md`。

---

## 6. 使用库

**零训练筛查** —— 最常见的用法，既不需要插瓶颈、也不需要重训：

```python
import math
from capacity_lens import task_info_fano, DistillCapacity

C = 27
acc_reference = 0.977
R_IB  = task_info_fano(1.0 - acc_reference, C)       # 任务需求下界（nats）
K_min = DistillCapacity().min_codebook_size(R_IB)    # max(2, ceil(e^R_IB))

K_candidate = 16
if math.log(K_candidate) < R_IB:
    print("certainly does not fit -- no training needed")     # 可证、方向安全
```

**完整 preflight 判定**（五项必检；任何一项缺失都让判定降级，而不是悄悄给个数）：

```python
from capacity_lens import wall_geometry_probe, preflight_report

probe = wall_geometry_probe(teacher_feats, teacher_labels)          # 一次前向，零训练
report = preflight_report(acc_t, h_z_t, k_teacher, k_list, accs, acc_min=0.95,
                          wall_probe=probe, r_ib=R_IB)
print(report["verdict"])        # "VERDICT USABLE: K*=…; practical wall K_wall=128 (certified bound sits x3.9 lower)"
print(report["wall_band"])      # {0.05: 256, 0.10: 128, 0.15: 128} —— 墙要带容差带引用；
                                # 摆动 ≥4× 时判定会写明 "TOLERANCE-SENSITIVE"
```

**插入瓶颈**（诊断脚手架 —— 训练目标不变）：

```python
from capacity_lens import VQBottleneck, TrainMonitor

net.vq = VQBottleneck(d=64, n_codes=256)            # 或 metric="euclid"，或 GaussBottleneck 做连续信道
monitor = TrainMonitor(net.vq)
...
out, idx = net(x)
m = monitor.step(idx)                               # {"I_XT": ..., "utilization": ...}，单位 nats
```

**拓扑决定用哪个容量公式**（弄错会严重低估 —— 实测：同一组串行瓶颈，只加一条并行支路，0.103 → 0.876）：

```python
from capacity_lens import min_capacity_serial, sum_capacity_parallel, conditional_capacity, kv_storage_capacity

min_capacity_serial([log_K1, log_K2])                       # 纯串行、无旁路
sum_capacity_parallel([log_K_chain, log_K_branch])          # 残差 / 并行
conditional_capacity(n_experts=8, per_expert=math.log(512), top_k=2, shared=math.log(256))
kv_storage_capacity(seq_len=4096, per_token=0.5)            # 乐观地板，见 kv-memory.md
```

**蒸馏选型与量化地板：**

```python
from capacity_lens import DistillCapacity, QuantCapacity, measured_wall, k_wall_band, looseness

DistillCapacity().min_codebook_size(R_IB)       # 码本地板
QuantCapacity().min_bits(R_ib=3.2, d=8)         # 每维比特
k_wall, rel = measured_wall(k_list, accs, acc_ref, rel_tol=0.10)
k_wall_band(k_list, accs, acc_ref)              # 同一道墙在 0.05 / 0.10 / 0.15 下
looseness(R_IB, k_wall)                         # ln(K_wall) / R_IB
```

---

## 7. 仓库地图 —— 每个文件是干什么的

### 7.1 `capacity_lens/` —— 库（10 个文件，无副作用）

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 37 个公开名字与 `__version__` |
| `bottleneck.py` | `VQBottleneck`（离散，euclid/cosine 度量）与 `GaussBottleneck`（连续信道） |
| `capacity.py` | 纯容量公式：`min_capacity_serial`、`sum_capacity_parallel`、`conditional_capacity`、`kv_storage_capacity` |
| `distill.py` | `DistillCapacity`：`min_codebook_size`，蒸馏地板 |
| `quant.py` | `QuantCapacity`：`min_bits(B, d)`，量化地板 |
| `training.py` | 最小训练/评估环（处理承诺损失与蒸馏）、`entropy_of`、`mutual_info_of`、`task_info_fano`、`joint_code`、`arc_length_step` |
| `preflight.py` | 判定机制：`g0_gate`、`distill_plateau_ok`、`distill_gap_k_star`、`measured_wall`、`k_wall_band`、`looseness`、`excluded_set`、`coverage`、`false_exclusions`、`ratio_verdict`、`preflight_report` |
| `geometry.py` | `silhouette_by_label`、`separation_ratio`、`vq_quantization_error`、`err_1nn`、`wall_geometry_probe` |
| `monitor.py` | `TrainMonitor`：训练期的码本信息量与利用率（以及有记录的"不提供过拟合捷径"） |
| `fisher.py` | `FisherEffectiveDim`、`FisherArcLength`、`arc_length_step` |

### 7.2 `experiments/` —— 证据（32 个脚本，各自独立且给出判定）

**核心验证线**

| 脚本 | 作用 |
| --- | --- |
| `mnist_demo.py` | MNIST + MLP 线：容量紧致性（P1–P3）、串行拓扑界（P2）、旁路反例（P4）、量化与蒸馏筛查、Fisher 曲线 |
| `cifar_demo.py` | CIFAR-10 + CNN：这套读法能否推广到卷积模型与更难的任务 |
| `transformer_demo.py` | 小 Transformer 线：紧致性（T1）与残差流下 `min` 的失效（T2） |

**排除规则与它的见证者**

| 脚本 | 作用 |
| --- | --- |
| `witness_demo.py` | 见证者强度如何决定 `R_IB` 与排除覆盖（"跳过见证者"的实测代价） |
| `pretrained_witness_demo.py` | 预训练模型能否在目标侧零训练的前提下充当见证者 |
| `tf_exclusion_demo.py` | 排除管线在 Transformer 上的端到端验证及其结构边界 |
| `cph_demo.py` | CIFAR-10 上的真实任务双轴验证（排除 + 定位是否迁移） |
| `cph_post.py` | CP-H 轴二灰区的零训练后分析 |
| `pa1_demo.py` | P-A1-3 预算阶梯判定（`K_t = 128` 有效容量审计） |
| `pa3_demo.py` | A-003 的开放预测：双口径离散性与墙探针的前瞻验证 |
| `pa4_demo.py` | P-A1-4 剂量响应：悬崖尖锐度 vs 输出映射离散性 |

**墙（测量侧）**

| 脚本 | 作用 |
| --- | --- |
| `looseness_demo.py` | 可证界比实测墙松多少（全任务普查） |
| `looseness_reaudit_demo.py` | 被证伪的松紧判据的更正版预注册，以及真值拆分 |
| `wall_saturation_demo.py` | 格点族里实测墙为何停止增长 |
| `wall_remeasure_demo.py` | 在更好配方下重测各墙、CIFAR-10 两个档位的配对探针、细阶梯增长检验与种子扫描 |
| `budget_dose_demo.py` | 保留赤字是不是训练预算的幂律 |
| `budget_mechanism_demo.py` | 配方里哪个成分让赤字非单调（2×2 因子设计 + 墙的迁移） |
| `scale_c_demo.py` | 尺度边界的类数轴（在项目自己的格点族上） |
| `scale_c_clean_demo.py` | 同一轴，但把墙的天花板固定住（决定性测量） |
| `scale_axes_demo.py` | 离线可达的三条尺度轴：样本数、参数宽度、分辨率 |

**跨任务可比性**

| 脚本 | 作用 |
| --- | --- |
| `cross_task_demo.py` | 是否存在跨任务的共同难度尺度（台账 P-A1-2） |
| `family_b_demo.py` | 第二个合成族（随机中心混合），真值可算 |
| `family_c_demo.py` | 第三个族（16 维混合）—— 不变性在这里被发现是**有条件的** |
| `family_library_demo.py` | 族内标定能否让读数对新实例变成准先验 |

**其他架构与信道**

| 脚本 | 作用 |
| --- | --- |
| `gauss_demo.py` | 两层读法与排除管线能否迁移到连续瓶颈 |
| `kv_demo.py` | 内容可寻址记忆读法在联想回忆上的验证 |
| `moe_demo.py` | 两层读法在路由专家网络上的验证 |
| `cp3_demo.py` | 度量干预（架构 × VQ 度量），拆解"架构族"这个复合变量 |
| `cp3_dyn.py`、`cp3_dyn_sens.py`、`cp3_multilevel.py` | CP3 动力学后续：优化器的信号、早期窗口敏感性、多层臂 |
| `opt_demo.py` | 训练闭环配置优化器（方法的第 ⑤ 步） |

### 7.3 `tests/` —— 不变量与每一条预注册判据（10 个文件，94 条测试）

| 文件 | 作用 |
| --- | --- |
| `run_all.py` | 零依赖 runner（`python tests/run_all.py [子串]`） |
| `conftest.py`、`support.py` | 路径引导与最小 `approx` 辅助，使测试在有/无 pytest 时都能跑 |
| `test_capacity.py` | 容量公式与接线规则 |
| `test_bottleneck.py` | 瓶颈层，含 Gauss 战役背后的回归 |
| `test_training.py` | 训练胶水：熵、互信息、Fano 反解，一个端到端冒烟 |
| `test_preflight.py` | 头号规则：排除判据、墙、容差带、preflight 降级路径 |
| `test_quant_distill.py` | 量化与蒸馏筛查 |
| `test_geometry.py` | 可分性度量与墙几何探针 |
| `test_experiment_criteria.py` | 每个实验模块的预注册判据函数，全部被导入并执行 |

### 7.4 `tools/` 与 `examples/`

| 文件 | 作用 |
| --- | --- |
| `tools/reproduce_all.py` | 一条命令、离线地复算全部已发布判定 |
| `tools/reproduce_all_snapshot.json` | 上一次复现运行的可 diff 结果 |
| `tools/check_wheel.py` | 校验构建出的 wheel 只装库、不夹带其它东西（`--selftest` 证明它"能说不"） |
| `tools/check_bilingual.py` | 校验每份文档都有 `_cn` 孪生版，且标题、数字、文件名引用与代码与英文版一致 |
| `examples/quickstart.py` | 自包含的 30 秒演示（自建任务，只导入本包） |

### 7.5 `docs/` —— 21 份文档，每份都有中文孪生版

**每份文档都有中英两版**：中文版文件名 = 英文名 + `_cn`（`docs/transformer.md` →
`docs/transformer_cn.md` ✓），孪生版必须与英文版**同标题、同数字、同文件名引用、同代码** ✓ ——
`python tools/check_bilingual.py` 正是机械校验这些的 ✓，CI 不通过就是红的 ✓。`LICENSE`、`CITATION.cff`、
`.gitignore`、`pyproject.toml` 与记录证据（`*.jsonl`、`*_report.txt`）**有意豁免**：它们要么文件名被工具读取、
要么内容不是散文 ✓。

新读者按这个顺序读：

| 顺序 | 文档 | 给你什么 |
| --- | --- | --- |
| 1 | `framework.md` | 理论框架：容量、可分性、训练难度 —— 以及三者中哪个才是硬判据 |
| 5 | `ledger.md` | **台账**：一张 33 行的表 —— 每个战役一行，含判定、关键数字，并链接到载有完整记录的那份文档 |
| 6 | `verification-plan.md` | 每个战役**预注册**的成功/失败判据，已压缩成表 |
| 6 | `falsified.md` | 否证与修订归档 —— 什么被推翻了、又被什么取代 |

按主题：

| 文档 | 主题 |
| --- | --- |
| `witness-strength.md` | 见证者强度及其对需求与覆盖的控制 |
| `pretrained-witness.md` | 预训练见证者 |
| `bound-looseness.md` | 可证界 vs 实测墙 |
| `kv-memory.md` | KV/内容可寻址记忆读法及其乐观常数 |
| `continuous-bottleneck.md` | 连续瓶颈线及其可用性修正 |
| `transformer.md` | Transformer 排除管线与结构边界 |
| `family-amortisation.md` | 族内摊销 |
| `cross-task-comparability.md` | 跨任务可比性、三个族，以及不变性的成立条件 |
| `class-count-axis.md` | 类数轴 |
| `scale-axes.md` | 样本数、宽度、分辨率 |
| `wall-saturation.md` | 墙为何看起来饱和，以及真正限制它的是什么 |
| `budget-and-recipe.md` | 预算轴、其背后的机制，与墙的迁移 |
| `wall-remeasurement.md` | 墙的重测、真实数据探针、增长指数及其种子扫描 |
| `cifar10-anchor.md` | CP-H / PA1 判定报告（真实任务 + 反例审计） |
| `output-mapping-discreteness.md` | A-003 原始分析：输出映射离散性与悬崖形状 |
| `match-judgment-gaps.md` | 任务↔网络匹配判断还能如何补全 |
| `artifacts.md` | 运行产物清单：每个 jsonl、报告与检查点目录，以及它们各自支撑哪条主张 |

`docs/docs-zh/` 存放这些文档翻译前的中文快照；它不提交。

### 7.6 数据与产物（哪些提交、哪些不提交）

| 路径 | 内容 | 提交？ |
| --- | --- | --- |
| `experiments/data/*.jsonl` | 顶层 26 份记录结果 —— 每条判定都从它们重建 —— 外加嵌套在 `cph/` 里的两份 CP-H 读数（合计 751 KB） | **提交** —— 没有它们，`tools/reproduce_all.py` 什么都复算不了 |
| `experiments/data/*_report.txt` | 26 份追加式报告，含判据被修改时的修订轨迹（96 KB） | **提交** —— 记录的判定原文在它们里面 |
| `experiments/data/<topic>/<fingerprint>/` | 约 2000 个检查点（1.7 GB），按运行指纹分目录，因此改过的配方不会悄悄复用别的运行的数字 | 不提交 —— 重跑即可再生 |
| `experiments/data/MNIST/`、`experiments/data/cifar-10-batches-py/` | 手工放置的数据集（233 MB） | 不提交 —— 见 §5 自行放置 |
| `reference/` | 本地工作材料：翻译前快照、已退休的过程文档、旧运行转储（0.9 MB） | 不提交 —— 已 gitignore，仅留在本机 |
| `docs/`、`docs/`、`docs/` | 本地工作材料：历史快照与草稿 | 不提交 |

### 7.7 根文件

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | 打包：setuptools、从 `capacity_lens.__version__` 读取的动态版本、extras `experiments` / `test` / `dev`、pytest 配置 |
| `CHANGELOG.md` | 库的逐版本变更（0.5.0 → 0.6.1） |
| `CONTRIBUTING.md` | 变更必须满足的规则、标准实验形状、"两代脚本"政策 |
| `CITATION.cff` | 引用元数据（仓库 URL 故意留空，直到真的有仓库） |
| `LICENSE` | MIT |
| `.gitignore` | 排除数据集、检查点、运行产物与中文文档快照 |
| `README_cn.md`、`CONTRIBUTING_cn.md`、`CHANGELOG_cn.md` | 三份根文档的中文孪生版 —— 同结构、同数字、同文件名引用、同代码，均由脚本机械校验 ✓（规则见 §7.5） |

---

## 8. 怎么核对上面任何一条主张

1. **找到数字。** §3 每一行都点名了载有其设计与结果的笔记，笔记里点名了它的 jsonl 与报告。
2. **找到台账条目。** `ledger.md` 记录判定、数字，以及重要的——**产生该判定的判据自身被记录的缺陷**。
3. **零训练复算。** `python tools/reproduce_all.py` 从记录数据重建全部已发布判定，并与记录文本并排打印；`--check` 与已提交快照 diff。
4. **重跑判据函数。** `tests/test_experiment_criteria.py` 执行的正是拼装判定用的那些纯函数。
5. **重跑实验。** 每个脚本的 docstring 就是它的预注册；接力缓存让部分重跑很便宜，`--fresh` 则全部重算。

## 9. 版本、引用、许可

- **版本**：库版本写在 `capacity_lens/__init__.py`，由 `pyproject.toml` 动态读取；`CHANGELOG.md` 记录改了什么、为什么。目前没有发布到 PyPI —— 把版本号当作开发里程碑。
- **引用**：元数据在 `CITATION.cff`；仓库 URL 故意留空，直到真的有仓库为止，因为编造一个 URL 正属于本项目拒绝的那类细节。
- **许可**：MIT。
