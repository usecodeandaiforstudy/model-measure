# 运行产物清单（artifacts）

> 定位：本文件是**记录证据的清单**——每个战役产出的每一份 jsonl、报告与检查点目录，外加 README 重写期间被移出的非正式表述与过程记录。它们是真实的，保留下来是为了可追溯，但它们是过程记录而非产品约定：README 只承载正式内容，读者想知道哪条主张由哪份产物支撑，就来这里查。
> 发布说明：已记录的 `*.jsonl` 与 `*_report.txt` 文件**是提交的**（`tools/reproduce_all.py` 离线复算的就是它们），中文孪生版（`*_cn.md`）同样提交 —— 它们是文档，由 `tools/check_bilingual.py` 保证每一对都与英文版一致；检查点、数据集以及 `docs/` 下的归档不提交。
>
> **关于已发布证据的三点说明**（项目不修改已记录的产物，以便数字与产生它的那次运行保持耦合；唯一的例外记录在下面）：
> 1. **八份早期报告含中文文本**（合计约 830 个字符：`cp3_dyn_report.txt`、
>    `cp3_dyn_early_report.txt`、`cp3_dyn_sens_report.txt`、`cp3_multilevel_report.txt`、`pa4_report.txt`、
>    `moe_report.txt` 以及另外两份），另有一行 jsonl（`cp3_multilevel.jsonl`）带有 14 个中文字符。
>    它们产生于项目"文档只用英文"这条规则之前，因此按记录原样保留。它们**现在**特殊的地方并不在于它们是中文：本仓库每份文档都有中英两版（`<name>.md` 与 `<name>_cn.md`），所以这八份报告是这里**唯一没有英文孪生版**的中文文本。
> 2. **有一处报告表头被中性化了，这是对已记录报告做过的唯一一次编辑。**
>    `cph_post_report.txt` 第 1 行过去打印它所读 jsonl 的带盘符绝对路径；现在打印的是仓库相对路径（`experiments/data/cph/cph_readings_0d8045d2.jsonl`）。这一改动只涉及生成表头里的一条本机路径，不涉及数字，所以报告所*测量*的内容没有任何变化——而且它被记录在这里，而不是悄悄改掉。
> 3. **另有四处本机环境名被中性化。** `cifar10-anchor.md` §3、其中文版、`cph_demo.py` 第 127 行与
>    `pa1_demo.py` 第 18 行原本写明了战役运行所用的虚拟环境名。现在改为"同一台 GPU 机器"/"a GPU"。
>    判据、数字、判定与代码路径均未改动；记录的 torch 版本与显卡型号保留。

---

## I. 过去用来开篇 README 的理论表述（战役体原文，现已中性化为当前 README 的"立足点"）

> 把神经网络当作一个信号处理系统：训练 = 在**输出空间**里为输入找到一个合适的位置，使任务要求的区分关系成立；内部状态只是一条信道，属性只在输出边界上定义。由此拆出三条约束（是"与"的关系，不是三个旋钮）：(1) **信息容量**——输出边界能带过多少关于输入的信息；(2) **可分性**——与任务相关的差异是否在输出空间里铺开；(3) **训练难度**——优化能否把输入移动到正确的位置，代价如何。
>
> 三条轴已经过一轮穷尽式检验：**(1) 通过**（唯一拿到硬判据的轴），**(2) 收口**（全部有效内容都被容量吸收；三个位置上的负判定各有其理由），**(3) 通过但只在一个任务内可比**。因此工具的能力边界被钉成：**它能判定"学不会 / 装不下"（必要条件的否证），不能判定"学得好"（充分条件的成立）**。

（"钉成""穷尽式检验"这类措辞在正式 README 中被改为中性的事实陈述。）

## II. 过去"2026-09-30 战役对比结果"一节的全文（战役叙事版；正式 README 只保留中性的验证矩阵）

| 线 | 模型 | 实验 | 验证内容 |
| --- | --- | --- | --- |
| 合成 grid64 | CNN-euclid | **PA1 阶梯（P-A1-3，2026-09-30）** | **A-001 反例审计（判定 A：反例不成立）**：预算阶梯 40→200ep × 3 种子；K_t=128 教师在每个种子下 120ep 都过门、200ep 都为 1.0000 ——"容量足够但铺不开"不存在；它属于轴 3 的"余量薄 + 代价高"（约 120ep 对健康层的约 40ep）。**已归档的"K_t=128 最终无效"被推翻**（那段平台是轨迹/预算伪影）；顺带发现，从 160→200ep，H(Z_t) 从 4.63 走到 4.32，而 acc→1.0（后期阶段在压缩非任务熵） |
| 真实 CIFAR-10 | CNN width96 | **CP-H 轴 2（2026-09-30）** | **收口原理的域外检验（混合结果；弱形式存活）**：S1 前提成立（err_1nn=0.228，silhouette=0.187；确认处于非饱和区）；S2 ×4 全部通过（I_plug/silhouette/separation/d′ 在拐点窗口内携带质量信息，rho(acc) 0.60–0.92，与合成饱和区的"恒定"形成对照）；S3 全程灰区（rho(logK) 0.64–0.82）→ **H_indep 不成立、H_proxy 不确认**：没有找到独立轴，但强的"纯代理"形式也不成立——可分性与容量高度纠缠。仲裁观察：没有任何读数胜过 H(Z)/acc |
| 合成 grid64 | CNN-euclid | **PA4 剂量响应（P-A1-4，2026-09-30）** | **悬崖尖锐度 = 输出映射离散性的函数（判定 SUPPORT；A-003 条目已开）**：δ∈{0,0.15,0.3,0.5} × 3 种子；可定位性 0.67→0.33→0.33→0.00，尖锐度 0.62→0.60→0.53→0.40，单调非增；Q1 排除 12/12 通过（K_min 随 δ 自适应下移）；err_1nn 0.000→0.390 单调上升——**对"有墙无悬崖"的一次受控复现与机制归因**，并且 `wall_geometry_probe` 的阈值 0.02 得到正式标定。第一轮的 δ=0.5"否证"是网格覆盖缺陷（照实记账） |
| 合成 route64 | CNN × oracle 路由 MoE | **MOE 两层容量读法验证（2026-09-30，6 种子）** | **`conditional_capacity` 的两层公式得到验证**：T2 存储层通过（联合码本利用率 84-94%）；F1 流量层以**实测稠密墙**为参照成立（routed 在 flow=ln256 达到 1.000/0.999；F1b 3/3：等总码本下路由不差、在高档位更好——A-002 §5.3 干扰机制的证据）。**第一轮预注册否证的根源 = 参照锚点选错**（Fano 下界 ln64 把实测墙低估 4×，与 CP-H 的 32× 余量同族）：**流量层的锚点必须是实测墙，而不是 Fano 下界**（照实记账） |
| 真实 MNIST / 合成检查点池 | MLP / CNN | **PA3（A-003 开放预测，2026-09-30）** | **P-A3-1 前瞻（SUPPORT）**：在未参与标定的 MNIST 上，探针预测弥散×3，且实测不到有意义的悬崖（1/3 < 2/3）——正向臂前瞻通过；**P-A3-2 双口径（PARTIAL）**：rho=+0.936 通过，且 0.02 两侧分类一致 53/54，唯一的违背是一个没过 G0 的欠训练教师（表征可分、读出欠训练）——**表征约定对欠训练的参照模型偏乐观**（部署纪律：先过 G0 再探测） |

## III. 过去"训练闭环配置优化器"一节的全文（含读数约定修订史；正式 README 只保留一段中性文字）

> 建立在两条轴上：(1) 信息容量（G0 双门作为硬约束）与 (3) 训练难度（归一化代价作为主目标），与工具本体解耦；跨两个任务的泛化已通过。它不是"预测性能"，而是"给定任务搜索代价最低的训练配置"。
>
> **训练闭环配置优化：三层奖励 + SH 多保真剪枝**。空间 = metric×width×lr（v1 为 18 个配置）→ **v2 加入教师码本 K_t 轴**（{128,256,512,1024,2048}，共 90 个配置；新增层是追加式的，旧索引逐比特不变）。L0 硬约束 = G0 双门，L1 主目标 = 到门代价，L2 廉价信号 = 早期窗口的 L_func。杠杆 = **"穷举一次 → 离线精确复放任意方案"**（由真值自检确认，25/25 逐比特一致）。**读数约定修订史（三次，每次都改变了判定）**：墙钟时间（P5 被否证：同一配置、同一份代码仅因测量窗口不同就差 4×）→ 原始到门 epoch（P5 修订）→ **归一化代价 = 到门 epoch ×(conv(w)+K·d)**（与信号/P9 统一为三方对齐）。**修正后的主要结论**：(1) **P2 从"不支持"翻转为"SUPPORT"——剪枝有价值**（在代价约定下，SH regret 3.93 对等预算随机搜索的 15.82；旧记录"剪枝没有价值"是原始 epoch 约定的产物——那个约定只看 epoch，不看每 epoch 高出的 3.83×/15× 代价）；(2) P3 的 L0 硬约束在"选择"上有贡献；(3) P4 有附带损伤；(4) P6 多保真大致持平。**信号迭代**：v1.0 acc → v1.1 到门 epoch 更优 → **v1.2 加入宽度归一化**（在代价约定下唯一 regret 为 0）。**v2/v2.3 K_t 轴（容量贪婪）**：Q1 被否证（K_t=128 层可行 9/18；真实结构是码本×度量的相互作用）；G1/R1 被否证（K_t=2048 的码本门 6.10 没有排除任何配置）；**但贪婪在 K_t=1024 处自我限制**（层内最优代价 128:7.0→256:6.1→512:4.2→**1024:3.4**→2048:5.0，触底后回升）。**v3.0 第二任务（shape64）泛化测试**：T1 核心判据通过（归一化信号跨任务 regret=0），T2 SUPPORT（1024 转折点在跨任务上复现），T4 不支持（L_func 仍只是一个被记录的量），T3 PARTIAL；另修一个数据管线 bug（种子被误当作 sigma_pix 传入）与两处 harness 修复，并记录一条方法论教训：**跨层比较"层内最优代价"时，各层的种子集必须相同**。

（完整的修订轨迹见 falsified.md 第 III 节。）

## IV. 运行产物清单（当前保留；2026-09-30 随失败实验脚本一同删除的产物已不再列出）

| 路径 | 内容 |
| --- | --- |
| `opt_demo.jsonl` + `opt_demo/` | grid64 正式数据与检查点（90 个配置 × 3 种子） |
| `opt_demo_shape64.jsonl` + `opt_demo_shape64/` | shape64 正式数据与检查点（216 个单元，v3.0 最终版） |
| `cp3_dyn*.jsonl` / `cp3_dyn_sens*.jsonl` / `cp3_multilevel*.jsonl` + `cp3*/`, `cp3_multilevel/` | CP3 四个格、L_func 与多层级剂量的原始记录 |
| `cph_0d8045d2.jsonl` / `cph_readings_0d8045d2.jsonl` / `cph/0d8045d2/` | CP-H 双轴战役（CIFAR-10，3 种子，42 个单元检查点 + 轴 2 读数） |
| `pa1.jsonl` / `pa1_report.txt` / `pa1/` | PA1 预算阶梯（P-A1-3 判定 A） |
| `pa4.jsonl` / `pa4_report.txt` / `pa4/` | PA4 剂量响应（P-A1-4 判定 SUPPORT，4 个水平 × 3 种子） |
| `moe.jsonl` / `moe_report.txt` / `moe/` | MOE 两层容量验证（route64，6 种子） |
| `gauss.jsonl` / `gauss_report.txt` / `gauss/` | GAUSS 连续瓶颈排除管线与自然效率（grid64，27 个单元 + 3 个探针；见 continuous-bottleneck.md） |
| `pw.jsonl` / `pw_report.txt` / `pw/` | PW 冻结预训练骨干作见证者（CIFAR-10，目标侧零训练；见 pretrained-witness.md） |
| `family.jsonl` / `family_report.txt` / `family/` | FAMILY 族内摊销库（CIFAR-10 类子集族，两种协议；见 family-amortisation.md） |
| `scale_c.jsonl` / `scale_c_report.txt` / `scale_c/` | SCALE-C 类数轴尺度扫描（三轮：无瓶颈参照 / 同族大 VQ / 固定天花板；见 class-count-axis.md） |
| `scale_c_clean.jsonl` / `scale_c_clean_report.txt` / `scale_c_clean/` | SCALE-C-3 固定天花板（`K_t=2048`）+ 目标缓存（含 targets_C*.pt；见 class-count-axis.md §7） |
| `wall_sat.jsonl` / `wall_sat_report.txt` / `wall_sat/` | WALL-SATURATION 蒙特卡洛真值（步骤 0）、sigma 臂、KD 预算探针，以及 CROSS-TASK 的臂行（标签 `cross`）；见 wall-saturation.md |
| `cross_task.jsonl` / `cross_task_report.txt` | CROSS-TASK 真值表（尺度等价参数化）+ P-A1-2 判定；臂行本身在 `wall_sat.jsonl` 中；见 cross-task-comparability.md |
| `scale_axes.jsonl` / `scale_axes_report.txt` / `scale_axes/` | SCALE-AXES 样本数 / 学生宽度 / 输入分辨率；见 scale-axes.md |
| `budget_dose.jsonl` / `budget_dose_report.txt` / `budget_dose/` | BUDGET-DOSE 赤字对训练预算（含 `arm="overfit_check"` 行）；见 budget-and-recipe.md |
| `budget_mechanism.jsonl` / `budget_mechanism_report.txt` / `budget_mechanism/` | BUDGET-MECHANISM 2×2 配方因子设计（LR 调度 × 死码重启）与好配方下的墙迁移；见 budget-and-recipe.md §6–§7 |
| `wall_remeasure.jsonl` / `wall_remeasure_report.txt` / `wall_remeasure/` | WALL-REMEASURE 在 cosine 退火下重定位本项目的各道墙（形状 vs 数值、cp3 复制、CP-H CIFAR-10 配方在 20 epoch 与记录的 120 epoch 上的配对探针，以及细阶梯增长检验及其种子扫描）；见 wall-remeasurement.md |
| `family_b.jsonl` / `family_b_report.txt` / `family_b/` | FAMILY-B 随机中心高斯混合（步骤 0 真值 + 四个训练配置）；见 cross-task-comparability.md §5 |
| `family_c.jsonl` / `family_c_report.txt` / `family_c/` | FAMILY-C 16 维混合（配置表 + 三个训练配置）；见 cross-task-comparability.md §6 |
| `witness.jsonl` / `witness_report.txt` / `witness/` | WITNESS 见证者阶梯（干净 / 合并类 grid64，5 种子；见 witness-strength.md） |
| `looseness.jsonl` / `looseness_report.txt` | LOOSENESS 松紧普查（38 个场景，零训练；见 bound-looseness.md） |
| `looseness_reaudit_report.txt` | LOOSENESS-REAUDIT：带机械留出的 D2' + 真值拆分（读取 `looseness.jsonl`、`wall_sat.jsonl`、`cross_task.jsonl`、`scale_c_clean.jsonl`；见 bound-looseness.md §6–§7） |
| （`scale_c_clean.jsonl` 中 `arm="budget_c1000"` 的行） | 由 `scale_c_clean_demo.py --budget-probe` 写入的 C=1000 预算后续（见 class-count-axis.md §7.4） |
| `kv.jsonl` / `kv_report.txt` / `kv/` | KV 内容可寻址记忆读法验证（联想回忆，三个臂，93 个单元；见 kv-memory.md） |
| `tf.jsonl` / `tf_report.txt` / `tf/` | TF-EXCLUSION 排除管线与残差旁路边界（TinyGPT+MNIST，45 个单元；见 transformer.md） |

（已删除的失败实验产物：CP1/CP2 的原始记录随 `cp1_demo.py`/`cp2_demo.py` 一同删除；CP-D/CP-E/CP-F/CP-G 的
`detect_all*`/`cpe*`/`cpf_*`/`cpg_*` 与 `cp3_l_info*` 随对应脚本一同删除。它们的结论归档在 falsified.md 第 I 节。）

## V. 形式化改写对照（要点）

| 原始措辞（非正式） | 正式 README 中的措辞 |
| --- | --- |
| "能力边界被**钉成**……" | 能力边界：它只判定必要条件的否证 |
| "**穷尽式检验**""**诚实的负结果**""**照实记账**""**否证**"这类战役措辞 | "已经过检验""结果是负的""已记录"这类中性事实陈述 |
| "**战役**""**判死**""**推翻**""**伪影**" | "实验系列""结论是不成立""修订" |
| 整个"2026-09-30 战役对比结果"一节 | 折入"实验验证矩阵"的中性行 |
| 优化器的"读数约定修订史（三次，每次都改变判定）" | 一段中性文字 + 指向本归档与 falsified 的指引 |
| 作者的观点（framework.md 全文） | 移入 `docs/framework.md`；该文件已从根目录删除 |
