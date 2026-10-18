# 更新日志（CHANGELOG）

`capacity_lens` 包的所有值得一提的变更。版本号只在源码树内递增；**尚未发布到 PyPI**，
所以请把版本号当作开发里程碑，而不是发行版。

## 0.7.0 — 2026-10-15

### Added
- `capacity_lens/compress.py`：压缩预算顾问。**可证**一侧：`task_info_floor`（Fano，需传经验 `h_y`，
  并在"均衡类别默认值不成立"时由 `balance_warning` 报警）、`capacity_floors`（码本 / 容量 / 每维比特下界）、
  `veto`（**只是必要条件**的 `capacity < R_IB` 否决门）、以及 `tightness_interval` —— 它把真实的任务相关松紧夹在
  `I_lb / R_IB` 与 `H(y) / R_IB` 之间，**已登记：MNIST ≤1.253×、CIFAR-10 ≤1.917×**，即在匹配单位下证书**可证是紧的**。
  **实测**一侧（零训练）：`partition_retention`（硬量化在 K=64 时只保留 `I(z;y)` 的 **0.15–0.75**，且始终低于 1）与
  `partition_ceiling`（**松 2–3×**，只在方向可靠）。
- `python -m capacity_lens.compress`：把可证报告做成命令行（含 `--selftest`）；实测部分需要训练好的模型，
  因此留在实验侧驱动脚本里。
- `docs/compression-advisor.md`（含 `_cn` 孪生）：说明本模块的证据分级 ——
  **【可证】/【实测】/【归档】/【估计·松】** —— 以及它拒绝掩盖的边界（地板比实用拐点低 10–30×；
  秩饥饿不可逆、位宽饥饿可被 QAT 找回；只能用验证集上的整网评估来分配；每堵墙都是协议相对的）。
- `tests/test_compress_advisor.py`：把已登记数值钉住（`R_IB` 1.8367 / 1.2009、两条区间、否决门边界、
  MNIST 上退化的比特下界），使算术上的变化在 CI 里暴露，而不是在报告里。

## 0.6.1 — 2026-10-05

### Added
- `k_wall_band(k_list, accs, acc_ref, rel_tols=(0.05, 0.10, 0.15))`：在多个保留率容差下给出实测墙。
  墙是一次阈值穿越，所以在平坦的落差曲线上，它的取值有一部分来自判据而不是容量 —— 实测：保留率差 2 个点，
  就把定位到的墙从 2048 挪到 512（`docs/scale-axes.md`，A2）。`preflight_report` 现在返回
  该容差带（`wall_band`），在 `items` 里暴露它（`wall_band`、`wall_band_wide`），并在容差带摆动达 4 倍及以上时
  把它写进判定，因为"把它当区间引用、而不是当数字"才是一个对敏感性敏感的回答的诚实形式。

## 0.6.0 — 2026-09-30

### Added
- `excluded_set`、`coverage`、`false_exclusions` 现在属于这个包。它们是产品的头号规则
  （`log K < R_IB` ⇒ 该瓶颈可证地装不下这个任务），此前只能从实验脚本里拿到；
  一条作为*产品*发布的规则就该住在库里。
- `pyproject.toml`：库可以 pip 安装（`pip install -e .`），版本从 `capacity_lens.__version__` 动态读取，
  构建出的 wheel **只**含 `capacity_lens/` 加许可证 —— `experiments/`（证据）与 `reference/`（文档）
  留在仓库里。
- `tests/`：71 条用例，覆盖库的实现不变量与已发布规则，外加实验脚本的预注册判据函数。
  用 pytest **或** `python tests/run_all.py` 都能跑（已验证的 GPU 环境有 torch 但没有 pytest）。

### Removed
- `TrainMonitor` 的 `overfit` 标志（`H(Z) > R_IB`）与 `step()` 的 `task_mi_estimate` 参数。
  这个方向不只是未验证，而是不成立：`H(Z)` 是 `I(X;T)` 的表征熵**上**界，会系统性高估任务信息
  （实测 5.36 nats vs 任务真值 4.16），所以这个比较是在估计器自身的偏差上触发的。容量 vs 任务的问题
  改由 Fano 排除方向来判定。一个未验证的判定形状的 API 比没有 API 更糟。

### Fixed
- `GaussBottleneck` 现在是一个真正的信道：表征被归一化到单位 RMS（于是 SNR、从而标称容量都有定义），
  噪声在**训练与评估时**都生效（更早的版本只在训练时加噪声，也就是说评估时它根本不是信道），
  `capacity()` 报告 Shannon 界，`forward` 遵循包的 `(Z, idx)` 约定，此时 `idx = None`。
  需要逐维方差界时可用 `normalize="per_dim"` 模式。
- `evaluate` 能容忍不上报离散码的瓶颈（`info = [None]`）：写成 `return self.fc2(z), [idx]` 的模型，
  现在配 `VQBottleneck` 或 `GaussBottleneck` 都能用。
- `TrainMonitor.summary()` 把它给出的量标注为 **nats**（括号里附比特等价值）；更早的版本在自然对数值上
  印的是"bits"。

## 0.5.0 — 2026-09-30

### Added
- `measured_wall(k_list, accs, acc_ref, rel_tol)`：实测墙，即学生保留参照精度 (1 − rel_tol) 的最小 `K`
  —— 部署定尺寸决策应该用的那个数字。
- `looseness(rib, k_wall) = ln(k_wall) / rib`：可证界比墙低多少。
- `preflight_report(..., r_ib=...)` 现在还返回 `k_wall` 与 `looseness`，新增 `practical_wall` 条目，
  并把墙追加到判定字符串里。

### Changed
- `kv_storage_capacity` 从 `CANDIDATE` 改为 `PARTIALLY VERIFIED`：存储层与逐 token 流量机制
  已在联想回忆上验证，同时写明实测边界 —— 该公式是一个乐观地板（常数高 1.5–4×，
  且倍数随序列长度增长）。
- `gauss_capacity` 记为端到端已验证，**并附**可用性边界（一个严格上界，在实测任务上高估训练后的编码器
  实际用到的量约 5×）。

## 更早的版本（0.1 – 0.4）

当时没有保留 changelog。到 0.4.0 时，包里已经含有今天这十个模块：VQ 与高斯瓶颈、容量公式与 min/sum
接线界、两层 `conditional_capacity` 读法、量化与蒸馏筛查、Fisher 诊断、几何探针、带 Fano 反解的
训练/评估胶水，以及 preflight 检查表。
