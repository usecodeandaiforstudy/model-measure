# 贡献指南

本仓库既是库，也是一份**测量记录**：`capacity_lens/` 是产品，`experiments/` 是产品每一条主张的证据，
`docs/` 是书面记录。欢迎按下面的规则贡献 —— 它们之所以存在，是因为本项目的价值来自
**可被核查的主张**，而不是看上去有用的功能。

## 开始上手

```bash
pip install -e ".[dev]"          # 库 + torchvision（实验）+ pytest
python tests/run_all.py          # 不依赖 pytest 的测试（已验证的 GPU 环境没有 pytest）
pytest tests/                    # 装了 pytest 时跑同一批测试
```

## 合并之前 CI 会检查什么

`.github/workflows/tests.yml` 跑四道关卡，全部离线（不需要 GPU、数据集或联网）：

1. **测试套件** —— 在 Python 3.9、3.11 与 3.13 上跑 `python tests/run_all.py` 与 `pytest tests/`；
2. **每个脚本的 `--selftest`** —— 预注册判据就住在实验脚本里，所以它们会被执行；
3. **证据** —— `python tools/reproduce_all.py --check` 从记录的 jsonl 重建全部已发布判定，只要有一条重建不出来、
   或与已提交快照不一致就失败。**如果你改了判据却没有记录这次改动，正是这一关抓住它**（报告文件是追加式的，
   所以旧判定仍然与新判定并排可见）；
4. **发行包** —— 构建 wheel、跑 `python tools/check_wheel.py`（wheel 必须只装库、不夹带其它东西；
   跑它的 `--selftest` 看它"能说不"）、再跑 `examples/quickstart.py`。

实验脚本从检出目录运行（`python experiments/<name>_demo.py`），数据集需要手工放到 `experiments/data/` 下 ——
见 README 的数据布局一节。**本项目从不下载数据、从不访问网络**，每个脚本都必须保持离线可用。

## 一次改动必须满足的规则

1. **判据先于数据。** 任何新实验都在**运行之前**把预注册写进模块 docstring：定义、网格、判据、判定规则。
   看到结果之后再改这些内容，只有作为**明确记录的修订**才可以接受（见 `docs/class-count-axis.md`
   §6.1 与 §7，那里有两处更正被留在记录里，而不是被藏起来）。
2. **判据是纯函数，并且被测到。** 阈值与判定拼装住在小函数里：吃进普通数据、吐出普通结果，
   因此无需训练就能单测；每个脚本通过 `--selftest` 暴露它们，跨脚本的那些由
   `tests/test_experiment_criteria.py` 覆盖。
3. **不许重掷种子来过门。** 某个种子没过门，就按失败报告。这里没有"跑到变绿为止" —— 那是噪声选择，
   而这正是 G0 门存在的目的所在。
4. **负结果原样归档。** 被证伪的预期也是结果：它连同机制写进笔记、写进 `docs/falsified.md`，
   并作为新条目写进台账补丁。
5. **接力模式。** 每个工作单元都按（配置、种子、K）缓存在
   `experiments/data/<campaign>/<fingerprint>/` 下，而指纹只覆盖**基础**网格，所以扩展运行会复用已缓存的
   基础单元。长时间运行必须能扛住中断，离线重判（`--verdict-only`）必须不重训就复现已发布的数字。
6. **代码与输出只用英文。** 标识符、docstring、注释、`print` 输出与异常消息。
   （`docs/` 里每份文档都有中英两版：`<name>.md` 与 `<name>_cn.md`。）
7. **状态标签是强制项。** 公开函数的 docstring 里必须带 `VERIFIED` / `PARTIALLY VERIFIED` /
   `CANDIDATE`，并把实测边界写清楚。**不要发布方向尚未验证的"判定形状"API** ——
   `capacity_lens/monitor.py` 记录了这个先例：它的 `H(Z) > R_IB ⇒ overfit` 标志是被删除，
   而不是留在原处再加个警告，因为调用者分不清启发式与规则。
8. **不编造元数据。** 不编造仓库 URL、DOI、作者名单或单位。字段未知就留空。

## 新增一个实验
命名成 `experiments/<topic>_demo.py`，并给它标准形状：

```
docstring      — the pre-registration (design, grids, criteria, verdict rule, iron rules)
pure functions — the criteria and the verdict assembly, unit-testable
--selftest     — runs those unit tests with fixture data, no training
--quick        — a smoke configuration (tiny grids) for wiring checks
--fresh        — recompute instead of the default relay mode
--verdict-only — rebuild the verdict from the recorded jsonl, zero training
artifacts      — <topic>.jsonl, <topic>_report.txt, <topic>/<fingerprint>/ checkpoints
```

**两代脚本，以及为什么更早的那些保持原样。** 最早的九个实验
（`cifar_demo.py`、`cp3_demo.py`、`cp3_dyn.py`、`cp3_dyn_sens.py`、`cp3_multilevel.py`、`cph_post.py`、
`mnist_demo.py`、`opt_demo.py`、`transformer_demo.py`）早于 `--selftest` / `--verdict-only` / 接力模式。
它们被**刻意保持不动**：它们记录的产物（jsonl、报告、检查点）正是这批代码产生的，改写它们会让数字与
产生数字的代码脱钩 —— 而这正是本项目拒绝做的那一件事。把它们当作历史记录来读，只在复现它们自己的产物时
运行它们，并且给每个新脚本加上上面那套完整形状。一次收尾审计（2026-10-05）核实：后面那二十三个脚本
全都暴露了离线路径。新增一个之后跑 `python tools/reproduce_all.py`：它会从记录的 jsonl 重放每一条已发布
判定，并告诉你你那条能否干净地重判（以及它是否需要一次导入期写入 —— 它绝不能需要）。

然后写 `docs/<topic>.md`（小写 kebab-case：设计、结果、判定、意味着什么、诚实的边界）及其 `_cn` 孪生版，并往
`docs/ledger.md` 加一行台账、往 README 的验证矩阵加一行。就本项目而言，
一个既没有笔记、也没有矩阵行的新战役等于不存在。

## 文档布局

- `README.md` —— 唯一的入口：能力、边界、快速开始、验证矩阵。那里的每一条主张都必须能追溯到一份笔记
  或一条台账。
- `docs/*.md` —— 每个战役一份文档（小写 kebab-case）：测量结果、判定与诚实的边界。
- `docs/framework.md` —— 存活下来的理论框架与立场。
- `docs/verification-plan.md`、`docs/ledger.md` —— 预注册的判据，以及索引每个战役的 33 行判定台账。
本地工作材料（翻译前快照、已退休的过程文档、旧运行转储）放在 `reference/` 下，该目录已被 gitignore，从不提交。
- `artifacts.md`（在 `docs/` 下）—— 运行产物清单，也是披露已发布证据中任何异常之处的地方
  （有两处这样的披露记录在那里，而不是在数据里被"修好"）。

**每份文档都以两种语言发布。** 中文版文件名 = 英文名 + **`_cn`**
（`README.md` → `README_cn.md`，`docs/transformer.md` → `docs/transformer_cn.md`），
而且两者必须互为**孪生版**：同标题结构、同数字、同文件名引用、同代码块。数字是本项目的主张，
所以一个数字在翻译里漂移，这份翻译就是在说谎 —— 这正是
`python tools/check_bilingual.py` 机械校验的正是这些内容（CI 也会跑它）。**有意豁免**，
因为它们的规范文件名被工具读取、且内容不是散文：`LICENSE`、`CITATION.cff`、
`.gitignore`、`pyproject.toml`，以及记录证据（`*.jsonl`、`*_report.txt`）。

**两段历史、两个文件，以及 CHANGELOG 为什么留在根目录。** `CHANGELOG.md` 是**库的**历史
（API 的增加/删除/修复，每次改动都带原因和它背后的测量），它和任何包的 CHANGELOG 一样属于仓库根目录：
读者、评审者和发布工具都在那里找它。**研究的**历史 —— 判定、被取代的读法、判据缺陷 ——
住在 `ledger.md` 与各份笔记里，永远不会被并入 CHANGELOG。把这两者分开是刻意的；
改动其中一个不等于改动另一个。预发布里程碑写进 CHANGELOG 没有问题，尽管什么都还没发布：
它记录的是包在各个版本上做了什么，而不是什么时候发布的。

## 代码风格

只用纯 PyTorch，不引入它之外的框架；导入期没有全局状态（日志在入口点内部惰性配置，所以模块保持可导入、
便于测试）；不做任何下载；凡是要记录结果的地方都用确定性种子。函数保持小，数学留在拥有它的那个模块里
（容量公式放 `capacity.py`，判定规则放 `preflight.py`），并且宁可用显式的 `None` 加一句注释，
也不要静默回退。
