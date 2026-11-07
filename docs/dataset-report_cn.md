# 数据集报告

`capacity_dataset` 对分类数据集测三件事并给出一条建议。它只调用已发布的 `capacity_lens` API（不复制任何公式），
除 torch 外不需要别的依赖。

| 区块 | 报告什么 | 成本 |
| --- | --- | --- |
| 重复报告 | 精确重复率、标签冲突、H(y)、R_IB、K_min | 秒级，无需训练 |
| 饱和点 | level、参数 knee、样本 knee | 一次短训练 |
| 复核报价 | 每抓一个错标要读几行（需要真值） | 一次训练 |

一条建议由固定规则表给出：`DEDUP FIRST`、`TOO EASY`、`USABLE`、`UNREACHABLE`。

## 它不做什么

- 不做自动清洗：删掉高损失行实测 +0.0010 精度，因此不提供；
- 不做标签噪声检测：三种方法都在自己的零假设对照下失败；
- 不给单一难度分：knee 测的是"多快饱和"，不是"能到多高"；
- 没有真值就不给 precision/recall：无真值时 `review_price` 返回 `None`。

## 编解码归因

`codec_report(enc, dec, X, y, reference_acc=...)` 返回 `LOSS IN ENCODER`、`LOSS IN DECODER` 或 `NEITHER`。
参照必须在原始输入上实测；由探针下界反推的精度天花板是无效的（一次实测中预测 0.500、实际 0.643）。
没有参照时判定为 `UNDECIDED`。

## 演示

    python -m capacity_dataset.demo

在带已知缺陷的合成数据上运行，并在当前目录写出 `demo_report.md`。
