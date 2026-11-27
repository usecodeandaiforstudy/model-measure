# 编码器选型：PCA 与自编码器

`capacity_lens_ext.tradeoff`

给定两条率–失真曲线，找出交叉点与自编码器的失真地板。

## API

| 函数 | 含义 |
| --- | --- |
| `codec_tradeoff` | crossover_k, winner_at, ae_floor, pca_best, ae_max_advantage_pct, recommendation |

## 实测

| 量 | 值 |
| --- | --- |
| MNIST crossover | AE wins at k=50 (+54%) and k=100 (+30%), PCA wins at k=200 (+115%) and k=400 (+1260%) |
| AE floor | 0.00296, essentially independent of k >= 100 |
| bottleneck width | h=512 moved the crossover from ~50-100 to ~100-200 and no further |

## 能声明 / 不能声明

| 能声明 | 不能声明 |
| --- | --- |
| 低于某个率时非线性编码器更小 | PCA 永远更好的保证 |
| 自编码器的失真**地板** | 能迁移到别的数据集的交叉点 |
| 操作规则：低率用 AE、高率用 PCA | 新数据集上的绝对 MSE 预测 |
