# 比特侧曲线：激活比特与剪枝断崖

`capacity_lens_ext.bit_curves`

找出激活量化不再损失精度的比特宽度，以及剪枝代价突增的点。

## API

| 函数 | 含义 |
| --- | --- |
| `activation_curve` | Delta(b) and saturated_at = the smallest b after which every larger b passes |
| `pruning_curve` | cliff_k = first k whose cost grows by at least cliff_ratio over the previous k; safe_k before it |

## 实测

| 量 | 值 |
| --- | --- |
| activation saturation | ~4 bits per dimension (0.0809, 0.0221, 0.0076, then -0.0009) |
| weight saturation | ~6 bits per parameter |
| structured pruning | k=1..3 of 6 units costs +0.003..+0.021; k=4 jumps to +0.264 (safe_k = 3) |
| non-monotonicity | the real activation curve dips by 1e-3 at b>=4, hence saturation is defined conservatively |

## 能声明 / 不能声明

| 能声明 | 不能声明 |
| --- | --- |
| 再多比特也白费的宽度 | 损失曲线**单调**的保证 |
| 断崖**之前**的剪枝数量 | 未见过架构的断崖位置 |
| 最低代价的工作点 | 剪枝后的精度保证 |
