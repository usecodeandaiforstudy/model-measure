# 记忆容量与四条可复现规则

`capacity_lens_ext.capacity_probe`

测一个架构能拟合多少个随机点，并拒绝把未饱和的数字叫容量。

## API

| 函数 | 含义 |
| --- | --- |
| `p_fit` | fraction of independent label draws fitted at size n; fewer than 5 draws raises |
| `bisect_threshold` | largest grid size that still passes (no interpolation) |
| `measure_capacity` | capacity = the SATURATED threshold; returns saturated / lower_bound_only / warning |

## 实测

| 量 | 值 |
| --- | --- |
| capacity on MNIST MLPs | ~2x the parameter count: 2.12 / 1.88 / 1.79 points per parameter |
| scaling | capacity ~ p^0.88 |
| depth | does not change the coefficient (1.86 vs 1.88) |
| saturation | B and 2B agree at B=4000 for h=8 (225 vs 225) |

## 能声明 / 不能声明

| 能声明 | 不能声明 |
| --- | --- |
| **仅在已饱和时**给出容量 | 未饱和运行得到的容量数字 |
| 每次调用都带预算与抽样数 | 不需要这些就**可比**的容量 |
| 饱和失败时给出**下界** | 该架构一定能拟合 k 个点的保证 |
