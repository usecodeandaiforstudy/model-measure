# 约束的损失代价（损失单位）

`capacity_lens_ext.cost`

估计一个限制要付多少损失，并如实说明只有**下界**可以证明。

## API

| 函数 | 含义 |
| --- | --- |
| `constraint_cost` | point estimate of Delta = (1/2) g^T H^-1 g, computed with 30 conjugate-gradient Hessian-vector products |
| `cg_solve_hinv_g` | solves H x = g with Hessian-vector products only; cost is independent of the parameter count |
| `certify_lower` | proven lower bound ||g||^2 / (2 nu) using smoothness alone |
| `is_constrained_local_min` | checks the theorem's premise on the surviving subspace |
| `alignment_prior` | pot_E, a prior usable for RANKING only |

## 实测

| 量 | 值 |
| --- | --- |
| constraint_cost quadratic check | 1.3e-15 in closed form; 7.9e-08 in the selftest |
| small-gap accuracy | ratio 0.996, IQR 0.001 |
| CG-HVP vs full Hessian | 9.0% error at 30 iterations, 60 backward passes |
| certified lower bound | holds in 4/4 cells but is 7-48x loose |

## 能声明 / 不能声明

| 能声明 | 不能声明 |
| --- | --- |
| 给定约束的**损失**代价 | 代价**至多**是 X 的证书 |
| 代价的**可证下界** | 与比特侧的换算（价格差约 9 倍） |
| 受约束局部最优前提的检查 | 在两个约束之间做**选择**的规则 |
