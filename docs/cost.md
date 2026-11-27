# Constraint cost in LOSS units

`capacity_lens_ext.cost`

Estimates what a restriction costs, in **loss**, and states honestly that only a **lower** bound can be proven.

## API

| function | meaning |
| --- | --- |
| `constraint_cost` | point estimate of `Delta = (1/2) g^T H^-1 g`, via 30 conjugate-gradient Hessian-vector products |
| `cg_solve_hinv_g` | solves `H x = g` with Hessian-vector products only; cost independent of the parameter count |
| `certify_lower` | proven lower bound `||g||^2 / (2 nu)`, using smoothness alone |
| `is_constrained_local_min` | checks the theorem's premise on the surviving subspace |
| `alignment_prior` | `pot_E`, a prior usable for **ranking only** |

## Measured

| quantity | value |
| --- | --- |
| `constraint_cost` on a quadratic | `1.3e-15` closed form; `7.9e-08` in `selftest` |
| small-gap accuracy | ratio **0.996**, IQR 0.001 |
| CG-HVP(30) vs full Hessian | **9.0%**, 60 backward passes |
| certified lower bound | holds **4/4**, but **7-48x loose** |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| a cost in **loss** units for a stated constraint | a certificate that the cost is **at most** X |
| a **proven lower bound** on the cost | a conversion to bits (prices differ ~9x) |
| a check of the constrained-local-minimum premise | a rule that **selects** between two constraints |
