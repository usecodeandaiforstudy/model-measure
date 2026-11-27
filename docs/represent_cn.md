# 子集代表度（rho'）

`capacity_lens_ext.represent`

标出**不像**全集的子集；它回答"不像"，从不回答"更好"。

## API

| 函数 | 含义 |
| --- | --- |
| `calibrate_threshold` | tau = max(1.5 * q95(baseline), 2.0); reports whether the baseline is large enough |
| `subset_rho_prime` | rho' = align / median(baseline), dimensionless |
| `align_report` | per-subset verdict plus the mandatory validity.window_steps |

## 实测

| 量 | 值 |
| --- | --- |
| threshold plateau | lower bound 1.30-2.61, upper bound 12.4-15.9 across three datasets |
| stability | CV 8.4% / 8.1% / 3.9% with enough baseline subsets; 36% at 400 samples per arm |
| verdicts | 9/9 correct on the MNIST report |
| validity window | rank correlation with post-retraining loss 0.900 to ~100 steps, 0.600 by 1000 |

## 能声明 / 不能声明

| 能声明 | 不能声明 |
| --- | --- |
| 某子集**不像**全集 | 挑出最好子集的排序 |
| 带预算限定的陈述 | 某子集的精度预测 |
| 跨任务可比的**无量纲**分数 | 没有基线也能迁移的阈值 |
