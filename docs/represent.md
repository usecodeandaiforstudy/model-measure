# Subset representativeness (rho')

`capacity_lens_ext.represent`

Flags a data subset that looks **unlike** the full task. It answers "unlike", never "better".

## API

| function | meaning |
| --- | --- |
| `calibrate_threshold` | `tau = max(1.5 * q95(baseline), 2.0)`; reports whether the baseline is large enough |
| `subset_rho_prime` | `rho' = align / median(baseline)`, dimensionless |
| `align_report` | per-subset verdict plus the mandatory `validity.window_steps` |

## Measured

| quantity | value |
| --- | --- |
| threshold plateau | lower 1.30-2.61, upper 12.4-15.9 across three datasets |
| stability | CV 8.4% / 8.1% / 3.9% with enough baselines; 36% at 400 samples per arm |
| verdicts | **9/9** correct on the MNIST report |
| validity window | rank correlation 0.900 to ~100 steps, 0.600 by 1000 |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| a flag that a subset is **unlike** the full task | a ranking that selects the best subset |
| a budget-qualified statement | an accuracy prediction for a subset |
| a dimensionless score comparable across tasks | a threshold that transfers without a baseline |
