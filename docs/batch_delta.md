# Per-batch Delta: distribution, stratified selection and an online training meter

`capacity_lens.batch_delta`

Per-batch cost accounting for data work and a training-time meter. Pure functions: the caller supplies measured
losses and gradients; nothing here trains or reads files.

## API

| function | meaning |
| --- | --- |
| `per_batch_delta` | `Delta(b) = L_probe(theta_b) - L_probe(theta_full)` for every batch |
| `delta_summary` | mean, sd, coefficient of variation, extreme batches |
| `rank_batches` / `select_batches` | ordering and `high`/`low`/`middle`/`random` strata (deterministic) |
| `online_step_delta` | online per-batch Delta: `-lr * <g_probe, g_b>`, one extra backward pass per step |
| `track_probe` | spread across batches and how well the epoch means follow the probe loss |

## Measured

| quantity | value |
| --- | --- |
| per-batch Delta distribution (40 batches of 100) | mean **+0.4255**, sd 0.0905, **CV 0.213**, identical across runs |
| correlation with class entropy | **-0.523** (more balanced batches cost less) |
| correlation with the batch's own fit loss | -0.061 (fitting well says nothing about Delta) |
| stratified selection (equal size, equal entropy) | high **0.8803** vs low 0.8697 (**+0.0107**) |
| within-class, fixed 200 per class | hard **0.8952** vs easy 0.8400 (**+0.0552**) |
| Delta-weighted sampling | **0.8712**, worse than random 0.8752 |
| online first-order vs exact one-step | correlation **+1.000** |
| online epoch mean vs real probe loss | Spearman **-0.969** |
| online spread across batches | sd/abs(mean) **0.10** |
| online as a corrupt-batch detector, loss-matched | AUC **0.641 -> 0.491** while the batch loss goes 0.611 -> **0.847** |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| per-batch Delta varies and is reproducible | large Delta guarantees better generalisation |
| rank/stratify data by Delta and gain points | use Delta as a sampling weight (it loses to random) |
| the online value faithfully reports what a batch does to the probe set | the online value detects bad data better than the batch loss |
| spread across batches is measurable (10%) | the signal keeps working late in training (it dies by epoch 20) |
| probe-measured Delta cannot be read as an expected test improvement (it overstates by 4-5x) |
