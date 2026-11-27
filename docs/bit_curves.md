# Bit-side curves: activation bits and pruning cliff

`capacity_lens_ext.bit_curves`

## API

| function | meaning |
| --- | --- |
| `activation_curve` | `Delta(b)` and `saturated_at` = the smallest b after which **every** larger b passes |
| `pruning_curve` | `cliff_k` = first k whose cost grows by at least `cliff_ratio` over the previous k; `safe_k` before it |

## Measured

| quantity | value |
| --- | --- |
| activation saturation | ~**4 bits per dimension** (0.0809, 0.0221, 0.0076, then -0.0009) |
| weight saturation | ~6 bits per parameter |
| structured pruning | k=1..3 of 6 units costs +0.003..+0.021; k=4 jumps to **+0.264** (`safe_k = 3`) |
| non-monotonicity | the real activation curve dips by 1e-3 at b>=4, so saturation is defined conservatively |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| the bit width beyond which more bits are wasted | a guarantee that the loss curve is monotone |
| the pruning count before the cliff | a prediction of where the cliff is for an unseen architecture |
| a lowest-cost operating point | an accuracy guarantee after pruning |
