# Memorisation capacity with four reproducibility rules

`capacity_lens_ext.capacity_probe`

Training is injected as a callback, so the module is pure and exactly testable.

## API

| function | meaning |
| --- | --- |
| `p_fit` | fraction of independent label draws fitted at size n; fewer than 5 draws raises |
| `bisect_threshold` | the largest grid size that still passes (no interpolation) |
| `measure_capacity` | capacity = the **saturated** threshold; returns `saturated`, `lower_bound_only`, `warning` |

## Measured

| quantity | value |
| --- | --- |
| capacity on MNIST MLPs | ~**2x** parameter count (2.12 / 1.88 / 1.79 points per parameter) |
| scaling | capacity ~ `p^0.88` |
| depth | no effect on the coefficient (1.86 vs 1.88) |
| saturation | B and 2B agree at B=4000 for h=8 (225 vs 225) |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| a capacity number, **only when saturated** | a capacity number from an unsaturated run |
| the budget and draw count with every call | a capacity comparable without them |
| a **lower bound** when saturation fails | a guarantee that the architecture can fit k points |
