# Encoder choice: PCA versus autoencoder

`capacity_lens_ext.tradeoff`

Given two rate-distortion curves, locates the crossover and the autoencoder's distortion floor.

## API

| function | meaning |
| --- | --- |
| `codec_tradeoff` | `crossover_k`, `winner_at`, `ae_floor`, `pca_best`, `ae_max_advantage_pct`, `recommendation` |

## Measured

| quantity | value |
| --- | --- |
| MNIST crossover | AE wins at k=50 (**+54%**) and k=100 (**+30%**); PCA wins at k=200 (+115%) and k=400 (+1260%) |
| AE floor | **0.00296**, essentially independent of k >= 100 |
| bottleneck width | h=512 moved the crossover from ~50-100 to ~100-200, and no further |

## Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| the rate below which a nonlinear encoder is smaller | a guarantee that PCA is always better |
| the autoencoder's distortion floor | a crossover that transfers to another dataset |
| an operational rule: AE at low rate, PCA at high rate | an absolute MSE prediction on a new dataset |
