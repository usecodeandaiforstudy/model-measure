# Where Delta comes from: selection and approximation on the zero-loss manifold

Conceptual frame behind the per-batch Delta in `batch_delta`. Everything below was measured in this repository;
the falsified claim is kept in the record on purpose.

## 1. Definition

A train/val split is itself a task extension: the fitted set is the small task, fitted + probe is the big one.

$$\Delta = L_{probe}(\theta^{\ast}_{ERM}) - \min_{\theta} L_{probe}(\theta)$$

| role in Delta | in a train/val split |
| --- | --- |
| small task | the fitted set |
| big task | fitted set + probe set |
| $\theta^{\ast}_{S}$ | the ERM solution: global optimum on the fitted set |
| $\theta^{\ast}_{B}$ | the oracle: a solution that has seen the probe set |

## 2. The constraint is a manifold, not a subspace

$$M_{fit} = \{\theta : L_{fit}(\theta) = \min L_{fit}\}$$

$$\theta^{\ast}_{ERM} \in M_{fit} \quad \text{but} \quad \theta^{\ast}_{ERM} \neq \arg\min_{\theta \in M_{fit}} L_{probe}(\theta)$$

A linear subspace is only a tangent approximation of this set. This is the object the implicit-regularisation
literature studies: where SGD stops on the zero-loss manifold.

## 3. The decomposition (an identity)

$$\Delta = \Delta_{sel} + \Delta_{approx}$$

$$\Delta_{sel} = L_{probe}(\theta_{ERM}) - \min_{\theta \in M_{fit}} L_{probe}(\theta)$$

$$\Delta_{approx} = \min_{\theta \in M_{fit}} L_{probe}(\theta) - \min_{\theta} L_{probe}(\theta)$$

The sum is algebraically exact, so the decomposition cannot be wrong. The only falsifiable claim is that the two
terms respond to their own intervention axis -- and that claim is refuted below.

## 4. Measured: the independence claim is REFUTED

45 runs (3 selection settings x 3 widths x 5 seeds), mean +- 95% CI, t(0.975, 4) = 2.776.

| selection | sel h=8 | sel h=64 | sel h=256 | approx h=8 | approx h=64 | approx h=256 |
| --- | --- | --- | --- | --- | --- | --- |
| plain | +0.543 +- 0.059 | +0.487 +- 0.015 | +0.605 +- 0.013 | **+0.209 +- 0.016** | +0.0025 +- 0.0001 | +0.0008 +- 0.0000 |
| noise aug | **+0.121 +- 0.045** | +0.400 +- 0.016 | +0.441 +- 0.012 | **+0.422 +- 0.029** | +0.0198 +- 0.0056 | +0.0039 +- 0.0037 |
| label smoothing | **+0.162 +- 0.013** | +0.317 +- 0.018 | +0.332 +- 0.011 | **+0.444 +- 0.026** | +0.0732 +- 0.0034 | +0.0186 +- 0.0031 |

Of 18 pairwise comparisons between the two axes, **1 is clean**. Selection-side interventions also move
Delta_approx (h=8, noise aug: sel -0.42, approx +0.21); widening also moves Delta_sel (64 to 256: approx -0.002,
sel +0.118).

**One explanation fits all of them:** the three interventions act on the same underlying quantity, the effective
capacity. Smoothing the loss shrinks the manifold (the probe set can no longer be memorised), which raises the
approximation term; widening grows it, which raises the selection term. The two terms are two readings of one
axis, not independent components.

## 5. Probe-measured Delta overstates the realisable gain by 4-5x

The same best point on the manifold, evaluated on the TEST set:

| selection | h=8 | h=64 | h=256 |
| --- | --- | --- | --- |
| plain | **-0.014 +- 0.030** (indistinguishable from 0) | +0.061 +- 0.022 | +0.121 +- 0.011 |
| noise aug | +0.062 +- 0.079 | +0.090 +- 0.032 | +0.094 +- 0.025 |
| label smoothing | +0.054 +- 0.015 | +0.131 +- 0.012 | +0.122 +- 0.015 |

Probe says 0.32-0.60, test says 0.06-0.13: about 80% of the apparent gain is memorising the probe set. The sign is
right (significantly positive at h=64/256), so it is an overstatement, not an artefact.

## 6. Can claim / cannot claim

| can claim | cannot claim |
| --- | --- |
| Delta splits into a selection and an approximation term (an identity) | the two terms are independent, or that each responds only to its own intervention (refuted, 1/18) |
| capacity strongly reduces the approximation term (-0.21 to -0.43) | increasing capacity leaves the selection term alone |
| selection-side interventions reduce the selection term (-0.09 to -0.42) | they do so without raising the approximation term |
| probe-measured Delta overstates the realisable gain by 4-5x | it can be read as an expected test improvement |
| a linear subspace is only a tangent approximation of the manifold | the manifold framing is validated on CNNs or at scale (untested) |

## 7. Scope

MNIST, MLP widths 8/64/256, 2000/2000/2000 splits, 5 seeds. No CNN, no real-scale model, no second dataset. The
manifold constraint is a penalty approximation (band epsilon 0.1, lambda 50), not an exact projection.
