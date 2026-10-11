# Continuous bottleneck note (continuous-bottleneck)

> Record date: 2026-09-30　Experiment: `experiments/gauss_demo.py` (pre-registered criteria in the script docstring) + `--probe 600`
> Purpose: generalize the **exclusion pipeline** from discrete VQ to a **continuous (additive Gaussian noise) bottleneck**——this is the last
> capacity-side path in the package never verified end to end, and the counterpart of the "two-tier capacity reading" on a real network's continuous intermediate layer.
> 27-unit main run (3 seeds) + 3-unit targeted probe; the VQ side **zero training**, reusing the cp3 cache (same task, same teacher, same budget, same seed).

---

## 1. Design and pre-registration

Task/model: the grid64 three-dimensional task (`n_per_class=150, sigma=0.5`), with the shape `GridMLP` (3→64 → bottleneck(64) → 64);
budget 120 epoch / Adam 1e-2 / KD(CE+KL T=3)——exactly the same as the cp3 MLP grid, **the only variable = the bottleneck type**.

| Arm | Bottleneck | Capacity accounting |
| --- | --- | --- |
| **VQ** (cached, zero training) | `VQBottleneck(64, K)`, K ∈ the cp3 calibration grid | **exact**: `log K` |
| **GAUSS** (newly trained) | `GaussBottleneck(64, σ)` (unit-RMS normalization + noise added throughout training/evaluation) | `C_rig = (d/2)ln(1+λ̄/σ²)`, λ̄ = the **mean** per-dimension variance of the normalized signal |

**Every measured point yields a provable "capacity sandwich"**: `R_IB(ε_measured) ≤ I(X;Z) ≤ C_rig(σ)` (the lower bound from the project's own Fano inversion, the upper bound from Jensen). Capacity target grid `[1,2,3,4,5,6,8,12,16,24]` nats + the noiseless ceiling.

## 2. Results

**Curves (retention rate = acc/acc_teacher; three seeds)**

| C_nom (nats) | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 24 | noiseless |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 0 | 0.10 | 0.18 | 0.32 | 0.43 | 0.47 | 0.48 | 0.65 | 0.80 | 0.86 | **0.92** | **1.000** |
| seed 1 | 0.09 | 0.17 | 0.29 | 0.36 | 0.49 | 0.44 | 0.66 | 0.80 | 0.79 | **0.91** | **0.997** |
| seed 2 | 0.08 | 0.19 | 0.26 | 0.41 | 0.42 | 0.47 | 0.53 | 0.81 | 0.88 | **0.91** | **0.992** |

**Controls (same task, same protocol)**: teacher acc 0.974/0.983/0.986 → `R_IB` 3.93/4.00/4.03 nats, `K_min` 52/55/57;
**VQ wall = 4.56 nats (K=96)**, consistent across the three seeds; VQ side `looseness` **1.13–1.16**.

**Continuous wall**: σ* = 0.95 → `C_rig = 24.0` nats (same point for all three seeds), retention 0.909–0.922.
⇒ **To reach the same 90% retention, the noisy channel needs ~24 nats while the discrete codebook needs only 4.56 nats: a 5.3× gap** (`looseness` on the `C_nom` side = **5.95–6.10**, whereas the VQ side is 1.13–1.16).

**Targeted probe (`--probe 600`, 5× budget, boundary point `C_nom = 4` nats)**

| Seed | 120 ep retention | 600 ep retention | Δ |
| --- | --- | --- | --- |
| 0 | 0.425 | 0.365 | **−0.060** |
| 1 | 0.359 | 0.376 | +0.017 |
| 2 | 0.411 | 0.403 | −0.008 |

**0/3 read as "insufficient optimization"** (all Δ within ±0.06, and two seeds slightly decreased) ⇒ that gap **is structural under this protocol**.

## 3. Verdict: **SUPPORT** (after the bound correction, C0/C1/C2/C3 each 3/3)

| Criterion | Result |
| --- | --- |
| **C0** arm self-check (noiseless ceiling ≥0.95 retention) | **PASS** (1.000 / 0.997 / 0.992) |
| **C1** upper-bound consistency (per point `R_IB(ε) ≤ C_rig`) | **PASS** 3/3 —— the rigorous upper bound was never violated by a measured student |
| **C2** certified exclusion transfers (`C_rig < R_IB` ⇒ retention ≤0.75) | **PASS** 3/3, **non-vacuous** (3/4/4 points; measured retention ≤0.32) |
| **C3** wall located + sandwich contains the discrete wall | **PASS** 3/3 (`[R_IB(ε*), C_rig]` = [3.4, 24] ∋ 4.56) |
| **C4** (report only) `C_nom(wall)` vs `log K_wall` | FAIL 3/3 (24 vs 4.56, 5.3×) |

## 4. Three conclusions

1. **The exclusion pipeline holds overall on a continuous channel** (first end-to-end evidence): the rigorous upper bound was never violated, the certified exclusion is non-vacuous, and the continuous wall's sandwich contains the discrete arm's exact wall——**two independent paths give mutually consistent accounts of "how much capacity this task needs"**.
2. **But the continuous channel's "usable capacity" is systematically overestimated, and this is the largest discount measured in this project**: to reach the same 90% retention, the noisy channel needs ~24 nats vs 4.56 nats for the discrete codebook (**5.3×**); at the certification boundary (4 nats, where the channel **could** carry 4.0 nats ≥ R_IB 3.9) the trained encoder only uses a retention of 0.42.
3. **That discount is not "not trained enough"**: at 5× budget the retention is unchanged (0/3 optimization-limited). Combined with KV's zero-slack regime (VQ needs 1.5–4× more codewords) and LOOSENESS (38/38 scenarios `K_wall > K_min`), this is **the third independent instance of "nominal capacity ≠ usable capacity"**, and the largest in magnitude.

**Engineering implication**: if a capacity bottleneck must be inserted, **a discrete codebook is far more cost-effective than a Gaussian channel of the same nominal capacity** (5.3× on this task)——contrary to the intuition that "continuous is more expressive", because a deterministic discrete task wants **exact code identity**, while a noisy channel must continuously fight the noise.

## 5. Methodological accounting (important)

**The first version of the `C_rig` derivation was wrong and was corrected and re-judged offline before reporting any verdict** (`--reaudit`):

- The first version used the **maximum** per-dimension variance: `(d/2)ln(1+V_max/σ²)`——this is also a legitimate upper bound, but too loose (it raised values by 5–10 nats in this case), and it directly **made C2 vacuous** (the number of points with `C_rig < R_IB` was 0);
- The correct form uses the **mean** (Jensen, `ln` concave): `I(z;z+n) = Σ½ln(1+λᵢ/σ²) ≤ (d/2)ln(1+λ̄/σ²)`, and the per-vector RMS normalization fixes λ̄ at ~1;
- The measured `var_ratio` (max/mean) reaches 2.45–8.58, **with no effect whatsoever on the bound in its correct form**——which also explains why the first version was so wildly wrong;
- After the correction C2 became non-vacuous and passed, and the verdict changed from PARTIAL to SUPPORT. **The criterion text was not touched; what changed was only my own derivation of the bound**; the correction record was written both into the script docstring and into the docstring of `capacity_rigorous()`.

## 6. Boundaries (honest)

- Single task (grid64, deterministic 64 classes) and single architecture (3→64→64 MLP, bottleneck width 64); "5.3×" is a **measured value under this protocol**, not a universal constant;
- The precise meaning of "structural" is **under this protocol** (this architecture + 120–600 epoch + SGD): whether a wider encoder or a different noise-annealing strategy could push the discount down was not tested;
- Normalization option: the main run used `normalize="rms"` (the mean is fixed, the maximum is uncontrolled); the package now adds `normalize="per_dim"` (runtime per-dimension RMS, making each dimension's variance ≤1) for scenarios that need per-dimension rigor, **not used this time** (because the Jensen form is already rigorous enough); its self-check covers only the forward pass and variance control;
- The noiseless ceiling shows that **the architecture and optimization are fine without noise** (retention 0.99–1.00), so the gap can only be attributed to the noisy channel itself and to the difficulty of "learning a noise-robust encoding".
