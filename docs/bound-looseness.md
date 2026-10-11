# Looseness of the bound note (bound-looseness)

> Record date: 2026-09-30　Experiment: `experiments/looseness_demo.py` (pre-registered definitions and expectations in that script's docstring)
> Where the question comes from: WITNESS showed that "what limits the usability of the criterion is not witness strength but the looseness of the bound"; MOE showed that "the anchor of the flow tier must be the measured wall".
> This note merges the two into one **measurable** quantity: across the project's existing 38 scenarios, **how far the provable Fano lower bound actually sits from the "measured wall"**.
> All data are read from cache with **zero training** (cp3 12 + cph 3 + pa4 12 + moe 3 + witness 8 = 38 scenarios).

---

## 1. Definitions (fixed, no tuning)

| Quantity | Definition |
| --- | --- |
| `R_IB` | `task_info_fano(1 − acc_ref, C)`: the lower bound on the information requirement (provable) read back from the reference model's **achieved error** |
| `K_min` | `ceil(e^R_IB)`: the **provable exclusion** threshold (`log K < R_IB` ⇒ cannot fit) |
| `K_wall` | `measured_wall(grid, accs, acc_ref, rel_tol=0.10)`: the smallest K at which the student **retains ≥90% of the reference accuracy**, i.e. the **measured wall** (a relative criterion, comparable across tasks) |
| `wall_ratio` | `K_wall / K_min`: how many times higher the measured wall is than the provable threshold |
| `looseness` | `ln(K_wall) / R_IB`: how far the lower bound sits below the wall on the **log scale** (1.0 = the two coincide) |

## 2. Results (38 scenarios, all with a located wall; 0 with "wall beyond the grid")

Grouped by error regime (numerical ranges within a group):

| Scenario group | acc_ref | ε | R_IB | K_min | K_wall | wall_ratio | **looseness** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean deterministic (cp3 four cells / pa4 δ0 / moe / task A) | 0.999–1.000 | ~0 | 4.15–4.16 | 64 | 96–256 | 1.5–4.0 | **1.10–1.33** |
| Near-deterministic (cp3 MLP cells, ε 0.014–0.039) | 0.961–0.986 | 0.014–0.039 | 3.84–4.03 | 47–57 | 96 | 1.7–2.0 | 1.13–1.19 |
| Slightly diffuse (pa4 δ0.15 / moe) | 0.886–0.894 | 0.106–0.114 | 3.33–3.38 | 28–30 | 128–256 | 4.6–8.5 | 1.46–1.64 |
| **Real CIFAR-10 (cph / task C)** | 0.768–0.803 | 0.197–0.232 | 1.25–1.37 | **4** | **128–192** | **32–48** | **3.53–3.88** |
| Moderately diffuse (pa4 δ0.3) | 0.741–0.764 | 0.236–0.259 | 2.51–2.64 | 13–14 | 192 | 13.7–14.8 | 2.00–2.09 |
| Strongly diffuse (pa4 δ0.5) | 0.578–0.593 | 0.407–0.422 | 1.73–1.80 | 6–7 | 256 | 36.6–42.7 | 3.08–3.21 |
| Unlearnable fine-grained (task B) | 0.122–0.134 | 0.866–0.878 | 0.15–0.18 | 2 | 16–32 | 8–16 | **15.6–22.9** |

**The two hardest facts**:

1. **In 38/38 scenarios `K_wall > K_min` (factor 1.5–48×) —— the provable lower bound never coincides with the measured wall**; all of the project's criteria fall on the "safe but conservative" side.
2. `looseness` is strongly correlated with the error rate (**Spearman +0.837**), but its **substance is a ratio of distances on the log scale**: because `K_min = ceil(e^R_IB)`, we have `looseness ≈ ln K_wall / ln K_min`. A high error rate ⇒ `K_min` collapses to 2–7, while the wall still sits at 128–256 ⇒ the ratio is amplified. **It is not an independent physical quantity, but "how many lower-bound units apart the wall and the lower bound are on the log scale".**

## 3. Verdict (per pre-registration, as-is)

**Verdict = MIXED**:

| Criterion | Expected | Measured | Result |
| --- | --- | --- | --- |
| **D1** | on deterministic 64-class tasks (acc_ref ≥ 0.99), `looseness ≈ 1` | median **1.17** (n=13) | **PASS** |
| **D2** | for every scenario with ε ≥ 0.20, `looseness ≥ 3` | among the 12 scenarios the **minimum is 2.00** (pa4 δ0.3) | **FAIL** |
| **D3** | across scenarios Spearman(ε, looseness) ≥ 0.6 | **+0.837** | **PASS** |

D2's threshold was falsified: **ε ≥ 0.2 does not necessarily give looseness ≥ 3** (at the same ε, tasks with fewer classes / a smaller lower bound have a larger ratio: CIFAR-10 at ε≈0.20 has looseness 3.5–3.9, while pa4 at ε≈0.25 has only 2.0–2.1). The statement the data support (**for future pre-registration, not changing this round's verdict**): looseness is determined by `ln K_wall / ln K_min`, with the piecewise bands
`ε ≤ 0.04 → 1.1–1.2`; `ε ≈ 0.11 → 1.5–1.6`; `ε ≈ 0.25 → 2.0–2.1`; `ε ≈ 0.41 → 3.1–3.2`; `ε ≈ 0.87 → 15.6–22.9`.

## 4. Product implications (the behavior change landed this time)

**A report should give both numbers** (already implemented: `preflight_report(..., r_ib=<R_IB>)` now returns `k_wall` and `looseness`, and appends the measured wall to the verdict string):

| Use | Which number to use | Nature |
| --- | --- | --- |
| Saying "**does not fit**" (exclusion) | `log K < R_IB` (=`K < K_min`) | **provable**, direction-safe; always conservative |
| Deployment **sizing** (how small it should be) | `K ≥ K_wall` | **empirical**, depends on the protocol (KD budget/rel_tol); this is where the answer is |
| Judging "is the lower bound worth using" | `looseness` | ≈1.1–1.3: the two nearly coincide (clean synthetic); ≥3: the lower bound is only a floor far below the answer (real/diffuse/unlearnable tasks) |

Supporting tools: `measured_wall()` (the measured wall) and `looseness()` (the looseness), exported with `capacity_lens` v0.5.0; `preflight_report`'s fifth item (old) is followed by a new seventh item `practical_wall`.

## 5. Boundaries (honest)

- `R_IB` depends only on the reference model's error ⇒ **protocol-robust**; `K_wall` depends on the training protocol of the students on the same grid (KD budget, `rel_tol=0.10`) ⇒ **protocol-dependent**, and the protocol must be declared when comparing across experiments (this note uses each one's original protocol throughout);
- all 38 scenarios come from this project's small-scale tasks (synthetic grid64/route64 + CIFAR-10); no real large models;
- `looseness`'s numerator is `ln K_wall`, so it **underestimates when the top of the grid is insufficient** (this survey has 0 "wall beyond the grid", because the grids of cph/task C and pa4 all contain the twin student rungs);

## 6. Re-audit of the falsified criterion D2 (added 2026-10-05, `experiments/looseness_reaudit_demo.py`)

The round above pre-registered D2 as a **threshold** ("whenever the reference error is large, eps >= 0.20, looseness
>= 3"); it failed, and the failure is kept in the record. This section is the corrected pre-registration, run on the
same 38 scenarios with **zero training**, plus the two checks the first round lacked.

**The original verdict, re-run unchanged for the record**: D1 PASS (n=13, median looseness 1.169), **D2 FAIL**
(n=12 with eps >= 0.20, minimum looseness **1.995**), D3 PASS (Spearman(eps, looseness) = **+0.837**) ⇒ MIXED.

**D2' (the corrected form)**: looseness is an *affine* function of the reference error, `looseness ~ a + b*eps`, and
the class count adds nothing. The step-function form was the error: it put a cliff where the data has a slope.
Fitted on a **mechanically split** half of the scenarios (sorted by (source, name), alternating) and scored on the
other half:

| check | result |
| --- | --- |
| fit half (n=19) | `looseness ≈ 0.554 + 14.248*eps`, **R² = 0.848**, slope positive ⇒ C1 **PASS** |
| held-out half (n=19) | median absolute error **0.607** looseness units (pre-registered bar 0.5), 0 sign inversions ⇒ C2 **FAIL** |
| class-count control | adding `c*ln(n_classes)` raises R² by **+0.001** (c = −0.139) ⇒ C3 **PASS** |
| **verdict** | **PARTIAL**: the relation is real and the class count is irrelevant, but its out-of-sample accuracy misses the pre-registered bar |

The honest lesson is in that C2 result: an earlier in-sample fit of the same relation (a = 0.281, b = 16.97,
R² = 0.772, quoted in the README before this re-audit) **looked better than the relation generalises**. Anyone
using the slope should carry the ±0.6-unit out-of-sample error, and D2' stays labelled post-hoc-fitted on this
dataset until a fresh one confirms it.

## 7. How much of the reported looseness is the witness's weakness? (ground truth, added 2026-10-05)

On the lattice family the true `I(X;Y)` is computable (`wall-saturation.md` section 1), so the reported
looseness — which divides by the **witness-relative** requirement — can be recomputed against the **Bayes**
requirement. The difference is the part of the "slack" that is not the bound's fault:

| C | sigma | `K_wall` | eps (Bayes) | `R_IB` (Bayes) | looseness reported | looseness vs Bayes | witness share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.00 | 8 | 0.175 (0.187) | 1.275 (1.234) | 1.63 | 1.68 | **-3%** |
| 27 | 1.00 | 128 | 0.296 (0.245) | 1.723 (1.940) | 2.82 | 2.50 | 11% |
| 64 | 1.00 | 512 | 0.348 (0.273) | 2.071 (2.443) | 3.01 | 2.55 | 15% |
| 216 | 0.70 | 512 | 0.187 (0.077) | 3.887 (4.689) | 1.61 | 1.33 | 17% |
| 216 | 1.00 | 512 | 0.446 (0.299) | 2.292 (3.159) | 2.72 | 1.97 | **27%** |

**Median 15%, range −3% to 27%**, and the order is exactly the one the mechanism predicts: the share grows with
the reference's distance from the Bayes error (at C = 8 the reference sits *at* Bayes and the share is ~0; at
C = 216, sigma = 1.0 it is 0.147 above Bayes and the share is 27%). So of the "looseness" the tool reports, **up
to ~27% on this family is the witness being weak rather than the bound being loose** — a second, independent
measurement of the witness-relativity quantified in `wall-saturation.md` section 1 (there in `K_min` units:
2.4x at C = 216; here in looseness units: 27%).

> Filtering note (recorded because the first pass got it wrong): the artifacts accumulate rows across runs, and the
> first version of this table picked up a 5-epoch smoke row and a 2-epoch diagnostic row from other campaigns'
> jsonls. The table above keeps only the official runs (SCALE-C-3's full run, and the sigma arm at m = 6), deduped
> on (C, sigma, K_wall, R_IB).
- this survey is **descriptive**: the criteria are only D1–D3, with no post-hoc additions or removals; D2's failure is kept as-is.
