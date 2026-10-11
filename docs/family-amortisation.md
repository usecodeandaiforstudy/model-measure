# Family-internal amortization library note (family-amortisation)

> Record date: 2026-09-30　Experiment: `experiments/family_library_demo.py` (pre-registered criteria in the script docstring)
> Question: can **second-scale cheap descriptors + same-family calibration** predict readings that would otherwise require a **trained witness** (quasi-ex-ante)?
> Why it is worth doing right now: the PW experiment has just proved that the readings require a witness **trained on the task** (threshold acc ≥ 0.577 to exclude K=2),
> and that is minutes to hours ⇒ what amortization saves is exactly this cost.
> Honest premise: the family library is **calibration**, not a law, and applies only within a family, only under a **fixed witness protocol**, and only as a **planning prior** (not a verdict).

---

## 1. Design

Family = a class subset of CIFAR-10 (fixed permutation seed 0) C ∈ {2,3,4,5,6,10}; per instance:
- **Measured ground truth**: a **witness trained on the task** (`SmallCNN(64,256,96)`) → acc → `R_IB` → `K_min`;
- **Cheap descriptors (second-scale, zero training)**: raw-pixel ridge regression accuracy `acc_ridge`, randomly initialized backbone feature ridge regression `acc_randfeat`;
- **Predictor (pre-registered, two-stage)**: ① `acc_pred = clip(a + b·acc_ridge)` (fitted **leave-one-out**) ② the project's own formula → `R_IB_pred` → `K_min_pred`.

Two **witness protocols** were run (the family library is protocol-dependent, and this is itself one of the things being verified):

| Protocol | Result |
| --- | --- |
| 12 ep / 20% subset | `R_IB` for 6/6 instances **all < ln 2** ⇒ **the measured exclusion set is entirely empty** (degenerate ground truth) |
| **25 ep / 50% subset** | 4/6 instances exclude `{2}` (ground truth has content, used for the main verdict) |

## 2. Main result (25 ep / 50%)

| C | acc_ridge | trained witness acc | R_IB measured | R_IB predicted (LOO) | Δ | K_min measured→predicted | exclusion set measured→predicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.819 | 0.985 | 0.613 | 0.693 | **+0.08** | 2→2 | ∅→∅ |
| 3 | 0.642 | 0.927 | 0.787 | 0.645 | −0.14 | 3→2 | `[2]`→∅ (under-exclusion, safe direction) |
| 4 | 0.548 | 0.784 | 0.628 | 0.762 | **+0.13** | 2→3 | ∅→**`[2]` (over-exclusion!)** |
| 5 | 0.442 | 0.795 | 0.819 | 0.693 | −0.13 | 3→3 | `[2]`→`[2]` |
| 6 | 0.390 | 0.786 | 0.929 | 0.695 | −0.23 | 3→3 | `[2]`→`[2]` |
| 10 | 0.299 | 0.591 | 0.728 | 1.092 | **+0.36** | 3→3 | `[2]`→`[2]` |

**Verdict = MIXED/REFUTED**:

| Criterion | Result |
| --- | --- |
| **F-1** within-family law (Spearman(`acc_ridge`,`acc_t`) ≥ 0.80) | **PASS** (+0.829; +0.943 for the 12ep round) |
| **F-2** prediction usable (leave-one-out `|ΔR_IB| ≤ 0.30` for all instances) | **FAIL** (C=10 off by **0.36** nats) |
| **F-3** safe (predicted exclusion set ⊆ measured exclusion set) | **FAIL** (one **over-exclusion** at C=4) |
| F-4 control: labels only (`R_IB = ln C`) under the same conventions | over-exclusion on **4/6** instances (`{2}`/`{4}`/`{4,8}`) |

## 3. Two failure mechanisms (both quantified)

1. **Threshold flip (main cause)**: the exclusion verdict is a **threshold function** (`log K < R_IB`), while the grid threshold spacing is only `ln 2 ≈ 0.69` nats, and **a prediction error of 0.10–0.36 nats already reaches 15–50% of the spacing** ⇒ instances near the boundary must flip. C=4 is the specimen: the measured `R_IB = 0.628` is only **0.065 nats** from `ln 2 = 0.693`, so a prediction biased by +0.13 crossed over it.
2. **Extrapolation at the family edge**: C=10 is the endpoint of the family, and its `acc_ridge = 0.299` falls **outside** the calibration interval [0.39, 0.82] of the other 5 instances; the linear calibration extrapolates the witness accuracy up to 0.719 (measured 0.591) ⇒ Δ=0.36 nats. **Amortization is usable inside the family, unusable at the family edge.**

## 4. Usable safety rule (derived from the data, already validated on this family)

**First compute the "threshold margin"**: `m = min_K |R_IB_pred − ln K|`. If `m` is no greater than the upper bound on the prediction error (0.36 nats measured for this family), then that prediction's **verdict is unstable and must be measured**; only when `m` is clearly greater than the error is the predicted exclusion set trustworthy.
On this family: C=10 has `m = 0.399` ✓ correctly judged stable (and indeed did not flip); C=4 has `m = 0.069` ⇒ correctly flagged "unstable" (it is precisely the only flipped instance)——**the rule gave the correct warning at the only point of failure**.

But the same table also yields a more sobering conclusion: **5 of the 6 instances have a reading within 0.3 nats of some threshold** ⇒ within this family, the amortization library **can only say "go measure" most of the time**. The reason is structural: `R_IB` is on a logarithmic scale, the grid thresholds are `ln 2` apart, and the prediction error is of the same order as that.

## 5. Conclusions and product implications

- **Quantitative prediction holds** (F-1 passes; within the family `|ΔR_IB| ≤ 0.23`): cheap descriptors do track the strength of the trained witness, which is the measured evidence for the "within-family law";
- **Verdict-level use does not hold** (F-2/F-3 do not pass): the error is of the same order as the threshold spacing ⇒ boundary flips are the norm rather than an accident; the family edge additionally carries extrapolation error;
- **Far safer than "labels only"** (over-exclusion 1/6 vs 4/6), so its correct use is as a **screener + planning prior**: first use second-scale descriptors to locate "roughly where the reading is, how far from the threshold", **adopt directly only the instances with a sufficiently large margin**, and go measure the rest;
- **Does not change the project's overall principle**: a provable exclusion can still only come from a **real witness**; the family library turns "should we spend those few minutes" into a computable question.

## 6. Boundaries (honest)

- Single family, 6 instances, single dataset (CIFAR-10), single architecture, two witness protocols; **a leave-one-out calibration with n=6 is only enough to establish "this route is viable/not viable", not enough to support any quantitative accuracy promise**;
- Only two descriptors are used (raw-pixel ridge regression, random-feature ridge regression); a larger descriptor set (data volume, inter-class separability measures, wall-geometry probes) was not included;
- The definition of "family" is decided by the user's fixed pipeline (same data pipeline, same architecture, same witness protocol)——changing the protocol changes the target (the two protocols this round have demonstrated this empirically);
- The verdict threshold 0.30 nats and the 0.36 nats in the safety rule are both **measured values for this family**, not universal constants.
