# Witness note (witness-strength)

> Record date: 2026-09-30  Experiment: `experiments/witness_demo.py` (pre-registered criteria in that script's docstring)
> Origin of the question: the requirement reading `R_IB` is **witness-relative** (inverted via Fano from the reference model's achieved error). Two questions were open:
> (1) How cheap can a witness get and still be usable? (2) Can the witness be replaced by a "read straight off the labels" reading (`H(Y)=ln C`)?

---

## I. Design and pre-registration

| item | content |
| --- | --- |
| task A | clean grid64 (a 64-class deterministic task; the encoding constants are copied verbatim from `cifar_demo.make_grid64_images`) |
| task B | merged-class grid64: 8 templates × 8 **unlearnable** duplicate labels per template → `H(Y)=ln64=4.159` while `H(Y\|X)=ln8=2.079`, Bayes ceiling ≈ **0.125** |
| witness ladder | W1 = 1-NN (raw pixels, **zero training**) / W2 = linear probe (seconds) / W3 = same-architecture CNN **4ep** / W4 = same-architecture CNN **40ep** (the calibrated reference, also the KD teacher) |
| label-only control W0 | requirement := `ln C` (no model) |
| measured truth | the student grid (K varies, everything else fixed) distilled from W4; `rel(K) = gap/acc_W4`, success `rel≤0.10`, failure `rel≥0.50` (relative thresholds; the amendment record is in the script docstring (3)) |
| criteria | T1 `R_IB` is non-decreasing along the ladder; T2 **soundness**: no witness may exclude a student that measurably matches the reference model; T3 coverage profile; U1 label-only false-exclusion count |

## II. Results (3 seeds / 2 seeds)

**Task A (clean grid64)**

| witness | acc | R_IB | K_min | coverage | false exclusions |
| --- | --- | --- | --- | --- | --- |
| W1 1-NN (0 ep) | **1.000** | 4.16 | 64 | **1.00** | 0 |
| W2 linear probe | 0.832–0.839 | 3.01–3.05 | 21–22 | 0.70–0.78 | 0 |
| W3 CNN 4ep | 0.796–0.836 | 2.81–3.04 | 17–21 | 0.70–0.78 | 0 |
| W4 CNN 40ep | 0.999–1.000 | 4.15–4.16 | 64 | 1.00 | 0 |
| W0 label-only | — | 4.16 | 64 | — | 0 |

**Task B (merged-class grid64, Bayes ceiling 0.125)**

| witness | acc | R_IB | K_min | coverage | false exclusions |
| --- | --- | --- | --- | --- | --- |
| W1 / W2 / W3 | 0.117–0.124 | 0.14–0.15 | 2 | 0.00 | 0 |
| W4 CNN 40ep | 0.122–0.134 | 0.15–0.18 | 2 | 0.00 | 0 |
| **W0 label-only** | — | **4.16** | **64** | — | **2** |

W0's two false exclusions: `K=32` (seed 0) and `K=16` (seed 1) -- **measurably matching the reference model**, yet judged dead by "`K < C` ⇒ does not fit".

### Extension run (`--extend-top`, 2026-09-30, additional measurements reusing the cached base levels)

The base grid's top stops at K=64 (exactly at the wall), with no "twin-student" levels, so task A's "measured success set" was bound to come out empty (the long KD tail decay). After adding A:`+[128,192,256]` and B:`+[128,256]`, the measurements are:

| | measured success set S | measured failure set F |
| --- | --- | --- |
| task A seeds 0/1/2 | `[128,192,256]` / `[192,256]` / `[128,256]` | `[2..48]` (K=64 falls in the gray zone: retention 50–90%) |
| task B seeds 0/1 | `[32,64,128,256]` / `[16,256]` | `[2,3,4]` / `[2,3,4,6]` |

**T2 thereby becomes a real test**: the witnesses' excluded sets (W1/W4 up to 48, W2/W3 up to 16) and the success set (≥128) are **non-empty on both sides and disjoint** → no false exclusions, **but the safety margin is large** (excluded-set upper bound 48 vs success-set lower bound 128, with the level 64 in between falling in the gray zone). The criteria and the remaining conclusions are unchanged (MIXED; U1 is still 2 false exclusions).

### Task C (the real CIFAR-10 task, reusing the CP-H cache; `--task C`)

**Design**: the reference W4 = the CP-H calibrated teacher (SmallCNN width96 / K=256 / 120ep cosine, acc 0.799 / 0.768 / 0.803, **no training**); the measured truth = the CP-H student grid (13 levels × 3 seeds, **read from cache**); three new cheap witnesses: W1 1-NN on raw pixels (memory 2000 / query 1000, zero training), W2 ridge-regression linear probe (raw pixels, a 10000-sample train subset), W3 = same-architecture CNN **8 epoch** (15× cheaper than the CP-H budget).

| witness | acc | R_IB | K_min | excluded set | coverage | false exclusions |
| --- | --- | --- | --- | --- | --- | --- |
| W1 1-NN raw pixels | 0.243 | **0.08** | 2 | ∅ | 0.00 | 0 |
| W2 linear probe | 0.303 | 0.16 | 2 | ∅ | 0.00 | 0 |
| W3 CNN 8ep | 0.517–0.571 | 0.55–0.68 | 2 | ∅ | 0.00 | 0 |
| **W4 CP-H teacher** | 0.768–0.803 | **1.25–1.37** | 4 | `[2, 3]` | **0.33–0.40** | 0 |
| W0 label-only | — | 2.30 | 10 | `[2,3,4,6,8]` | — | **0** |

Measured: the success set `{192,256}` / `{128,192,256}` / `{128,192,256}`; the failure set `{2,3,4,6,8}` (seed 1 also includes 12).

**Verdict (the pre-registered aggregation for task C alone) = SUPPORT**: T1 PASS (R_IB monotone `0.08 → 0.16 → 0.65 → 1.36`, span **17×**), T2 PASS (zero false exclusions), T3 **STEEP** (coverage `0 → 0.40`). **The aggregate verdict for the three tasks combined = MIXED** (the aggregation rule is "all tasks", and A/B did not pass T1).

**This is the answer, on a real task, to "how much is witness strength worth"**:

- **On this task the bound given by the cheap witnesses is zero** -- raw-pixel 1-NN, the linear probe, and even the 8-epoch CNN all have coverage **0**;
- Only the calibrated teacher excluded 2 levels (coverage 40%), **but even so it falls far short of the success boundary** (excluding `{2,3}` vs success at `≥128`) -- the "32× margin" between the Fano bound and the practical wall (recorded by CP-H) shows up here again;
- **Contrast with task B**: on task C the label-only reading (`K<10`) produces **no** false exclusions -- because CIFAR-10's fine-grained distinctions are learnable. This confirms that the failure condition of "label-only" is **the task containing unlearnable fine-grained structure**, not "the task being hard".

## III. Verdict (per pre-registration, as-is)

**The verdict differs with the "aggregation scope"; both readings are recorded**:

- **Three tasks combined (the aggregation of `--task all`, the script's default convention) = MIXED**: because the aggregation requires **all tasks** to pass T1, and tasks A/B do not (the cost ladder is non-monotone); T3 is SHALLOW on A/B.
- **Task C alone (the aggregation of `--task C`) = SUPPORT**: T1 PASS, T2 PASS, T3 **STEEP** (coverage 0 → 0.40).

Per-task detail:

| task | T1 monotonicity | T2 soundness | T3 coverage | U1 label-only false exclusions |
| --- | --- | --- | --- | --- |
| A (clean grid64) | **FAIL** (the free 1-NN is strongest, 4.16 > 3.04 > 2.89) | PASS (non-empty and true after the extension) | SHALLOW (W1/W4 both 1.00) | 0 |
| B (merged-class grid64) | **FAIL** (non-monotone) | PASS (the excluded set is empty → a weak test) | SHALLOW (all 0.00) | **2 (hits the prediction)** |
| C (CIFAR-10) | **PASS** (0.08→0.16→0.65→1.36) | PASS (zero false exclusions) | **STEEP** (0 → 0.40) | 0 |

## IV. Conclusions

1. **"Label-only" is refuted by measurement, not merely by argument**: on a task whose fine-grained distinctions are unlearnable (B) it judges students that **measurably match the reference model** dead (2/2 seeds); while on tasks with learnable fine-grained distinctions (A/C) it produces no false exclusions. ⇒ The failure condition is **the task containing unlearnable fine-grained structure** (`H(Y|X) > 0`), not "the task being hard".
2. **Witness strength is unrelated to training cost** (the real meaning of T1 being falsified on A): **the zero-training 1-NN is as strong as the 120/40-epoch teacher on the clean task** (acc 1.000), whereas the linear probe and the 4ep CNN are instead weak (0.83). The criterion for choosing a witness is **achieved accuracy**, not budget.
3. **On a real task witness strength really does determine the criterion's usability** (C's T1/T3): R_IB spans **17×** (0.08 → 1.36 nats) and coverage **0 → 0.40**; the raw-pixel witnesses (1-NN, linear probe) and even the 8ep CNN **give no exclusion at all**.
4. **But even on a real task the bound falls far short of the success boundary**: the strongest witness excludes only `{2,3}`, while measured success starts at `K=128` -- the "32× margin" between the Fano bound and the practical wall (CP-H) shows up here for the second time. ⇒ **The main factor limiting the criterion's usability is not witness strength but the looseness of the Fano bound itself**; this points the same way as the MOE experiment's conclusion (the flow tier's anchor must be the **measured wall**, not the Fano lower bound).
5. **Soundness (T2) shows zero false exclusions throughout**: on A it is a real test after the grid extension, and on C the excluded set and the success set are non-empty on both sides and disjoint. The shortcoming is that there is **never a "hugging the boundary" test** (A's K=64 and C's 12–64 all fall in the gray zone, and no witness excludes them).

## V. Boundaries

- Three tasks: **A** clean grid64 (3 seeds, a base grid of 11 levels + 3 top-extension levels), **B** merged-class grid64 (2 seeds, 8 + 2 levels), **C** real CIFAR-10 (3 seeds, 13 levels; the reference and the student grid are both read from the CP-H cache, with only one new 8ep cheap CNN witness);
- The grid fingerprint covers only the **base grid** (project convention, research-summary §10.4): the top extension reused every cached base unit and newly trained only 13 units; task C reused the CP-H's 42 cached units;
- Task B's reference model necessarily FAILs the calibrated G0 acc gate (the task's own ceiling is 0.125); the script has declared that gate inapplicable here (Fano needs only an "achieved error"); task C's reference model (0.768–0.803) is likewise below the default 0.95 gate -- the calibrated gate for CIFAR-10 is the **0.78** recorded by CP-H, and the script handles it as "report only, do not filter";
- Task C's W3 witness uses a **cheap protocol** (8ep, no cosine), different from the CP-H 120ep cosine budget; this is a constructional difference (already recorded).

## VI. Corrections and additions accounting (same day)

1. **Verdict aggregation crash**: `is_monotone` was passed a dict → `vals[1:]` KeyError; changed to take values in ladder order, and the self-test fixture was changed to a dict accordingly.
2. **A cache-hit teacher was not put in `eval()`**: BatchNorm used batch statistics → the KD soft labels differed from the first run, so with mixed cached runs the student results were not reproducible; `m.eval()` was added in `rebuild()`.
3. **The fingerprint violated the project convention**: originally the whole student grid went into the fingerprint, so adding a level invalidated everything; changed so that only the base grid goes into the fingerprint (the convention is in research-summary §10.4), and `--extend-top` was added for the top extension.
4. **Task C (CIFAR-10) added**: `--task C` / `--task all`; the reference and the measured truth reuse the CP-H cache, and the criterion set is unchanged (recorded in the script docstring (10)).
