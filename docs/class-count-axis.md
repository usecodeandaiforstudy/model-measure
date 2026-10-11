# Scale (class-count axis) note (class-count-axis)

> Record date: 2026-09-30 · Experiment: `experiments/scale_c_demo.py` (pre-registered criteria in the script docstring)
> Question: when the **class count C is scaled up 125×** (C = 8 → 1000), do the tool's two halves — the certifiable **requirement/exclusion** and the empirical **measured wall/looseness** — still hold?
> Why only the C axis: see the analysis of "can a scale experiment be done without downloading data" — C is the only axis that costs almost no compute (it only changes the classification head) and can span 100× offline;
> this machine has only CIFAR-10/MNIST and 8 GB of VRAM, so the three axes "natural diversity / large pretraining / 224×224" cannot be done offline.

---

## 1. Design and two design corrections (both on record)

- **Task**: the project's own 3-dimensional lattice family (`mnist_demo.make_grid64`'s m=4 special case generalised to m³): C = m³,
  m ∈ {2,3,4,6,10} → C ∈ {8, 27, 64, 216, 1000}; **the cell geometry is completely unchanged** (spacing 3.0, noise σ),
  150 samples per class, 80/20 split, 2 seeds.
- **Model**: `Linear(3→64) → ReLU → bottleneck(64) → Linear(64→C)`; **the reference has no bottleneck**, students VQ(K), KD, 60 epoch, Adam 1e-2.
- **σ calibration (the selection process is recorded)**: a first sweep over σ ∈ {0.5, 1.0, 1.5, 2.5} × 5 C values (20 epoch): **σ=1.0 is the only setting that puts all five class counts inside the information band 0.02–0.60** (ε = 0.158/0.275/0.318/0.365/0.451), so σ=1.0 is adopted.
  **Anchor check**: at σ=0.5 the m=4 (C=64) case gives ε=0.035, matching the 0.014–0.039 already measured for cp3's MLP ✓
- **Design correction 1 (the reference must have capacity)**: if the reference kept the 256-codeword bottleneck, then at C ≥ 256 **the reference itself is jammed** (it needs `ln C ≥ 5.5` nats while `log 256 = 5.55`) — what is measured is the reference's bottleneck, not the task. So the reference was changed to **no bottleneck**.
- **Design correction 2 (the abandoned synthetic families)**: three synthetic families were tried first — "white-noise projection / fixed smooth basis / self-similar basis" — and all three failed in measurement (C≥256 permanently at chance; one version was not even learnable at C=10, because the basis vectors were unit *norm* rather than unit *per-pixel RMS*, making the signal 55× weaker). **Switching to the project's own lattice family passed on the first try**, and it could be anchored against cp3.

## 2. Results (ten instances, all passing the S1 information band)

| C | m | ε (two seeds) | R_IB | K_min | `K_min/C` | Excluded set | Relative gap at the largest K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2 | 0.204 / 0.188 | 1.18 / 1.23 | 4 | 0.50 | `[2]` | 0.09 / 0.25 (K=8) |
| 27 | 3 | 0.269 / 0.284 | 1.84 / 1.77 | 7 / 6 | 0.22 | `[2]` | 0.62 / 0.47 (K=32) |
| 64 | 4 | 0.315 / 0.306 | 2.23 / 2.28 | 10 | 0.16 | `[2, 8]` | 0.28 / 0.33 (K=128) |
| 216 | 6 | 0.346 / 0.352 | 2.87 / 2.84 | 18 | 0.083 | `[2, 8]` | 0.66 / 0.68 (K=128) |
| **1000** | 10 | 0.421 / 0.422 | **3.32** | **28** | **0.028** | `[2, 8]` | **0.66 / 0.68 (K=512)** |

**Verdict = PARTIAL**: S4 (zero false exclusions at every rung) **PASS**, S5 (the exclusion's reach grows with C) **PASS**, S3 (looseness direction/asymptote) **FAIL** — because **only 1/10 instances located a wall inside the K grid** (so S2 cannot be fitted).

## 3. Four conclusions

1. **The requirement side follows the formula exactly at 1000 classes**: `R_IB` goes from **1.18 → 3.32 nats** (C 8→1000), `K_min` from 4 → 28 (sublinear, `K_min ∝ C^0.58` at ε≈0.42). The formula itself is the same function, but **its input is the ε measured on the 1000-class model** — this half has no problem at scale.
2. **Soundness survives across 125×**: **ten instances with zero false exclusions** (including C=1000) ⇒ the headline rule `log K < R_IB ⇒ cannot fit` is still safe at scale.
3. **The `ln C` ceiling is lifted "relatively", but the absolute reach is still governed by the error rate**: `K_min/C` falls from **0.50 (C=8) to 0.028 (C=1000)**, i.e. the ceiling constraint is **18×** looser ⇒ the larger the scale, the more room the tool's exclusion has (the pre-registered P1 is partially confirmed); but in absolute terms the excluded set only grows from 1 rung to 2, because ε=0.42 pins `R_IB` at 3.32, far below `ln 1000 = 6.9`. **⇒ At scale the truly binding constraint has changed from the "class-count ceiling" to the "achievable error rate".**
4. **The wall could not be located at scale — and the way it failed is itself a conclusion**: at C ≥ 27 the student cannot catch the reference anywhere inside the K grid (**at C=1000, K=512 (`log K = 6.2` nats, already 1.9× the requirement) still has a 66% relative gap**). By contrast cp3 (C=64, its teacher itself carrying a 256-codeword VQ) had its wall at K=96 ≈ **1.2×** the requirement ⇒ **the empirical wall is a property of the "reference–student pairing", not of the task**. This is a strengthened version of LOOSENESS's "`K_wall` depends on the protocol" (with a much larger effect size).

## 4. Two repairable defects this run exposed (for the re-run)

1. **The grid-cap rule cuts off its own evidence**: the script's `if K > 2C: continue` removes the large-K points — at C=216, K=512 is skipped, at C=1000, K=2048 is skipped, **and the wall sits exactly at those positions**. The re-run should change it to `K ≤ 8C` or set no cap (give the K grid up to 8192).
2. **The reference should be an architecture "of the same family as the student"**: this round's reference had no bottleneck (to satisfy "at large C the reference must itself have capacity"), at the cost of **a VQ student having to chase a teacher with a continuous representation**, a huge gap. Better: give the reference a **large but finite VQ** (e.g. K_t = 8C), so that it is neither jammed by its own bottleneck nor of a different family from the student's quantized one (this is how CP-H/cp3 do it, and that is why their walls land at 1.2× the requirement).

## 5. The net conclusion on the "scale boundary" question

- **The certifiable half (requirement + exclusion) is robust across 125× the class count**: exact scaling, zero false exclusions, and the ceiling constraint relatively looser;
- **The empirical half (measured wall + looseness) is still unmeasured at scale**: this round's failure is diagnosable (the two defects above), not evidence that "the wall does not exist";
- **A methodology lesson confirmed in passing**: at scale, "the reference must itself have enough capacity" (otherwise you measure the reference's bottleneck) and "reference–student must be of the same family" (otherwise you measure distillation difficulty) **hold simultaneously** — neither is exposed at small C (CIFAR-10/MNIST), so they are **scale-specific methodological requirements**.

## 6. Round 2 (corrected protocol): the wall is located, but it reveals a "requirement–reference" sandwich

**Protocol correction** (recorded before round 2 started): ① the reference becomes a **large but finite VQ of the student's own family**, `K_t = min(8C, 8192)` (`log K_t = ln C + 2.08`, so it is not itself a bottleneck); ② the student grid cap is relaxed from `2C` to **`8C`** (in round 1 exactly this cap cut off the point where the wall sits); ③ only 1 seed is run (in round 1 the two seeds differed by <0.01 in ε and gave identical excluded sets). Criteria S1–S5 unchanged.

| C | `K_t` | ε | R_IB | K_min | **K_wall** | Looseness | Student relative gap (per K) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 64 | 0.262 | 0.99 | 3 | **8** | 2.09 | K2:0.74 · K8:0.02 · K32:0.04 |
| 27 | 216 | 0.348 | 1.52 | 5 | **128** | 3.20 | K2:0.90 · K8:0.68 · K32:0.47 · K128:0.05 |
| 64 | 512 | 0.404 | 1.81 | 7 | **512 (=K_t, saturated)** | 3.44 | K2:0.97 · K8:0.84 · K32:0.62 · K128:0.29 · K512:0.02 |
| 216 | 1728 | 0.437 | 2.34 | 11 | ∈(512, 1728] not located | — | K2:0.99 · K8:0.96 · K32:0.85 · K128:0.60 · K512:0.23 |
| 1000 | 8000 | (supplementary run in progress, see the end of this note) | — | — | — | — | — |

**Verdict = PARTIAL**: S4 (zero false exclusions at every rung) **PASS**, S5 (reach grows: [2] → [2,8]) **PASS**, **S3 (looseness non-increasing) FAIL** — the measured looseness **rises 2.09 → 3.20 → 3.44**, the opposite of round 1's pre-registered directional prediction.

### 6.1 Main finding: the measured wall is squeezed between "requirement" and "reference capacity"

All three located doses agree on **`K_min ≤ K_wall ≤ K_t`**: at C=8 `3 ≤ 8 ≤ 64`, at C=27 `5 ≤ 128 ≤ 216`, at C=64 `7 ≤ 512 = K_t` (**saturated against the reference's bottleneck**).
Together with round 1's result (reference **with no bottleneck**) — the same task's **wall could not be located (>2048)** — this gives a **controlled comparison**:

> **the same C=64 task, the same σ, the same budget, the same architecture, the same grid**, with only the reference changed:
> reference with no bottleneck ⇒ wall **>2048**; reference `K_t=512` ⇒ wall **=512**. **The wall moved ≥4× with the reference** ⇒ the measured wall is a property of the "task–reference" pairing.

> **Where this stands now (added 2026-10-05, `wall-saturation.md`).** The "task–reference pairing" reading was
> too narrow: the wall was later shown not to respond to the task's true information (a 10× better witness left it at
> 512) and to respond strongly to the *students' training budget* (3× the epochs halved the retention deficit). The
> correct statement is that the wall is **protocol-relative** — it is set by how far the student converges under the
> distillation objective, not by the task's requirement. This section stands as the record of what that round measured.

This also explains all of the project's earlier wall readings: cp3 (C=64, `K_t=256`, wall 96) and CP-H (CIFAR-10, `K_t=256`, wall 128–192) **both fall inside their own `[K_min, K_t]` interval**, and neither saturates against `K_t`.

> **Correction (given by §7's fixed-ceiling run)**: what this section originally wrote, "C=64 saturates against `K_t=512`", is **wrong**. After raising the ceiling to `K_t=2048`, C=64's wall **is still 512** ⇒ that dose was not limited by the ceiling (`8C=512` and the true wall 512 simply coincide numerically). Hence "the wall's upper bound = the reference's capacity" shows up as "the wall is pushed outside the grid" only when the **reference is continuous (unquantized)**; with a quantized reference the wall's upper bound is obviously ≥ the measured wall, and the true wall is decided by the **task + the distillation protocol**.

### 6.2 How α is measured, and its limits

The three located walls give `ln K_wall = 2.018·ln C − 2.023` (**R² = 0.991**, n=3) ⇒ wall ∝ C². Round 1 worried that this α was contaminated by `K_t = 8C`; **§7's fixed-ceiling run (`K_t=2048` unchanged across C) gives the same three walls 8/128/512**, showing that within C ≤ 64 this α is **not caused by the ceiling** (details in 7.1).
One more note: given the α fit, S3's "asymptotic value" clause (looseness → `α/(1−ε)`) is **nearly a tautology** (`looseness = ln K_wall/R_IB` while `R_IB ≈ (1−ε)ln C`); the measured `2.018/0.596 = 3.38` vs looseness `3.44` (difference 0.06) does agree, but it is **not independent evidence** — the genuinely substantive directional clause was falsified (the official verdict: Spearman = **+1.000**, n=3).

### 6.3 Corrected conclusions and the next step

- **The requirement side is robust again**: `R_IB` 0.99 → 2.34 nats (C 8→216), `K_min` 3 → 11, **zero false exclusions**, reach growing ✓;
- **The wall's measurability depends on the reference's magnitude**: `K_wall ∈ [K_min, K_t]` is the structural sandwich this round gives;
- **The next round (clean design, budget about 2–3 hours)**: `K_t` **fixed at 8192** (the upper bound does not move with C) + the top of the grid refined to ratio 2 (`…,512,1024,2048,4096,8192`, avoiding the grid gaps that C=216/1000 hit this round) ⇒ only then can the **task's** α and `looseness(C)` be given;
- **Performance note**: when KD's teacher carries 8000 codewords, every step computes `cdist(128×8000)`, which takes a unit from 5.5 minutes to **18 minutes**; fixing `K_t=8192` pays the same cost, so the budget must be estimated accordingly.

## 7. Round 3 (clean design: ceiling fixed): α de-contaminated, the wall is "steep first, then saturating"

**Design** (recorded before the run): `K_t = 2048` **fixed across all C** (`log 2048 = 7.62 ≥ ln 1000 = 6.91`, so it is not a bottleneck at any C); the student grid is widened to `[2,8,32,128,512,1024,2048]` with ratio 2 at the top; everything else (task, σ=1.0, 150 samples/class, 60 epoch, Adam 1e-2, seed 0) is the same as round 2.
**Performance fix**: the teacher's logits are **cached once** per instance (fp16) and the student loop reads the cache — that loop is a **line-by-line replica** of the package's KD objective, numerically verified in `--selftest` (**the loss agrees bit for bit: 2.6661 vs 2.6661**). This takes the per-instance cost from round 2's 18 minutes/unit down to 0.1–13 minutes/unit this round (for C=1000, all instances in 67 minutes: reference 340 seconds + cache 2 seconds + 7 students 3656 seconds).

| C | ε | R_IB | K_min | Excluded set | **K_wall** | Looseness |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.175 | 1.28 | 4 | `[2]` | **8** | 1.63 |
| 27 | 0.296 | 1.72 | 6 | `[2]` | **128** | 2.82 |
| 64 | 0.348 | 2.07 | 8 | `[2]` | **512** | 3.01 |
| 216 | 0.446 | 2.29 | 10 | `[2, 8]` | **512** (stops growing) | 2.72 |
| 1000 | 0.613 | 2.01 | 8 | `[2]` | not located (still 0.26 short at K=2048) | — |

**Verdict = MIXED/REFUTED** (S4 passes; S3's direction ✗, S5 ✗): S2 `ln K_wall = 1.284·ln C + 0.066` (**R²=0.830**, n=4); S3's direction Spearman **+0.400** (looseness is **non-monotone**); S5's reach did not grow because the largest C falls outside the information band (ε=0.613 > 0.60).

### 7.1 Three substantive conclusions

1. **The pre-registered prediction hits**: round 2 worried that "α was contaminated by `K_t=8C`"; with `K_t` fixed at 2048 this round, **α falls from 2.018 to 1.284** ⇒ round 2's α was indeed too high (because the ceiling moved with C); after de-contamination the wall scaling is **significantly below 2**.
2. **The wall is not a single power law but "steep first, then saturating"**: 8 → 128 → 512 → **512** (C=216 and C=64 share 512) ⇒ in this task family the wall stops growing after C≈64 (≈2.4×C), so the α=1.284 (R²=0.830) obtained by fitting a single power law to all four points is itself only **the average across two regimes**.
3. **Looseness is non-monotone in C** (1.63 → 2.82 → **3.01** → 2.72): the mechanism is clear — `R_IB` rises slowly with `ln C` while the wall first jumps steeply in the low-C segment and then saturates ⇒ **looseness peaks in the middle (C≈64) and then falls back**. This is the **third falsification** of the derived prediction "looseness is non-increasing in C"; the correct statement is **"looseness has no single direction — it depends on whether the wall is in its growing or its saturating segment"**.

### 7.2 Soundness and the ceiling

- **S4 passes independently for the third time**: five C values (including 1000 classes) with **zero false exclusions** ⇒ the headline rule replicates three times across 125× the class count;
- **Corrects round 2's "saturation" statement**: this round, with `K_t=2048`, C=64's wall **is still 512**, the same as with round 2's `K_t=512` ⇒ that dose was not limited by the ceiling; round 2's "saturated against `K_t`" is a numerical coincidence (§6.1 has been corrected);
- **C=1000's reference is weak** (ε=0.613, falling outside the S1 band): 150 samples/class × 60 epoch is undertrained at 1000 classes; this is the **mechanism of S5's failure** and is a protocol problem, not a property of scale; its wall is also not located (still 0.26 short at the largest K=2048).

### 7.3 Final answer to the "scale boundary" question (the class-count axis)

- **The certifiable half is robust across 125× the class count** (consistent over three rounds: exact requirement scaling, zero false exclusions);
- **The empirical wall's scaling**: after de-contamination α≈1.28, but the true shape is **steep first, then saturating**; in this task family the wall caps out at ~512;

> **Correction (2026-10-05, `wall-remeasurement.md` §6).** The "saturating at ~512" reading was a
> **discretisation artifact of the ladder**: the coarse grid had no rung between 128 and 512, so C=64's true
> crossing (~256) could not be resolved and both C values reported 512. On fine ladders — now measured at four
> class counts — the walls are **128 (C=27), 256 (C=64), 512 (C=216), 768 (C=512)**, a wall ratio of **6.0** for
> a class-count ratio of **18.96**, and the fit is
> `ln K_wall = 0.603·ln C + 2.944` (**R² = 0.988, n = 4**) against the certified requirement's `K_min ∝ C^0.58`
> from round 1 measured below. The two slopes agree to 0.023: **the wall and the requirement grow at essentially
> the same rate once the ladder resolves the crossing**, so the "steep then saturating" sentence above stands
> only as a statement about the coarse ladder. One caveat, reported per instance: the crossing is *soft* at
> C=27 (the deficit falls 0.021 across it) and sharp at C=64/216/512 (0.090/0.135/0.082).
- **Looseness's behaviour at scale**: **non-monotone** (peaking in the middle); its shape is decided by "whether the wall is in its growing or its saturating segment", not by C or ε alone;
- **Still not covered**: the four axes of resolution, sample size, parameter count and natural diversity — the first three have since been measured in `scale-axes.md`, natural diversity remains unavailable offline — plus a more adequate reference budget at C=1000 (see §7.4, which has now been tried).

### 7.4 Follow-up: C=1000 with three times the student budget (added 2026-10-05)

Round 3 left C=1000 unlocated (at K=2048 the student was still 0.264 short of the reference). WALL-SATURATION
then showed the students' retention deficit is partly **budget**-limited (3x the epochs halved it at C=216,
sigma=0.5), so the same lever was applied here with the round-3 teacher and cached soft targets reused, leaving
the student budget as the only change. Pre-registered: W1 locate the wall (gap < 0.10), W2 still unlocated with an
improvement ratio of 0.45-0.65 (the C=216 value), W3 no response (< 20% improvement).

| K | 60 epochs | 180 epochs | ratio |
| --- | --- | --- | --- |
| 2048 | +0.264 | **+0.170** | **0.64** |
| 1024 | +0.263 | — | — |
| 512 | +0.393 | — | — |

**Verdict W2: the wall is still not located, and the budget-scaling hypothesis is confirmed** — the gap improves
by 35%, at a ratio (0.64) inside the pre-registered band around C=216's 0.55.

> **Correction (2026-10-05, `budget-and-recipe.md`).** "Confirmed" was too strong: it rested on two points.
> A three-point dose-response on the C=216 instance shows the deficit is **non-monotone** in the budget
> (0.1207 → 0.0685 → 0.1087 at 60/180/360 epochs), so a "budget-scaling law" does not exist and this section's
> 0.64 is a two-point ratio of a non-monotone quantity. The measurement stands; the generalization does not. So at C=1000 the retention bar is
missed for a reason that is continuous with the rest of the sweep (the students' convergence under the distillation
objective, not the codebook), and three times the budget does not close a 26% deficit. A rough extrapolation from
the same ratio suggests another 2-3 doublings of budget might, but that is **not measured** and is recorded as an
open endpoint rather than a result. Two consequences for how the C=1000 row should be read:

- the row's "wall not located" is **not** evidence that the requirement exceeds 2048 — the certified requirement at
  C=1000 is only `K_min = 8` (the weak reference reads ε=0.613, so `R_IB = 2.01`), and
- budget beats codes here too: the 180-epoch student at K=2048 (gap 0.170) beats every 60-epoch rung including
  K=1024 and K=2048 (0.263/0.264) — the same ordering WALL-SATURATION found.

## 8. Boundaries (honest)

- Three rounds combined: a single task family (a 3-dimensional lattice), a single architecture (a 64-wide MLP), and **"scale" means the class count only** (the four axes of resolution/sample size/parameter count/natural diversity are unavailable offline); round 1 was 10 instances × 2 seeds, round 2 4 instances × 1 seed, round 3 5 instances × 1 seed;
- The compute facts for C=1000: round 1 was 25–33 minutes per instance (reference with no bottleneck); round 2 rose to **18 minutes/unit** because the reference carried 8000 codewords; round 3 used the **target cache** to bring it back down to **67 minutes per instance** (including 7 students);
- **Looseness's behaviour at scale is no longer blank**: the answer is **non-monotone** (§7.1 item 3), and its shape is decided by whether the wall is in its growing or its saturating segment; but the mechanism behind "why the wall saturates at ~512" has not been found (candidates: the imitability ceiling of KD into a VQ student, the 3-dimensional lattice structure of the task family, the capacity of the 64-wide representation) — that is the question to ask next;
- Locating the wall at C=1000 needs a stronger reference (a larger budget) and a larger K grid (this round it is still 0.26 short at the maximum, 2048).
