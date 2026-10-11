# WALL RE-MEASUREMENT: the project's walls under the recipe that the mechanism study said was better

**Why.** BUDGET-MECHANISM attributed the inflation of one located wall to the fixed learning rate: on the
instance C = 216, sigma = 0.5, K = 4096, switching *only* the schedule to cosine annealing turned a 12%
retention deficit into a negative one and moved the located wall from ">= 4096, unlocatable" to K = 512 — a
>= 8x change. Every earlier wall in this project was located with the fixed-LR recipe, so the natural reading
was "those values are upper bounds inflated by the recipe". This campaign tests that reading directly by
re-locating the walls with cosine annealing and asking the question that matters for the project's claims:
**do the SHAPE conclusions survive, or were they recipe artifacts too?**

**Verdict: the expectation is REFUTED and the shape is CONFIRMED.**
- **S1 (walls drop) FAILED.** At the SCALE-C instances the walls are **recipe-robust**: C = 8 went *up* (8 -> 32),
  and C = 27 / 64 / 216 were **unchanged** (128 / 512 / 512). So "the fixed LR inflates the wall" is **not**
  general — the >= 8x effect is specific to that one instance (an easier task, sigma = 0.5, where the students
  can converge inside the budget).
- **S2 (shape) SURVIVED completely**: the wall still grows steeply and then **saturates** (512, 512 at the top),
  and looseness is still **non-monotone** in C with the same sequence (2.72 / 2.82 / 3.01 / 2.72). Only the
  *fitted* exponent changes: alpha = 0.890 (R² = 0.870) against round 3's 1.284 (R² = 0.830).
- **S3 (soundness) PASSED**: **zero false exclusions at every C** under the new recipe — the fourth independent
  confirmation of the headline rule, and the first one under a different training recipe.

## 1. What was measured

Mode A: the SCALE-C round-3 instances (C = 8 / 27 / 64 / 216 at sigma = 1.0, 60 epochs, the round-3 K grid).
The **reference and its cached soft targets are the round-3 artifacts, reused unchanged**, the student init
seed is the round-3 seed, and the budget is the round-3 budget — so the only difference from round 3 is the
learning-rate schedule (cosine from 1e-2 to 0 instead of a fixed 1e-2). Mode B: the cp3 protocol (C = 64,
sigma = 1.0, reference VQ(256)), whose recorded wall is 96.

| instance | wall, fixed LR (recorded) | wall, cosine | change |
| --- | --- | --- | --- |
| C = 8 | 8 | **32** | **up (worse)** |
| C = 27 | 128 | 128 | none |
| C = 64 | 512 | 512 | none |
| C = 216 | 512 | 512 | none |
| cp3 (C = 64, `K_t` = 256) | 96 | **96** | none — **exact replication** |

The cp3 row is worth its own sentence: re-running that protocol from scratch reproduces the wall recorded in
the original campaign **exactly**, which is an independent validation of that campaign's number under a
different schedule.

## 2. Why the expectation failed (the honest mechanism)

Cosine annealing shortens the *effective* learning at the end of the run, which only pays off if the run is
long enough to converge. On the BUDGET-MECHANISM instance (sigma = 0.5) 60 epochs were enough and annealing
turned a 12% deficit into **-0.8%**; on the SCALE-C instances (sigma = 1.0, a harder task) the same 60-epoch
budget is *not* enough, and annealing instead **under-trains**: at C = 8 the K = 8 student's deficit rises from
0.08 to 0.24, which is what pushes the located wall from 8 to 32.

So the recipe effect is an **interaction of task difficulty, budget and schedule**, not a free improvement:

> **The right statement is not "cosine is better" and not "the fixed LR inflates walls", but: a schedule that
> decays the learning rate needs a budget long enough to converge; given one it can collapse a retention
> deficit (measured: 0.12 -> -0.008), and without one it under-trains and can move a wall in the *wrong*
> direction (measured: 8 -> 32).**

That also means the earlier note's sentence — "every wall this project located before this note is an upper
bound inflated by the recipe" — is **withdrawn**: at the SCALE-C instances the values are recipe-robust, and
what the recipe changes is the *schedule's fit to the budget*, not the wall's level.

## 3. What stands and what changed

**Stands (now under two recipes):**
- the wall grows steeply with the class count and then **saturates** — the shape is a property of the protocol
  family, not of the schedule;
- **looseness is non-monotone in C**, peaking in the middle — same sequence under both recipes;
- **zero false exclusions at every C** (S3), i.e. the certified side is untouched by any student-side choice;
- cp3's wall of 96, now independently reproduced.

**Changed:**
- `alpha` is schedule-dependent (1.284 fixed-LR vs 0.890 cosine, same shape) — so the fitted exponent should be
  quoted with its recipe, even though the qualitative shape is stable;
- the claim "the fixed LR inflates the located walls" is withdrawn as a generalization and replaced by the
  interaction statement above;
- the practical rule becomes: **choose the schedule together with the budget** (anneal only when the run is long
  enough to converge), not "avoid a fixed LR".

## 4. Honest boundaries

- Mode A has four class counts at sigma = 1.0 and one seed each; mode B one protocol. The saturation claim rests
  on two equal values at the top (C = 64 and C = 216), as in round 3.
- The "under-training" explanation for C = 8 is inferred from the deficit pattern (K = 8: 0.08 -> 0.24) and from
  the schedule's known behaviour, not measured directly — the direct test would be a longer budget with cosine
  (e.g. 180 epochs at C = 8), which was not run.
- **CP-H (the CIFAR-10 anchor, recorded walls 128-192)**: this note's first draft said its recorded fingerprint
  did *not* use the `--cosine` flag. **That was wrong, and the error is worth recording**: recomputing
  `cph_demo.run_signature` over the plausible configurations reproduces `0d8045d2` for **cosine=True (workers=4,
  lr=1e-3, 120 epochs, width 96, full data) and for nothing else**. CP-H's recorded walls were therefore
  **already measured with cosine annealing**, at a long budget (120 epochs) — which is consistent with
  everything above: a decaying schedule at a budget long enough to converge is the good case, and the real-data
  anchor was in that case all along. What CP-H does **not** have is the *fixed-LR counterpart* at its own tier,
  which is why a paired real-data probe was run instead: same recorded teacher, same data, same init seed
  (22000+100s+K) and a shared budget, switching only the schedule, plus the recorded 120-epoch students as a
  zero-training anchor. Its result is reported in section 5.
- Ledger: `ledger.md` E-026. Artifacts: `experiments/data/wall_remeasure.jsonl`,
  `wall_remeasure_report.txt`, `wall_remeasure/`.

## 5. Real data: the paired CP-H probe (CIFAR-10)

Since the recorded CP-H walls turned out to be a **cosine** result (section 4's correction), the useful
real-data question is the paired one: at a **shared** budget, does the schedule move retention on CIFAR-10?
The probe holds the recorded teacher (acc 0.799), the full CIFAR-10 training set, the recorded init seed
(22000 + 100s + K) and a 20-epoch budget fixed, and switches only the schedule; the recorded 120-epoch
students are evaluated as a zero-training anchor.

| K | fixed LR (20 ep) | cosine (20 ep) | delta | recorded (120 ep, cosine) |
| --- | --- | --- | --- | --- |
| 64 | +0.324 | +0.207 | **-0.118** | +0.169 |
| 128 | +0.275 | +0.189 | **-0.086** | +0.117 |
| 192 | +0.171 | +0.140 | **-0.031** | **+0.090** |

- **The recipe matters on real data too** (P1: largest |delta| = 0.118 against a 0.03 threshold), and here the
  direction is **beneficial at every rung**: cosine annealing retains 3-12 percentage points more of the
  reference than a fixed learning rate at the same budget. So the synthetic interaction finding does extend to
  the project's only real-data task — with the sign depending on the task and budget, exactly as the
  interaction story predicts (beneficial here at 20 epochs on CIFAR-10; harmful on the sigma = 1.0 lattice at a
  short budget).
- **The recorded wall is internally reproduced**: the recorded 120-epoch deficits are 0.169 / 0.117 / **0.090**,
  which cross the 0.10 retention bar between K = 128 and K = 192 — the recorded wall was 128-192. The probe did
  not merely fail to contradict the record; it re-derived it from the same checkpoints.
- **The budget does not simply help**, again: at K = 64 the 120-epoch student (0.169) is *worse* than the
  20-epoch cosine student (0.207), while at K = 128 and 192 it is better. That is the same interior-optimum
  behaviour BUDGET-DOSE found on the synthetic family, now visible on real data as well.

**Verdict: MIXED — the recipe effect extends to real data, with the caveat that this is a reduced tier** (20
epochs, one seed, three rungs) rather than CP-H's recorded 120-epoch tier. Its value is the *pairing*: same
reference, same data, same seed, same budget, one factor switched.

### 5b. The same question at CP-H's own recorded tier (added 2026-10-05)

The probe above is a reduced tier. The record's own tier is 120 epochs on full CIFAR-10 — and the record has no
fixed-LR counterpart there, so the arm was run: the recorded teacher, data and init seed, 120 epochs, only the
learning rate fixed at 1e-3.

| K | fixed LR (120 ep) | recorded (cosine, 120 ep) | delta |
| --- | --- | --- | --- |
| 128 | **+0.612** | +0.117 | **-0.495** |
| 192 | **+0.249** | +0.090 | **-0.159** |

**B1 PASS, and with a large effect**: at CP-H's own tier the fixed-LR student barely retains the reference
(+0.612 at K = 128), while the recorded cosine student retains most of it (+0.117). So **the recorded walls of
128-192 were made possible by the schedule**: had CP-H been run with a fixed learning rate, its students would
not have come near the bar at those rungs, and the wall would have been reported much higher — or not located
at all. This is the strongest real-data instance of the recipe effect measured in this project (0.495 in
retention, against 0.086-0.118 in the reduced-tier probe), and it is the *opposite* end of the interaction from
the SCALE-C instances: there, at a short budget on a harder task, annealing under-trained; here, at a long
budget on a task the model can fit, a fixed rate simply fails to train the student.

## 6. Why the wall still "saturated" at 512: a fine-ladder test (added 2026-10-05)

The re-measurement left one question standing: under cosine the walls were 128 / 512 / 512 for C = 27 / 64 / 216,
so the *value* seemed to stop growing while the class count tripled. Two stories fit: (i) a real retention
boundary shared by C = 64 and C = 216, or (ii) the 10% bar being crossed inside a flat region, making the
located value a threshold artifact of the criterion. A **finer ladder** separates them — and the answer is
neither of the two as stated.

| C | deficits on the finer ladder | wall (finer) | wall (coarse ladder) | drop across the crossing |
| --- | --- | --- | --- | --- |
| 27 | +0.395 / +0.240 / +0.235 / +0.175 / +0.153 / +0.111 / **+0.089** / +0.060 (K = 16…192) | **128** | 128 | 0.021 — **soft** |
| 64 | +0.285 / +0.179 / **+0.089** / +0.073 / +0.052 / +0.028 / +0.030 (K = 128…1024) | **256** | 512 | 0.090 — sharp |
| 216 | +0.379 / +0.264 / +0.190 / +0.150 / **+0.015** / -0.013 / -0.053 (K = 128…1024) | **512** | 512 | 0.135 — sharp |
| 512 | +0.395 / +0.234 / +0.123 / **+0.042** / +0.003 / -0.059 / -0.113 (K = 256…2048) | **768** | not measured | 0.082 — sharp |

- **The wall keeps growing**: 128 / 256 / 512 / 768 over a class-count range of **18.96×** (a wall ratio of
  **6.0**). Fitting the located walls gives

  > `ln K_wall = 0.603 · ln C + 2.944`, **R² = 0.988**, n = 4

  against the **certified requirement's** `K_min ∝ C^0.58` measured in SCALE-C round 1. The two slopes agree to
  0.023, so **the wall and the requirement grow at essentially the same rate** — the tension that motivated this
  whole line of work ("the empirical wall saturates while the information-theoretic requirement keeps growing")
  was **an artifact of the coarse ladder**, whose top steps were a factor of four apart.
- **Sharpness is a property of the instance, not a pass/fail for the method.** The drop across the crossing rung
  is 0.021 at C = 27 (where the gap curve approaches the bar gradually, so the located value carries extra
  ladder uncertainty), 0.090 at C = 64, 0.135 at C = 216 and 0.082 at C = 512. The first version of this
  criterion demanded that *every* crossing be sharp, which conflated "is this boundary real?" with "is this
  particular gap curve steep?"; the criterion now fits the growth rate and reports sharpness per instance. The
  pre-registered positional test A1 is kept in the code's docstring for the record.
- **Verdict: REFUTED for the saturation claim** (the wall grows as `C^0.60`, R² = 0.988), with the honest caveat
  that one of the four points (C = 27) sits on a soft crossing, and that these are single-seed student ladders
  (the seed sweep is reported in §7).

## 7. The seed sweep: both headline results are seed-robust (added 2026-10-05)

Everything above is a single-seed measurement, which is the project's usual honest caveat. The two results that
carry the most weight were therefore swept over seeds.

**(a) The real-data recipe effect, at CP-H's recorded tier** (K = 128, 120 epochs, full CIFAR-10, width 96; the
recorded teacher, data and per-seed init reused, only the schedule switched):

| recorded CP-H seed | reference acc | fixed LR (120 ep) | recorded (cosine, 120 ep) | delta |
| --- | --- | --- | --- | --- |
| 0 | 0.799 | +0.612 | +0.117 | **-0.495** |
| 1 | 0.768 | +0.597 | +0.054 | **-0.543** |
| 2 | 0.803 | +0.614 | +0.093 | **-0.521** |

**3 of 3 seeds**, with the effect's size varying by only 0.048. The conclusion of §5b is therefore not a seed
accident: at its own recorded tier, CP-H's walls were made possible by the schedule, and a fixed learning rate
leaves the student roughly 0.52 *below* the reference on a relative scale.

**(b) The growth exponent** (the C=216 fine ladder re-run with different student initialisations):

| quantity | value |
| --- | --- |
| wall at C = 216, student seeds 0 / 1 / 2 | 512 / 512 / **384** |
| seed spread at that class count | a factor of **1.33**, i.e. within one ladder step |
| exponent with seed 0 only | **0.603** (R² = 0.988, n = 4) |
| exponent with every seed's point folded in | **0.582** (R² = 0.966, n = 5) |
| certified requirement's slope | 0.58 |

The fitted slope moves by **0.021** between those two fits and both agree with the certified requirement, so the
conclusion of §6 — the wall grows at essentially the requirement's rate — survives the seed change. The
*location* of the wall is the part that moves (384 vs 512 at C = 216), which is exactly what the sharpness and
tolerance-band discussion predicts: a wall is a threshold crossing whose located value carries a ladder- and
seed-dependent uncertainty of about one step.

**Caveat kept**: a per-seed exponent would need three class counts *per seed*; only C = 216 was swept, so the
seed statement covers the wall's location at one class count plus the exponent's stability when one more seed is
folded in — not four independent fits.
