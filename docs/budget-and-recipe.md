# BUDGET DOSE-RESPONSE: the wall's budget dependence is an interior optimum, not a decay law

**Why this run exists.** WALL-SATURATION measured the student's retention deficit at two budgets on two
instances and reported ratios of 0.55 (C=216, sigma=0.5, K=4096) and 0.64 (C=1000, sigma=1.0, K=2048); SCALE-C
§7.4 read the second as confirming the first. Two points per instance is a **ratio, not a law**. This campaign
turned it into a dose-response curve — several budgets on the same instance, a power-law fit, and a test of
whether one task's exponent predicts another's deficit.

**Verdict: REFUTED — the deficit is not a power law in the budget, and it is not monotone either.** The
follow-up check then refuted the obvious explanation (overfitting to the cached soft targets) as well, leaving
an optimization-trajectory effect as the surviving reading. The wall's protocol-relativity is therefore *worse*
than the earlier notes said: not "more budget helps" but "there is an interior optimum, and it is not
predictable from the budget alone".

## 1. The curves

| instance | K | 60 ep | 120 ep | 180 ep | 240 ep | 360 ep | 480 ep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **I1** C=64, sigma=0.5 | 1024 | 0.0554 | 0.0382 | — | 0.0355 | — | 0.0296 |
| **I2** C=216, sigma=0.5 | 4096 | 0.1207 | — | **0.0685** | — | **0.1087** | — |
| **I3** C=64, sigma=0.5 | 256 | 0.0361 | — | — | 0.0269 | — | — |

I1 flattens (the 120 -> 240 -> 480 steps buy 0.009 total), and **I2 is non-monotone**: the deficit halves from
60 to 180 epochs and then **regresses by 60% at 360** — with the 60- and 180-epoch rungs being exact reuses of
WALL-SATURATION's checkpoints, so those two numbers replicate that campaign (0.1207 vs 0.124; 0.0685 vs 0.068).

## 2. The pre-registered criteria

| criterion | result |
| --- | --- |
| P1 shape: log(rel) affine in log(epochs), R² >= 0.90 per instance | I1 **PASS** (b = -0.282, R² = 0.917), I3 "PASS" on **2 points** (R² = 1.000 by construction — a flaw in the criterion, recorded), **I2 FAIL (R² = 0.090)** |
| P2 shared exponent: I1 vs I2 within 0.15 | **FAIL**: -0.282 vs -0.100, |diff| = 0.182 |
| P3 predictive: I1's fit predicts I2 at 360 epochs within 30% | **FAIL**: predicts 0.0312, observed 0.1087 (**71% error**) |
| P4 extrapolated budget to reach rel = 0.10 | reported for the record, but **not credible**: I1/I3 are already below the bar, and I2's fit runs through a non-monotone curve |
| **verdict** | **REFUTED: the dose-response is not a power law at this resolution** |

So the two-point ratios of the earlier campaigns were exactly what they looked like: two-point ratios of a
non-monotone quantity. The 60 -> 180 improvement is real and replicates; the generalization to "3x the budget
halves the deficit" is not.

## 3. The check that refuted the obvious explanation

The natural mechanism for a regression at long budgets is overfitting: the students distil from a *finite*
cached sample of the teacher's soft targets (150 samples/class), so more epochs could fit that cache while
generalising worse. The check evaluates the saved checkpoints of all three rungs on the **train** and **test**
splits (zero training):

| budget | train acc | test acc |
| --- | --- | --- |
| 60 epochs | 0.8444 | 0.8400 |
| **180 epochs** | **0.8849** | **0.8898** |
| 360 epochs | 0.8532 | 0.8514 |

The 360-epoch student is worse **on the training set as well** (0.8532 vs 0.8849), so overfitting to the cache
is **not** the explanation — it is refuted, not merely doubted. What remains is an **optimization-trajectory
effect**: the students train with a fixed learning rate (1e-2, no schedule) plus the VQ's codebook updates and
dead-code restarts, and with that setup a longer run lands in a worse solution on both splits. (The specific
contributor — LR instability, codebook drift, or the restart schedule — is not identified here; that is the
natural follow-up, and it is a property of the training recipe rather than of the task.)

## 4. What this changes

1. **The wall's protocol-relativity stands and is sharper**: `K_wall` does not merely depend on the budget — it
   has an **interior optimum in it**. Any statement of the form "with more training the wall would fall" is
   wrong as stated, and a sizing recommendation has to carry the actual training recipe, not a budget trend.
2. **The earlier two-point claims are corrected here**: E-018's "3x the training budget halves the deficit" and
   SCALE-C §7.4's "budget-scaling hypothesis confirmed" are both now recorded as unsupported as laws; the
   underlying single-pair measurements stand (and replicate), the generalization does not.
3. **A criterion flaw is recorded**: P1 let a 2-point instance pass with R² = 1.000 by construction. The next
   version of this design should require >= 3 rungs for any fitted shape; that is written down rather than
   quietly patched, because the family's whole method rests on criteria being checkable.
4. **The practical answer to "how much budget does the student need"** is therefore: search it (or validate the
   chosen rung against a held-out split), do not extrapolate it from a trend.

## 5. Honest boundaries

- Three instances, all in the lattice family, all at sigma = 0.5, one seed each; the budget grid is coarse
  (x2 steps for I1, x3 steps for I2) and the regression at 360 epochs is a single point.
- The rungs are exact **prefixes** of one another in the RNG sense (same seed, same shuffle order), so the
  non-monotonicity cannot be blamed on different initialisations; it is the training trajectory itself.
- The optimization-trajectory explanation is a *residual* (the overfitting alternative was tested and rejected),
  not a measured mechanism; the contributors listed are candidates.
- Ledger: `ledger.md` E-023. Artifacts: `experiments/data/budget_dose.jsonl`,
  `budget_dose_report.txt`, `budget_dose/`.

## 6. The mechanism, attributed (added 2026-10-05, `experiments/budget_mechanism_demo.py`)

Section 5 left "an optimization-trajectory effect" as a residual with named candidates. This section removes
the residual: a **2x2 factorial** on the same instance (C = 216, sigma = 0.5, K = 4096), switching **only** the
learning-rate schedule (fixed 1e-2 vs cosine annealing to 0) and the **dead-code restart** (on vs off), with
seed, data, teacher, soft targets, architecture, optimiser, K and budgets held fixed. The baseline cell is
exactly BUDGET-DOSE's I2, so its three rungs are reused rather than retrained, and the local training loop is
verified in `--selftest` to reproduce the package-verified replica bit for bit under the default setting
(2.6089 vs 2.6089) while the restart knob genuinely changes the trajectory.

| recipe | 60 epochs | 180 epochs | 360 epochs | monotone in the budget? |
| --- | --- | --- | --- | --- |
| **fixed LR + restarts ON** (baseline) | +0.1207 | **+0.0685** | +0.1087 | **NO** (regresses by 0.040) |
| **cosine + restarts ON** | **-0.0076** | **-0.0158** | **-0.0189** | **YES**, and the deficit is *negative* |
| fixed LR + restarts OFF | +0.8716 | +0.8475 | +0.8153 | "yes" — but the student has collapsed (acc 0.12-0.18) |
| cosine + restarts OFF | +0.9097 | +0.8425 | +0.8535 | "yes" — same collapse (acc 0.09-0.15) |

**Verdict SUPPORT: the fixed learning rate is the cause.** With cosine annealing the deficit is monotone *and*
negative at every budget — the K = 4096 student **matches or beats the reference** (acc 0.963 / 0.970 / 0.973
against the reference's 0.955) where the fixed-LR student sat 7-12% below it.

Two things this table also settles:

- **The dead-code restart is not optional** under this objective: switching it off collapses the student to
  chance (acc 0.09-0.18) regardless of the schedule. That is the bottleneck design's own choice, now measured
  under the distillation objective rather than assumed.
- **A criterion flaw is recorded.** The pre-registered R3 ("restarts OFF restores monotonicity") is satisfied
  *vacuously* by the collapsed cells: a ruined model improves slowly and monotonically. The attribution should
  therefore be read as **the fixed LR**, not as "both factors"; the criterion as written cannot distinguish
  "fixed the problem" from "destroyed the model". The guard was then implemented rather than left as a caveat:
  a cell is credited only if its best deficit is within 0.30 of the baseline cell's best, and with it the
  verdict prints **"SUPPORT: the non-monotonicity is attributable to the fixed learning rate"** and marks R3 as
  *not credited*. The next version of this design should carry that floor from the start.

## 7. The consequence: the located wall moves by >= 8x with the recipe (added 2026-10-05)

If the recipe inflated the deficits, the walls located with it are inflated too — so the same K ladder was run
under the good recipe at the same budget (180 epochs, cosine + restarts ON, same reference):

| K | 32 | 128 | **512** | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- |
| deficit, cosine recipe | +0.8624 | +0.4376 | **+0.0019** | -0.0084 | -0.0158 |
| deficit, fixed-LR recipe (180 ep) | — | — | +0.17 (at 60 ep) | +0.19 (at 60 ep) | **+0.0685** |

- **Located wall under the good recipe: K = 512.** Under the fixed-LR recipe this instance had **no wall at
  all**: at 60 epochs the best rung (K = 4096) was still 12% short, and at 180 epochs only K = 4096 crossed the
  bar. So the practical answer moves from "≥ 4096, unlocatable" to **512** — a change of **>= 8x** produced by
  the learning-rate schedule alone, at fixed task, K, budget, seed and architecture.
- At K = 512 the deficit goes from **+0.17** (fixed LR, 60 ep) to **+0.0019** (cosine, 180 ep): the student
  matches the reference exactly, and the wall is located at a codebook **eight times smaller**.
- **Consequences that must be carried into the project's claims.** (i) Every wall this project located before
  this note used the fixed-LR recipe — **but `wall-remeasurement.md` has since tested that reading and
  refuted it as a generalization**: at the SCALE-C instances the walls are recipe-robust (C = 27/64/216
  unchanged, C = 8 *worse* under cosine at the same short budget), so the >= 8x effect here is specific to this
  instance (an easier task where the students can converge inside the budget). The corrected statement is the
  **interaction** one: annealing needs a budget long enough to converge. (ii) The certified side is untouched:
  `R_IB`, `K_min` and the exclusion rule do not depend on any student recipe, which is exactly why the headline
  rule has survived every campaign. (iii) The practical rule to add to the README: **choose the schedule
  together with the budget — anneal only when the run is long enough to converge — and never measure a wall
  with a recipe you have not checked against its budget.**

**Honest scope.** One instance for the relocation (C = 216, sigma = 0.5, 180 epochs), one instance for the
factorial, one seed each. "cosine is better" is demonstrated here as a *mechanism* (it removes the
non-monotonicity and the collapse of retention), not as a tuned optimum: no schedule search was run, and the
8x factor is instance-specific even though the direction is not.
