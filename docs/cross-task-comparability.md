# CROSS-TASK: is there a common difficulty scale? (the ledger's P-A1-2)

**Why this note exists.** The A-001 record contains one item that was never given a positive design:
**P-A1-2**, the hope that the requirement reading `R_IB` could place *different tasks* on one scale ("task X
needs twice the capacity of task Y"). Everything measured since argued the reading is witness-relative
(WITNESS, PRETRAINED-WITNESS, FAMILY-LIBRARY), but the argument was never tested against a known ground
truth. The lattice family makes that possible: its classes are equal-weight Gaussians with known centres, so
`I(X;Y)`, the Bayes error and the Bayes-implied requirement are computable exactly by Monte Carlo.

**Verdict: SUPPORT (ordinal only).** The reading **is** a function of the task's information — it survives a
re-parameterization that leaves the information unchanged, and it orders tasks the way the ground truth does
— but the readings are **not commensurable**: the gap between the Bayes requirement and the reading is
task- and witness-dependent, so `R_IB` values from different tasks may be ranked but not compared as numbers.

## 1. The ground truth, and the scale-equivalence construction

Spacing `s` and noise `sigma` enter the likelihood only through the ratio `sigma/s`, so **`(sigma 0.5, s 3)`
and `(sigma 1.0, s 6)` are the same task** up to a global rescaling of the input. The Monte-Carlo table
confirms it: at C = 216 both carry `I(X;Y) = 5.356` nats (identical to three decimals) with
`eps_bayes = 0.0069` and `K_min(bayes) = 200`. That pair is the invariance test: two different
parameterizations, one task.

The same table also shows how strongly the requirement grows as a task becomes learnable
(`experiments/data/cross_task.jsonl`, all rows): at C = 216, `I(X;Y)` and `K_min(bayes)` go

| sigma | 1.00 | 0.70 | 0.50 | 0.35 |
| --- | --- | --- | --- | --- |
| `I(X;Y)` (nats) | 4.545 | 5.161 | 5.356 | 5.375 |
| `K_min(bayes)` | 24 | 109 | 200 | 216 = C |

i.e. a cleaner task needs *more* codebook, up to the `K = C` ceiling where every class needs its own cell —
the ceiling that makes `K >= C` in-excludable.

## 2. The measurement

Four configurations at C = 216, identical protocol (reference VQ(2048), students up to K = 4096, d = 64,
60 epochs, 150 samples/class, seed 0):

| configuration | true `I(X;Y)` | reference acc | `R_IB` reading | `K_min` | `K_wall` |
| --- | --- | --- | --- | --- | --- |
| sigma 1.00, s 3.0 | 4.545 | 0.554 | 2.292 | 10 | 512 |
| sigma 0.70, s 3.0 | 5.161 | 0.813 | 3.892 | 49 | 512 |
| sigma 0.50, s 3.0 (B) | 5.356 | 0.955 | **4.952** | 142 | not located |
| sigma 1.00, s 6.0 (C) | 5.356 | 0.934 | **4.777** | 119 | not located |

- **X1 invariance: PASS.** The two scale-equivalent tasks read `4.952` and `4.777` — a difference of
  **0.175 nats** on tasks whose true information differs by **0.000**. The reading tracks the task, not the
  parameterization.
- **X2 ordering: PASS.** Over the four points there are **0 material inversions** against the true
  information (Spearman +0.949, the non-unit value coming from the exact tie between B and C, which the
  tie-robust criterion ignores by construction).
- **X3 commensurability: FAIL.** The under-reading `rib_bayes - rib` is `0.87 / 0.80 / 0.34 / 0.52` nats at
  true information `4.54 / 5.16 / 5.36 / 5.36` — it is **not a constant offset** (fitted slope -0.50). And
  across the class-count axis the same quantity grows instead (0.31 -> 0.91 -> 1.34 -> 2.25 -> 4.00 nats for
  C = 8 -> 1000, see `wall-saturation.md` section 1). The offset is therefore a joint property of the
  task and the witness, with no single sign.

## 3. What may and may not be claimed (the answer to P-A1-2)

**May be claimed.** "Task A requires more capacity than task B" — *provided* both readings come from
witnesses of comparable quality on comparable protocols. The ordering survived every configuration tested,
including one where the parameterization was changed and the information held fixed.

**May not be claimed.** "Task A requires twice the capacity of task B." The numbers are lower bounds whose
tightness varies with the witness: on this family the reference under-reads `K_min` by 1.0x / 1.2x / 1.5x /
2.4x / 7.4x across C = 8 -> 1000 at sigma = 1.0, and by 0.5-1.4x across the sigma axis. A ratio of two
readings is a ratio of two *witness qualities* as much as of two tasks.

**The practical rule** that follows is the one the FAMILY-LIBRARY campaign already used: predict a reading
for a new member of the same family from a cheap descriptor, and never compare readings across families with
different witnesses. P-A1-2 is thus closed as **"a common scale exists, but only up to order"** — with the
caveat, now removed by section 5, that this was established on one family whose ground truth is computable.

## 5. Second family: random-centre Gaussian mixtures (added 2026-10-05, `experiments/family_b_demo.py`)

The lattice's geometry is extremely regular (equal spacing, isotropic noise, all classes exchangeable), so the
obvious objection to section 1-3 is that the conclusion might be a property of that regularity. **Family B**
answers it: `C = 216` centres drawn i.i.d. from `N(0, spread^2 I_3)`, equal weights, isotropic noise — the same
closed-form ground truth, an irregular geometry. The single draw used here has nearest-centre distances of
**0.21 / 1.04 / 6.43** (min / median / max), i.e. some classes are near-twins (a pair 0.21 apart against a noise
scale of 0.5) while others are isolated; nothing about the draw was tuned, and it is the one seed 0 produces.

| tag | configuration | true `I(X;Y)` | Bayes eps | reference eps | `R_IB` reading | under-reading | `K_wall` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | sigma 0.5, spread 3 | 4.603 | 0.290 | 0.380 | 2.673 | 0.542 | 512 |
| B2 | sigma 1.0, spread 3 | 3.268 | 0.645 | 0.705 | 0.984 | 0.274 | 512 |
| B3 | sigma 1.0, spread 6 | **4.603** | **0.290** | 0.391 | **2.606** | 0.608 | 512 |
| B4 | sigma 0.5, spread 6 | 5.226 | 0.061 | 0.110 | 4.441 | 0.381 | 2048 |

- **Y1 invariance PASS, and tighter than the lattice's.** B1 and B3 are **exact** scale twins (every centre and
  the noise scaled by 2 leaves all posteriors identical: the ground truth agrees to the fourth decimal,
  `I = 4.603` both, `eps_bayes = 0.2902` both), and the readings differ by **0.067 nats** — against the
  lattice's 0.175. Invariance under re-parameterization survives on an irregular family.
- **Y2 ordering PASS**: 0 material inversions over the four points (Spearman +0.949, the non-unit value coming
  from the exact twin pair).
- **Y3 cross-family comparability FAIL.** At matched true information — the pairs within 0.07 nats —
  family B's under-reading is **0.54-0.61** nats where the lattice's is **0.87-0.80**: a **0.26-0.42 nat
  offset** between the families. So the reading is a function of the task *within* a family and carries a
  **family-specific bias**; **readings from different families are not comparable even ordinally.**
  (One row of the machine-generated table, B2 -> lattice, used a match 1.28 nats away and reported a 0.59-nat
  gap; that pair is not a matched comparison and should not be quoted — the well-matched gaps are 0.26-0.42.)

**The verdict for P-A1-2 is therefore MIXED, and that is the final word for now**: a common scale exists
**inside** a family (invariance and ordering both held on two structurally different families), and it does
**not** extend across families. That is a stronger statement than section 3 could make, because section 3 only
knew about one family and could not separate "the offset varies with the task" from "the offset varies with the
family".

> Pre-registration note, recorded because it happened before any measurement: the first draft of this
> experiment's configuration list wrote B3 as the twin with `sigma=0.5, spread=6.0`. The zero-training step-0
> table immediately showed that is not a twin (true information 4.603 vs 5.226 nats). The configuration was
> corrected to `sigma=1.0, spread=6.0` **before any training run**; the criterion Y1 was not touched, and the
> extra point was kept as B4.

## 4. Honest boundaries

- One family (3-D lattice), one protocol, seed 0 per configuration, four configurations in the arm plus 20
  in the ground-truth table. The invariance test has one pair; a stronger version would sweep several
  `(sigma, spacing)` pairs along the same ratio.
- "Cross-task" here means *within one family, across parameterizations*. Cross-family comparability (MNIST
  vs CIFAR-10 vs lattice) has no computable ground truth offline and remains untested; the consistency
  evidence from the other campaigns is qualitative.
- The two configurations whose wall was not located at K = 4096 are recorded as such: on a task a strong
  reference can actually fit, the students plateau before reaching the retention bar (`wall-saturation`
  section 3 shows that the plateau is budget-limited), so "wall not located" here means "the protocol's
  retention bar was not reached", not "the requirement exceeds 4096".
- Ledger: `ledger.md` E-019. Artifacts: `experiments/data/cross_task.jsonl`,
  `cross_task_report.txt`, and the arm rows inside `wall_sat.jsonl` (tag `cross`).

## 6. Third family: high-dimensional mixtures (d_in = 16), and a condition on the invariance (added 2026-10-05)

Both earlier families have a **3-dimensional input**, which the project flagged early as a possible confound:
with three input dimensions, how many regions K codes can carve out is constrained by the input geometry rather
than by the label count. Family C removes exactly that: C = 216 centres drawn i.i.d. from `N(0, spread² I_16)`,
equal weights, isotropic noise, the same closed-form ground truth, and the same exact scale twin (scale every
centre and the noise together).

**The configuration search is itself a result.** With the first candidate grid (sigma <= 1.0) **every**
configuration sat at the ceiling — Bayes error ~ 0, `I(X;Y) = ln 216 = 5.375`, `K_min(bayes) = 216`. That is
**distance concentration**: random centres separate like `sqrt(2·d)·spread` while the noise stays `sigma`, so in
16 dimensions a mixture that is hard in 3 dimensions becomes trivially separable. The grid was widened (sigma up
to 6.0) **before any reading was measured**, and the pre-registered band rule was not changed.

| tag | configuration | true `I(X;Y)` | Bayes eps | reference eps | reading | under-reading | `K_wall` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | sigma 1.0, spread 1.5 | 5.243 | 0.043 | **0.176** | **3.963** | 1.003 | not located (plateau at ~18%) |
| C2 | sigma 2.0, spread 3.0 — **exact twin of C1** | **5.243** | 0.043 | **0.240** | **3.534** | 1.432 | not located (plateau at ~16%) |
| C3 | sigma 2.5, spread 3.0 | 4.884 | 0.149 | 0.515 | 1.919 | 2.236 | 512 |

- **Z1 invariance FAILS: |dR_IB| = 0.429 nats on two parameterizations whose true information is identical**
  (5.243 in both). In families A and B the same test gave 0.175 and 0.067. The reason is visible in the table:
  the **task** is identical but the **reference's convergence is not** (eps 0.176 vs 0.240), and the reading
  inherits the witness's quality — the same mechanism that made the resolution axis wobble by ~0.2 nats when its
  witness was under-trained (`scale-axes` A3').
- **Z2 ordering PASSES** (0 material inversions over the three points).
- **Z3 cross-family gaps grow**: at matched information the offsets against family B reach **1.05 nats** and
  against family A **1.43 nats**, against 0.26-0.42 for the A-B pair — a third, independent confirmation that
  readings from different families are not comparable.

**Verdict REFUTED (Z1 failed) — and this is the most useful outcome of the three families**, because it turns
the earlier positive statement into a conditional one:

> **P-A1-2, final form.** A common scale exists **within** a family, in the ordinal sense, **provided the two
> readings come from witnesses of comparable quality**. The invariance under re-parameterization is therefore
> *conditional*, not automatic: families A and B passed it because their twins happened to converge equally
> well, and family C failed it because they did not. The condition is checkable in practice (compare the two
> witnesses' errors), which makes the rule actionable: **check the witnesses first, then compare readings.**
> Across families, no scale exists at all — the offsets are family-specific and now measured up to 1.4 nats.

This also explains why the project's headline exclusion rule is unaffected: it never compares readings across
witnesses, it compares one reading against a codebook size.
