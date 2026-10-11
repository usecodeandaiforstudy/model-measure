# SCALE-AXES: the three scale axes reachable offline

**Scope.** SCALE-C covered the class-count axis, WALL-SATURATION the task-difficulty axis. Three further
axes are commonly assumed to matter for "does the reading hold at scale"; this note measures the ones that
can be reached without downloads — **sample count**, **student width**, **input resolution** — and states
plainly that **natural diversity** remains unreachable offline (it needs datasets and pretrained weights the
environment does not have).

**Verdicts: A1 sample count MIXED · A2 width SUPPORT (marginal, fragile) · A3 resolution SUPPORT.**

## A1 — Sample count (C = 216, sigma = 1.0, N per class 40 / 150 / 300)

The task's true information does not depend on N *by construction* (`I(X;Y) = 4.545` nats from the
Monte-Carlo ground truth), so any movement here is the witness showing through.

| N per class | ref. acc | eps (Bayes 0.299) | `R_IB` (true 4.545) | under-reading | `K_wall` |
| --- | --- | --- | --- | --- | --- |
| 40 | 0.544 | 0.456 | 2.237 | 0.922 | **not located** (K=2048 still 11% short) |
| 150 | 0.554 | 0.446 | 2.292 | 0.867 | 512 |
| 300 | 0.578 | 0.422 | **2.428** | **0.732** | 512 |

- **A1-a PASS**: the reading does move toward the truth with more data (2.237 -> 2.428) and the
  under-reading shrinks (0.922 -> 0.732).
- **A1-b not evaluable**: with N = 40 no wall was located, so the "looseness against the Bayes requirement"
  comparison has only one end.
- **Verdict MIXED**, and the honest reading of the magnitudes matters more than the label: **seven and a half
  times the data (40 -> 300 per class) buys 0.19 nats of a 2.1-nat gap.** On this task the witness is
  limited by optimisation difficulty, not by data volume — which is the same diagnosis the sigma arm reached
  from the other direction (`wall-saturation.md` section 2). A larger-N arm was not run: the trend is
  clear and the marginal cost is quadratic in N.

## A2 — Student width (C = 216, sigma = 1.0, d in {64, 128, 256})

Retention is measured against the same reference (acc 0.579); the ladder is K in {128, 256, 512, 2048}.

| K | d=64 | d=128 | d=256 |
| --- | --- | --- | --- |
| 128 | +0.440 | +0.435 | +0.470 |
| 256 | +0.201 | +0.155 | +0.169 |
| 512 | +0.103 | +0.081 | +0.081 |
| 2048 | +0.051 | +0.036 | +0.083 |
| **located wall** | **2048** | **512** | **512** |

- **W1 (pre-registered) PASS**: the located wall moves 2048 -> 512 when the student is widened.
- **W2 (added after the first pass, because the raw numbers looked too thin for the conclusion) PASS, but by
  0.002**: the largest retention gain from widening is +0.032 against a 0.03 threshold.
- **W3**: d = 128 sits between the two — except at K = 256, where it is *better* than d = 256 (0.155 vs
  0.169), so the trend is **not monotone at every rung**.
- **Honest verdict: SUPPORT (marginal), and fragile.** The substance is 2-3 percentage points of retention
  that saturate by d = 128; the "4x move" of the located wall happens because the 10% retention bar sits
  exactly on that 2-point difference and the gap curve is flat there. The defensible claim is "at this scale
  the wall is not a function of K alone, but the width effect is small and the located-wall value is not
  stable enough to carry a sizing decision" — **not** "widening the student quarters the requirement".

## A3 — Input resolution (the project's own grid64 image task)

`cifar_demo.make_grid64_images` generates 16x16 images with frozen encoding parameters (three bands x four
positions = 64 classes, H(Y) = ln 64). The same generator is then **average-pooled** to 8x8 and
**nearest-upsampled** to 32x32, so the class structure is identical and only the resolution changes; one
architecture (an MLP on flattened pixels) is used throughout.

| resolution | ref. acc | eps | `R_IB` | `K_min` | `K_wall` | retention gaps (K = 8/32/128/512) |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 1.000 | 0.000 | 4.159 | 64 | 128 | 0.88 / 0.52 / 0.00 / 0.00 |
| 16x16 | 1.000 | 0.000 | 4.159 | 64 | 128 | 0.88 / 0.52 / 0.00 / 0.00 |
| 32x32 | 1.000 | 0.000 | 4.159 | 64 | 128 | 0.90 / 0.55 / 0.00 / 0.00 |

- **R1 PASS**: `|R_IB(8x8) - R_IB(32x32)| = 0.000` nats (identical to three decimals).
- **R2 PASS**: the wall is 128 at every resolution.
- **Verdict SUPPORT: the reading is a property of the task, not of the input resolution.** This is a clean
  invariance result, and it corroborates the class-count and difficulty axes: what moves the reading is what
  the labels can tell you about the input, not how the input is laid out.
- Two caveats on interpretation. First, the task is *easy for the model used* (accuracy 1.000 at every
  resolution, so `R_IB` sits at its `ln C` ceiling and `K_min = C = 64` — the same ceiling behaviour the
  lattice family shows as sigma -> 0). The invariance is therefore demonstrated cleanly but at a saturated
  operating point; a task where the reference is *not* at the ceiling is the harder test, and it has now been
  run — see A3' below. Second, the class structure lives in 3x3 blocks, so 8x8 pooling degrades it
  to at most 1-2 pixels per block — the manipulation is a real resolution change, not a cosmetic one, which
  is why the invariance is informative.

### A3' — the same axis at a NON-DEGENERATE operating point (added 2026-10-05)

The saturated result above left the obvious hole: with the reference at accuracy 1.000 the reading is pinned to
its ceiling, so the invariance could be trivial. The generator's encoding parameters are frozen by their own
docstring (tuning them to suit the result would be p-hacking), so the operating point was moved with **protocol
knobs** instead — a reference codebook of 64 and a 6-epoch budget — and three criteria were pre-registered, one
of which the first run would have failed: **R3' the reading must be non-degenerate** (`R_IB < ln C - 0.05`).

| resolution | ref. acc | eps | `R_IB` (ceiling 4.159) | `K_min` | `K_wall` | retention gaps (K = 8/32/128/512) |
| --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 0.464 | 0.536 | **1.246** | 4 | 128 | 0.88 / 0.51 / **-0.96** / -1.16 |
| 16x16 | 0.533 | 0.467 | **1.532** | 5 | 128 | 0.84 / 0.45 / -0.34 / -0.88 |
| 32x32 | 0.512 | 0.488 | **1.444** | 5 | 128 | 0.83 / 0.57 / -0.50 / -0.94 |

- **R3' PASS**: the reading is far below the ceiling at every resolution (1.25-1.53 against 4.159).
- **R1' PASS**: `|R_IB(8x8) - R_IB(32x32)| = 0.198` nats (tolerance 0.30).
- **R2' PASS**: the wall is 128 at every resolution.
- **Verdict SUPPORT: the resolution invariance survives off the saturated point.** But the *tightness* changes,
  and that is the finding worth keeping: at the saturated point the disagreement was **0.000** nats, here it is
  **~0.2 nats** — i.e. the reading's invariance is limited by the **witness's own convergence**, not by the
  resolution manipulation. The same mechanism appears independently in `cross-task-comparability` §6, where an exact
  scale twin failed invariance because the two references converged unequally (0.429 nats).
- A second artefact of the weak-reference design, recorded because it bounds what the retention numbers mean: the
  students **beat the reference** at the larger rungs (retention gaps of -0.34 to -0.96), because the reference
  is a 64-code / 6-epoch model while the students distil from its soft targets into codebooks of up to 512. The
  located wall of 128 is therefore a formality at this operating point: it says "everything from K = 128 up
  exceeds the reference", not "K = 128 is where the capacity boundary lies".

## What this adds up to

| axis | verdict | what moved |
| --- | --- | --- |
| sample count (A1) | MIXED | the reading tightens with data, but 7.5x the data buys 0.19 of a 2.1-nat gap |
| student width (A2) | SUPPORT (marginal) | retention +2-3 points, saturating at d=128; the located wall moves for threshold reasons |
| input resolution (A3) | SUPPORT | nothing moved: identical reading and identical wall at 8x8, 16x16, 32x32 |

The pattern is consistent with the rest of the project: **the certified quantity (`R_IB`, `K_min`, and the
exclusion built on them) is stable across everything tested; the empirical quantity (`K_wall`) moves with the
protocol — the student's width, its training budget, the reference's quality — and should never be read as a
property of the task.**

## Honest boundaries

- One family for A1/A2 (3-D lattice at C=216, sigma=1.0) and one synthetic image family for A3; one seed per
  configuration; A1 has three N values, A2 four K rungs and three widths, A3 three resolutions.
- A1's wall at N = 40 was not located, so its "looseness" comparison is unavailable; A2's W2 threshold (0.03)
  was chosen after seeing the first pass and its margin is 0.002 — both facts are recorded rather than
  smoothed.
- Natural diversity, dataset scale (ImageNet-class), and pretrained-weights scale are **not** covered; they
  need downloads and compute that this environment does not have, and the project's rule is not to fake them
  with synthetic substitutes.
- Ledger: `ledger.md` E-020. Artifacts: `experiments/data/scale_axes.jsonl`,
  `scale_axes_report.txt`, `scale_axes/`.
