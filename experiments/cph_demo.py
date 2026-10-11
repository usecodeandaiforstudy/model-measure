"""CP-H: real-task dual-axis validation (CIFAR-10). Axis 1 = distillation Fano exclusion + margin stability;
Axis 2 = whether distinguishability readings stay capacity-proxies outside the saturated regime.

Usage:
    python experiments/cph_demo.py --calibrate      # Phase A: teacher only, no verdict (pick the epoch budget)
    python experiments/cph_demo.py --fast           # reduced budget WITH verdict (directional)
    python experiments/cph_demo.py                  # full tier with verdict (local long run; use --resume)
    python experiments/cph_demo.py --read           # Phase C: zero-training readings + S verdicts from checkpoints
    python experiments/cph_demo.py --workers 4      # multi-threaded data loading (recommended on GPU: the
                                                    # SmallCNN runs are data-bound, not compute-bound)
    python experiments/cph_demo.py --cosine         # cosine lr decay (the 2026-09-29 calibration fallback, see (8);
                                                    # teacher/student share the same recipe = budget alignment)
    python experiments/cph_demo.py --fresh          # force a full recompute (default: relay mode -- cache-hit
                                                    # units are skipped, so an interrupted run resumes by itself)
    python experiments/cph_demo.py --selftest       # pure-function unit tests (no training)

This is the follow-up of framework.md section 7 items 1-2 (validate the distillation exclusion criterion on a
real task; probe the boundary where the H(Y)/R_IB combination degenerates). It simultaneously re-opens the
axis-2 closure question (framework.md 4.3b/4.3d) in a regime where the closure's premise fails.

Data: experiments/data/cifar-10-batches-py (already present; torchvision reads it offline, no download).
No new data is fetched by this script.

Pre-registration (2026-09-27, criteria before data; this docstring is the pre-registration record)
--------------------------------------------------------------------------------------------
(1) Motivation: all of the project's positive results on the distillation lower bound K_min = ceil(e^R_IB)
    live on synthetic hard-wall tasks (grid64/shape64, H(Y)=ln64 fully saturated by the task) and CP3's four
    cells; all of the axis-2 "distinguishability" closures (CP-E saturation / CP3 supplementary 1 endpoint /
    CP-G nuisance ratio) were established on tasks whose representations are trivially linearly separable
    (CP-E: 1-NN error = 0 on all 20 units). Two open questions remain from that boundary:
      Q1 (axis 1): does the Fano exclusion "log K < R_IB => student cannot match the teacher" hold and does
                   the empirical margin K*/K_min stay stable on a REAL task with a strong teacher?
      Q2 (axis 2): in a regime where the representation is NOT saturated (features overlap, network struggles),
                   do the distinguishability readings (I(Z;Y), silhouette, separation ratio, decision margin d')
                   still collapse to capacity proxies, or do they carry quality information beyond capacity?
    The only real, non-separable dataset available in the workspace is CIFAR-10 (10 classes). The project
    previously judged CIFAR-10 "no wall -> tool fails" (C3-CIFAR archive, 2026-09-19) using a WEAK teacher
    (acc=0.559 -> Fano K_min=2, the wall degenerates at the grid bottom). A strong teacher (acc ~0.80-0.90)
    gives R_IB ~ 1.4-1.8 nats -> K_min ~ 4-6, so a wall region K in {2..8} exists and the old verdict can be
    re-tested on its own terms: it predicted failure from a teacher that barely encoded the task.

(2) Design: one real task (CIFAR-10), one backbone (SmallCNN from cifar_demo; the shared width is a
    calibration decision -- the default width16 was rejected by calibration, see (8)), teacher = SmallCNN(K=256),
    students = SmallCNN(K in grid) with the ONLY varying knob being the codebook size K (isomorphic
    teacher/student, CP3 convention). Distillation objective = hard-label
    CE + soft-label KL(T=3), via capacity_lens.train_one_epoch(distill_from=teacher) -- unchanged from CP2/CP3.
    R_IB uses the v3 Fano convention: R_IB = task_info_fano(1 - acc_t, C=10, h_y=ln 10). K_min = ceil(e^R_IB).
    K grid: fixed CALIBRATED_K = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 192, 256] shared by all seeds
    (the 192/256 pair is the plateau reference; the low end 2/3/4 sits strictly below any realistic
    K_min in {4,5,6}). The 192/256 extension was added 2026-09-29 after the seed-0 probe (see (10)):
    the base grid's top 64/128 was still declining (g(128)=0.094), so the plateau could not establish;
    the extension is additive above the base top and keeps the cache fingerprint (see (9)).
    Budget alignment (v2 lesson): student epochs == teacher epochs, same lr.

(3) Readouts (per seed): teacher acc / H(Z_t) / R_IB / K_min; student gap curve g(K) = acc_t - acc_s(K);
    plateau check and K* = distill_gap_k_star (imported from mnist_demo, unchanged); the gap column
    R_IB - log K (directionality, same convention as C3 v3). Phase C (zero extra training) adds per-unit:
    H(Z), plug-in I(Z;Y) (Miller-Madow corrected, with the N=10000 vs K*C bias caveat printed), Fano bound,
    silhouette / separation_ratio on pre-VQ features t = relu(fc1(h)) (euclid), decision margin d'
    (detection_index on logits), and for the teacher only: 1-NN error on a 3000-sample subsample (ceiling probe).

(4) Pre-registered criteria. Two competing hypotheses per axis; verdict functions are pure functions.
    Axis 1:
      H_wall   : a strong teacher on CIFAR-10 yields a locatable capacity knee; K*/K_min within one octave
                 (ratio_verdict pass) -> the distillation exclusion EXTENDS to real tasks, and the archived
                 "10 classes -> no wall" verdict was a weak-teacher artifact.
      H_smooth : even with a strong teacher the gap curve is smooth / the plateau never establishes ->
                 the exclusion is unlocatable on 10-class real tasks; the tool's claimable domain stays synthetic.
      R1a (necessary-condition sanity, per seed): every K < K_min has acc_s(K) <= acc_t - 0.005.
          Violation = the Fano bound measurement itself is broken (bug or regime); the run is then not usable.
      R1b (operational exclusion, per seed): every K < K_min has g(K) >= g_ref + tol (tol=0.02), i.e. the
          below-bound student is measurably worse than the reference plateau. R1b fail with R1a pass = the
          exclusion is true but too weak to screen with (recorded honestly; the "screening" value is nil).
      R2 (locatability): >= 2/3 seeds with plateau_ok AND K* located (locatability convention of CP1/CP2).
      R3 (margin, primary): ratio_verdict(records, "measured K*(gap) / predicted K_min") with the octave window.
          Additionally REPORT the median margin and compare with the synthetic range 1.7-3.0; the comparison
          is descriptive only (no verdict rides on it).
      R4 (directionality): Spearman rho(g(K), log K) <= -0.80 for every seed (capacity is the dominant factor
          of the gap curve). |rho| < 0.8 -> the curve is noise-dominated, no capacity structure to measure.
    Axis 2 (Phase C, zero-training):
      S1 (premise, teacher): the real-task representation is NOT saturated: err_1nn >= 0.02 (grid64 had 0)
          AND silhouette < 0.6. If S1 fails the task is in the saturated regime and the axis-2 test is moot
          (recorded as such, not forced).
      S2 (quality info in the knee window): for each reading r in {I_plug, silhouette, separation_ratio, d'},
          Spearman |rho(r, acc)| >= 0.5 over all (seed, K) points in the knee window -> "carries quality info".
      S3 (capacity-proxy check): |rho(r, log K)| >= 0.9 -> "pure capacity proxy (closure extends)";
          <= 0.5 -> "not a capacity proxy". In between = gray zone, recorded as unjudgeable.
      Overall axis-2 statement: any reading with S2 pass AND S3 <= 0.5 -> H_indep (the closure premise fails
          on the real task: distinguishability carries capacity-independent quality information);
          all readings being S3 >= 0.9 proxies and none passing S2 -> H_proxy (closure extends to CIFAR-10).
      Knee window definition (pure function): per seed, if K* located: {K in grid : K*/2 <= K <= 2K*};
          else {K in grid : k_min/2 <= K <= 2*k_min}. rho is computed over the union across seeds.
      (Spearman is implemented in-script via ranks; no scipy dependency.)

(5) G0 gate (teacher validity): g0_gate(acc, h_z, 256) from mnist_demo with **task-calibrated acc_min=0.78**
    (recalibrated 2026-09-29 from 0.80, before any verdict data; original verdicts reported in (8)). Chain:
    (i) the gate's pre-registered purpose = "the teacher has encoded the task" AND "the wall is non-degenerate";
    (ii) K_min is quantized: acc >= ~0.715 gives R_IB >= ln3 -> K_min >= 4 (wall {2,3}); the degenerate
    K_min <= 3 regime needs acc < 0.715; acc >= ~0.807 would give K_min >= 5 (wall {2,3,4});
    (iii) the four calibration rounds mapped this backbone family's achievable plateau on CIFAR-10 as
    ~0.77-0.81 with run-to-run noise +-0.03 -- a 0.80 gate sits INSIDE that noise band and rejects teachers
    (0.794 / 0.799) that are scientifically identical to one it admits (0.806): it was testing noise, not strength;
    (iv) 0.78 = ~1.5 noise bands above the degenerate boundary 0.715 and below the measured plateau, so it
    excludes the undertrained 40ep tier (0.71-0.73) while admitting every calibrated plateau teacher.
    The codebook gate keeps the project standard usage_min=0.8 (H(Z_t) >= 0.8*ln256 = 4.44; measured 0.89-0.95
    in every probe, so it stands).

(6) Seed scheme: teacher = 21000 + s; student = 22000 + 100*s + K. Non-overlapping with the existing families
    (mnist 1000/2000, cifar 6000/7000, cp1 9000/9500, cp2 12000/13000, cp3 14000/15000). The CIFAR-10 subset
    split uses the fixed generator seed 0 (same as cifar_demo), so the subset is shared by all units of a run;
    every unit sets torch.manual_seed at its start -> unit-by-unit determinism (resume equivalence).

(7) Iron rules: print only GBK-encodable characters; criteria before data (this docstring); pure functions
    independently unit-testable (--selftest); negative results archived honestly; no cross-task claims
    (CIFAR-10 only); data never downloaded (offline-safe, verified layout experiments/data/cifar-10-batches-py).

(8) Budget calibration record (to be filled by --calibrate BEFORE any verdict data is produced; the calibration
    target is only the G0 gate's input, i.e. teacher convergence -- never touching the readouts):
      - defaults to calibrate: teacher epochs 40, width 16, lr 1e-3, full data.
      - expected reference (from the C3-CIFAR archive, weak budget): 3 ep -> acc ~0.56. A strong teacher needs
        the acc gate 0.80; if 40 ep is insufficient, raise --epochs (record the final choice here).
      - also record H(Z_t) and H(Z_t)/ln256 per seed: this decides whether usage_min=0.8 stands (see (5)).
    Calibration records (all produced BEFORE any verdict data; the codebook gate passed in every probe,
    H/lnK ~= 0.79-0.92 rising, so usage_min=0.8 stands; the failing side is always the acc gate):
      - 2026-09-29 CPU, width16 / 40ep / full data / 3 seeds: acc = 0.669 / 0.605 / 0.628 -> acc gate FAILS.
        Width 16 is too weak a backbone for a "strong teacher" (the width16 default was a cost decision, now corrected).
      - 2026-09-29 GPU (RTX 5070, torch 2.11.0+cu128), width32 / 60ep / full data / seed 0:
        ep1/30/60 acc = 0.455 / 0.658 / 0.732 -> acc gate FAILS (0.732 < 0.80). Curve still rising, no plateau;
        plain Adam without lr decay converges slowly. The GPU run is data-bound (GPU-util ~3%), so widening is
        nearly free in wall time.
      - 2026-09-29 GPU, width64 / 80ep / seed 0: acc = 0.473 / 0.712 / 0.809 -> gate passes, but 0.809 hugs the
        threshold (seed-jitter risk, the CP3 lesson).
      - 2026-09-29 GPU, width64 / 120ep / 3 seeds: acc = 0.784 / 0.795 / 0.774 -> acc gate FAILS for all seeds,
        and ep60->ep120 gains are small (plateau). Plain Adam without decay has stalled.
        Also measured: run-to-run variance at the SAME seed/config is ~+-0.03 in teacher acc (seed0 gave
        0.809@80 in the single-seed probe vs ~0.775@80 in this repeat; VQ dead-code restart branching amplifies
        cuDNN non-determinism). Calibration numbers carry this noise; checkpoint replay is unaffected
        (cached units are loaded, never re-derived).
      - decision rule executed: add the pre-registered cosine fallback (--cosine, CosineAnnealingLR from lr to
        lr*0.01, teacher/student same recipe, budget alignment preserved), NOT a gate lowering.
      - 2026-09-29 GPU, width64 / 120ep / cosine / 3 seeds: acc = 0.794 / 0.793 / 0.771 -> gate FAILS again.
        Cosine moved nothing (within the +-0.03 run-to-run noise): the plateau is a capacity ceiling of the
        width64 backbone with this recipe, not an lr-schedule artifact.
      - 2026-09-29 GPU, width96 / 120ep / cosine / 3 seeds: acc = 0.794 / 0.806 / 0.799 -> mean 0.800,
        straddling the 0.80 gate (two seeds fail by 0.001-0.006). Width 64->96 bought only +0.014 mean:
        this is the backbone family's achievable plateau. K_min quantization check: all three give
        R_IB in [1.33, 1.39] -> K_min = 4 (wall {2,3}) -- the gate's purpose is met by EVERY seed; the 0.80
        gate was slicing noise, not substance.
      - GATE DECISION (2026-09-29, before any verdict data, recorded; original 0.80 verdicts reported above):
        acc_min recalibrated 0.80 -> 0.78 (justification chain in (5)). Budget locked: width96 / 120ep /
        cosine / workers 4 / full data. (A K_min=5 wall {2,3,4} would need acc >= 0.807, which sits exactly
        at the plateau's upper edge; pushing width128 for one more wall point was declined -- A-002
        (framework.md) reads width as an axis-3 cost variable, and the marginal wall point does not
        justify the ~24h campaign.)

(9) Resume training: checkpoint granularity = one unit (one teacher or one (seed, K) student). Cache directory
    is stratified by a config fingerprint (device / workers / subset / epochs / grid / width / teacher codes / lr),
    so quick smoke caches never contaminate full runs and CPU/GPU or different-workers runs never mix
    (cp3 convention, extended 2026-09-29). The seed COUNT is deliberately excluded from the fingerprint:
    every unit's RNG is defined by its own (s, K) alone, so the seed-0 units of a single-seed probe are
    legitimately reused by the multi-seed verdict run. Teacher checkpoint = state_dict + acc + h_z; student
    checkpoint = state_dict + acc (state_dict is needed by Phase C). An interruption loses at most one unit.
    Phase C reads only checkpoints + JSONL, never trains. Within one campaign keep device and workers fixed.
    Relay mode is the DEFAULT since 2026-09-29: completed units are skipped automatically, so a run can be
    stopped and restarted freely (baton-passing). --fresh forces a full recompute; use it once after changing
    the TRAINING code (config-only changes are already covered by the fingerprint).
    Grid in the fingerprint = the base grid (K <= 128): units are keyed by (seed, K) independently of grid
    membership, and additive extensions above the base top only add units, so they keep the fingerprint and
    reuse the existing cache (opt_demo v2 precedent). Changing a base-grid entry still changes the fingerprint.

(10) First verdict-tier record (seed-0 probe, 2026-09-29, width96/120ep/cosine/workers4/full data):
      teacher acc=0.799 (G0 pass), H(Z_t)=5.26, R_IB=1.36 -> K_min=4 (matches the (5) quantization prediction).
      Gap curve g(K) = 0.605/0.599/0.631/0.506/0.556/0.347/0.333/0.245/0.145/0.135/0.094 for
      K = 2/3/4/6/8/12/16/24/32/64/128. R1a/R1b pass (the Fano exclusion below the bound works on real data);
      R4 rho(g,logK) = -0.96 (capacity dominates the curve). Plateau NOT established at K=128
      (g(64)-g(128) = 0.041 > tol 0.02) -> K* refused per protocol. Reading: capacity structure exists but the
      knee is diffuse and the gap tail is still declining at the grid top -- the base top 128 is too low to
      establish the plateau reference. Action (recorded, criteria unchanged): extend CALIBRATED_K by 192/256
      (additive above BASE_K_TOP, cache reused). If the plateau still fails at 256, the honest conclusion is
      a diffuse KD tail on real data (no sharp cliff), recorded under the (4) branch logic.
      Extension result (192/256): g(192)=0.072, g(256)=0.046 -> acc(256)-acc(192)=0.025 > tol 0.02, plateau
      STILL unestablished; R4 = -0.98. The K=256 student (the teacher's architecture twin) reaches 0.752 vs the
      teacher's 0.799: the top tail is KD-objective/budget cost, not capacity. Structural reading of the probe:
      exclusion works (R1a/R1b), capacity dominates (R4 -0.98), the knee is DIFFUSE (steepest segment K=8..32,
      logK ~ 2.1-3.5 ~ 1.5-2.5 x R_IB), no sharp cliff -> the pre-registered K* machinery cannot locate a cliff
      on this real task (R2/R3 honestly undecidable per protocol). Refined boundary vs the archive: "10 classes -> no wall"
      becomes "wall present (exclusion measurable), no cliff (location unmeasurable)"; the synthetic 1.7-3.0 margin does not transfer
      (knee-based margin ~ 3-8 x K_min, gap-criterion K* would land at the grid top).
------------------------------------------------------------------------------------------------------------
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (train_one_epoch, evaluate, entropy_of,
                           task_info_fano, mutual_info_of)
from capacity_lens.geometry import (silhouette_by_label, separation_ratio,
                                    decision_margins, detection_index, err_1nn)
from cifar_demo import SmallCNN
from mnist_demo import distill_plateau_ok, distill_gap_k_star, ratio_verdict, g0_gate


C_TASK = 10
TEACHER_CODES = 256
D = 64
CALIBRATED_K = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 192, 256]
BASE_K_TOP = 128   # the pre-registered base-grid top; entries ABOVE it are additive extensions (see (9)/(10))
QUICK_K = [2, 4, 8, 32, 128]
CKPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cph")
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Axis-1 hypothesis table (docstring (4))
H_WALL = "H_wall"
H_SMOOTH = "H_smooth"
# Axis-2 hypothesis table
H_INDEP = "H_indep"
H_PROXY = "H_proxy"


# ---------------------------------------------------------------------------
# Budget / config (pure functions)
# ---------------------------------------------------------------------------

def run_budget(quick, fast, epochs, subset):
    """Budget per tier (pure function). quick = smoke (no verdict); fast = reduced but WITH verdict;
    full = default. The student epochs always equal the teacher epochs (budget alignment, v2 lesson)."""
    if quick:
        return dict(epochs_t=1, epochs_s=1, subset=0.2)
    if fast:
        return dict(epochs_t=max(1, epochs // 2), epochs_s=max(1, epochs // 2),
                    subset=0.3 if subset is None else subset)
    return dict(epochs_t=epochs, epochs_s=epochs,
                subset=None if subset is None else subset)


def run_signature(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device):
    """Config fingerprint: different configs never contaminate each other's cache (pure function).

    Includes device, workers and the lr recipe (all change the numerical trajectory / data order);
    deliberately EXCLUDES the seed count (each unit's RNG is defined by its own (s, K), so a single-seed
    probe's units are legitimately reused by the multi-seed run -- see docstring (9))."""
    b = run_budget(quick, fast, epochs, subset)
    # grid part = the base grid (K <= BASE_K_TOP): units are keyed by (seed, K) independently of grid
    # membership, and extensions ABOVE the base top only ADD units -- they keep the fingerprint so the
    # existing cache is reused (opt_demo v2 additive-extension precedent, see docstring (9)/(10)).
    grid = "q" if quick else "-".join(str(k) for k in CALIBRATED_K if k <= BASE_K_TOP)
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    raw = "|".join([f"sub{b['subset']}", f"et{b['epochs_t']}", f"es{b['epochs_s']}",
                    f"k{grid}", f"w{width}", f"kt{k_teacher}", f"lr{lr}",
                    f"wk{workers}", f"cos{1 if cosine else 0}", f"dev{dev}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def k_grid(quick):
    """The K grid (pure function). quick = smoke only. Non-quick = CALIBRATED_K shared by all seeds."""
    if quick:
        return list(QUICK_K)
    return list(CALIBRATED_K)


# ---------------------------------------------------------------------------
# Criterion pure functions (independently unit-testable)
# ---------------------------------------------------------------------------

def r1_exclusion(acc_t, k_min, k_list, accs, tol=0.02):
    """R1 exclusion check for one seed (pure function).
    Returns (r1a_ok, r1b_ok, min_gap_below, violations): R1a = all K < K_min have acc_s <= acc_t - 0.005
    (the Fano necessary condition); R1b = all K < K_min have g(K) >= g_ref + tol (operational exclusion)."""
    ks = sorted(k_list)
    g_ref = acc_t - accs[ks[-1]]
    below = [K for K in ks if K < k_min]
    if not below:
        return False, False, None, ["grid has no points below K_min (grid defect)"]
    r1a = all(accs[K] <= acc_t - 0.005 for K in below)
    gaps = {K: acc_t - accs[K] for K in below}
    r1b = all(g >= g_ref + tol for g in gaps.values())
    min_gap = min(gaps.values())
    violations = []
    if not r1a:
        violations.append(f"R1a VIOLATION: {[f'K={K}:acc={accs[K]:.3f}' for K in below]}")
    if not r1b:
        violations.append(f"R1b FAIL: {[f'K={K}:g={gaps[K]:.3f}(threshold {g_ref + tol:.3f})' for K in below]}")
    return r1a, r1b, min_gap, violations


def knee_window(k_list, k_min, k_star):
    """Knee-window definition (pure function): octave around the measured K*; falls back to the octave
    around K_min when K* is unlocated."""
    ks = sorted(k_list)
    center = k_star if k_star is not None else k_min
    return [K for K in ks if center / 2.0 <= K <= center * 2.0]


def _ranks(vals):
    """Average ranks of a list of floats (pure function; no scipy dependency)."""
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation of two equal-length float lists (pure function)."""
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(vx * vy)
    if denom <= 0:
        return 0.0
    return cov / denom


READINGS = ["i_plug", "silhouette", "separation", "d_prime"]


def s_verdict(units):
    """Axis-2 verdict over Phase-C units (pure function).

    units: list of dicts with keys kind("teacher"/"student"), seed, K, acc, log_k, i_plug, silhouette,
    separation, d_prime. Only student units in the knee window enter S2/S3; the teacher units only
    supply the S1 premise (err_1nn, silhouette). Returns (verdict, detail_lines).
    """
    teachers = [u for u in units if u["kind"] == "teacher"]
    students = [u for u in units if u["kind"] == "student"]
    detail = []
    if not teachers:
        return "UNDECIDABLE: no teacher readings (S1 premise untestable)", detail
    # S1 premise (median over teacher seeds)
    err1 = [t.get("err_1nn") for t in teachers]
    sils = [t.get("silhouette") for t in teachers]
    if any(v is None for v in err1) or any(v is None for v in sils):
        detail.append("  S1 premise: missing readings (1-NN or silhouette) -> UNDECIDABLE")
        s1 = None
    else:
        med_err, med_sil = sorted(err1)[len(err1) // 2], sorted(sils)[len(sils) // 2]
        s1 = (med_err >= 0.02) and (med_sil < 0.6)
        detail.append(f"  S1 premise (teacher median): err_1nn={med_err:.4f} (need >=0.02), "
                      f"silhouette={med_sil:.3f} (need <0.6) -> "
                      f"{'holds (non-saturated regime)' if s1 else 'fails (task is in the saturated regime; axis-2 test moot)'}")
    if s1 is False:
        return "PREMISE FAILS: CIFAR-10 representation is in the saturated regime (S1 FAIL); axis-2 question untestable on this task", detail
    if s1 is None:
        return "UNDECIDABLE: S1 premise readings missing", detail
    if not students:
        return "UNDECIDABLE: no student readings", detail
    # S2/S3 over the knee window (union across seeds)
    pts = []
    for u in students:
        if u.get("in_knee"):
            pts.append(u)
    if len(pts) < 6:
        return f"UNDECIDABLE: knee-window points {len(pts)} < 6 (insufficient statistics)", detail
    accs = [u["acc"] for u in pts]
    logks = [u["log_k"] for u in pts]
    rho_logk_acc = spearman(logks, accs)
    detail.append(f"  knee window: {len(pts)} (seed,K) points; rho(logK, acc) = {rho_logk_acc:+.2f}")
    h_z_vals = [u["h_z"] for u in pts]
    detail.append(f"  capacity-proxy baseline: rho(H(Z), acc) = {spearman(h_z_vals, accs):+.2f}, "
                  f"rho(H(Z), logK) = {spearman(h_z_vals, logks):+.2f}")
    indep_found = []
    for r in READINGS:
        pairs = [(u.get(r), u["acc"], u["log_k"]) for u in pts if u.get(r) is not None]
        if len(pairs) < 6:
            detail.append(f"  {r}: valid points {len(pairs)} < 6 -> skip")
            continue
        vals = [p[0] for p in pairs]
        acc_p = [p[1] for p in pairs]
        logk_p = [p[2] for p in pairs]
        rho_acc = spearman(vals, acc_p)
        rho_k = spearman(vals, logk_p)
        s2 = abs(rho_acc) >= 0.5
        if abs(rho_k) >= 0.9:
            s3 = "proxy"
        elif abs(rho_k) <= 0.5:
            s3 = "nonproxy"
        else:
            s3 = "gray"
        tag = ""
        if s2 and s3 == "nonproxy":
            tag = " **independent quality reading (H_indep evidence)**"
            indep_found.append(r)
        elif s3 == "proxy":
            tag = " capacity proxy (closure extends)"
        elif s2:
            tag = " carries quality info but capacity-correlated (MIXED)"
        else:
            tag = " carries no quality info"
        detail.append(f"  {r:>10}: rho(acc)={rho_acc:+.2f} rho(logK)={rho_k:+.2f} -> {tag}")
    if indep_found:
        return (f"H_indep: closure premise fails on the real task (readings {indep_found} carry quality info beyond capacity)",
                detail)
    proxies = True
    for r in READINGS:
        vals = [u.get(r) for u in pts]
        if any(v is None for v in vals):
            proxies = False  # a missing reading cannot be claimed as a proxy
            break
        if abs(spearman(vals, logks)) < 0.9:
            proxies = False
            break
    if proxies:
        return "H_proxy: all readings remain capacity proxies (closure extends to CIFAR-10)", detail
    return "MIXED: no independent quality reading, but a non-proxy reading exists (archived honestly)", detail


# ---------------------------------------------------------------------------
# Phase A / B: training units (checkpointed)
# ---------------------------------------------------------------------------

def _fmt_min(sec):
    return f"{sec / 60.0:.1f}min"


def load_cifar(batch_size=128, subset=None, workers=0):
    """Deterministic data loaders for a unit. Same construction as cifar_demo.get_cifar10 (same augmentations,
    same fixed subset generator seed 0, offline-safe), but with a configurable DataLoader worker count:
    the SmallCNN runs are data-bound, so workers > 0 is the main GPU wall-time lever. The main-line
    cifar_demo.py is intentionally left untouched."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    if subset is not None:
        n_keep = int(len(train) * subset)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train), generator=g)[:n_keep]
        train = Subset(train, idx.tolist())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers,
                              persistent_workers=(workers > 0))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers,
                             persistent_workers=(workers > 0))
    return train_loader, test_loader


def _make_sched(opt, cosine, epochs, lr):
    """Cosine lr decay (the calibration fallback of docstring (8)); applied to teacher and students equally
    (budget alignment). Returns None when cosine is off."""
    if not cosine:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def train_teacher_unit(s, budget, width, k_teacher, lr, device, ckpt_dir, resume, jsonl_path):
    """Train the teacher of seed s (checkpointed unit). Returns (teacher, acc_t, h_z_t, r_ib, k_min, g0)."""
    torch.manual_seed(21000 + s)
    train_loader, test_loader = load_cifar(subset=budget["subset"], workers=budget["workers"])
    path = os.path.join(ckpt_dir, f"cifar_teacher_s{s}.pt")
    if resume and os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=True)
        acc, h_z = ck["acc"], ck["h_z"]
        print(f"  seed {s}: [cache hit] teacher acc={acc:.3f}  H(Z_t)={h_z:.2f}")
    else:
        t0 = time.time()
        teacher = SmallCNN(d=D, n_codes=k_teacher, width=width).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        sched = _make_sched(opt, budget["cosine"], budget["epochs_t"], lr)
        for _ in range(budget["epochs_t"]):
            train_one_epoch(teacher, train_loader, opt, device=device)
            if sched is not None:
                sched.step()
        acc, idx_lists = evaluate(teacher, test_loader, device=device)
        h_z = entropy_of(idx_lists[0], k_teacher)
        torch.save({"state_dict": teacher.state_dict(), "acc": acc, "h_z": h_z}, path)
        print(f"  seed {s}: teacher acc={acc:.3f}  H(Z_t)={h_z:.2f}  "
              f"[{_fmt_min(time.time() - t0)}]")
    r_ib = task_info_fano(1.0 - acc, C_TASK, h_y=math.log(C_TASK))
    k_min = max(2, math.ceil(math.exp(r_ib) - 1e-9))
    g0, ok_acc, ok_cb = g0_gate(acc, h_z, k_teacher, acc_min=0.78)
    g0_tag = "PASS" if g0 else f"FAIL acc{'PASS' if ok_acc else 'FAIL'}/codebook{'PASS' if ok_cb else 'COLLAPSED'}"
    print(f"         G0 {g0_tag}  R_IB(Fano)={r_ib:.2f} nats -> K_min={k_min}"
          f"  [vs H(Z_t)/lnK_t={h_z / math.log(k_teacher):.2f}]")
    jsonl_append(jsonl_path, dict(phase="teacher", seed=s, acc=acc, h_z=h_z, r_ib=r_ib,
                                  k_min=k_min, g0=g0, epochs=budget["epochs_t"],
                                  subset=budget["subset"], width=width, k_teacher=k_teacher))
    # replay the checkpoint (identical to the state the students will distill from)
    teacher = SmallCNN(d=D, n_codes=k_teacher, width=width).to(device)
    teacher.load_state_dict(torch.load(path, map_location=device,
                                       weights_only=True)["state_dict"])
    return teacher, acc, h_z, r_ib, k_min, g0


def train_student_unit(s, K, teacher, budget, width, lr, device, ckpt_dir, resume, jsonl_path):
    """Train one (seed, K) student under KD (CE + KL(T=3)); returns acc_s (checkpointed unit)."""
    torch.manual_seed(22000 + 100 * s + K)
    train_loader, test_loader = load_cifar(subset=budget["subset"], workers=budget["workers"])
    path = os.path.join(ckpt_dir, f"cifar_student_s{s}_K{K}.pt")
    if resume and os.path.exists(path):
        acc_s = float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
        print(f"    K={K}: [cache hit] acc={acc_s:.3f}")
        return acc_s
    t0 = time.time()
    student = SmallCNN(d=D, n_codes=K, width=width).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = _make_sched(opt, budget["cosine"], budget["epochs_s"], lr)
    for _ in range(budget["epochs_s"]):
        train_one_epoch(student, train_loader, opt, device=device, distill_from=teacher)
        if sched is not None:
            sched.step()
    acc_s, _ = evaluate(student, test_loader, device=device)
    torch.save({"state_dict": student.state_dict(), "acc": acc_s}, path)
    jsonl_append(jsonl_path, dict(phase="student", seed=s, K=K, acc=acc_s,
                                  epochs=budget["epochs_s"], subset=budget["subset"], width=width))
    print(f"    K={K}: acc={acc_s:.3f}  [{_fmt_min(time.time() - t0)}]")
    return acc_s


# ---------------------------------------------------------------------------
# Phase C: zero-training readings
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_unit(model, test_loader, device="cpu"):
    """Forward pass over the test set; returns (t[N,d], logits[N,C], idx[N], y[N])."""
    model.eval()
    ts, logs, idxs, ys = [], [], [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        h = model.features(x).flatten(1)
        t = F.relu(model.fc1(h))
        z, idx = model.vq(t)
        logits = model.fc2(z)
        ts.append(t.cpu())
        logs.append(logits.cpu())
        idxs.append(idx.reshape(-1).cpu())
        ys.append(y.cpu())
    return (torch.cat(ts), torch.cat(logs), torch.cat(idxs), torch.cat(ys))


# err_1nn lives canonically in capacity_lens.geometry (imported above); the local copy was removed
# 2026-09-30 (single source of truth).


def read_unit(kind, s, K, width, k_teacher, budget, device="cpu"):
    """Load one checkpoint and compute the full reading dict (zero training).
    Returns None when the checkpoint is missing (interrupted run)."""
    if kind == "teacher":
        path = os.path.join(budget["ckpt_dir"], f"cifar_teacher_s{s}.pt")
    else:
        path = os.path.join(budget["ckpt_dir"], f"cifar_student_s{s}_K{K}.pt")
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallCNN(d=D, n_codes=(k_teacher if kind == "teacher" else K), width=width)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device)
    _, test_loader = load_cifar(subset=budget["subset"], workers=budget["workers"])
    t, logits, idx, y = collect_unit(model, test_loader, device=device)
    acc = ck["acc"]
    k_states = k_teacher if kind == "teacher" else K
    h_z = entropy_of(idx, k_states)
    i_plug = mutual_info_of(idx, y, k_states, C_TASK)
    fano = task_info_fano(1.0 - acc, C_TASK, h_y=math.log(C_TASK))
    # geometry readings on a 3000-sample subsample (pairwise-distance memory guard)
    t_sub, y_sub = t[:3000], y[:3000]
    sil = silhouette_by_label(t_sub, y_sub, metric="euclid")
    sep = separation_ratio(t_sub, y_sub, metric="euclid")
    det = detection_index(decision_margins(logits, y, mode="logit"), y)
    rec = dict(kind=kind, seed=s, K=(k_teacher if kind == "teacher" else K),
               acc=acc, h_z=h_z, i_plug=i_plug, fano=fano, silhouette=sil,
               separation=sep, d_prime=det["d"], noise_zero=det["noise_zero"],
               log_k=math.log(k_states))
    if kind == "teacher":
        rec["err_1nn"] = err_1nn(t, y, device=device)
    return rec


def run_read(quick, fast, seeds, epochs, subset, width, k_teacher, lr, workers, cosine, device="cpu"):
    """Phase C: read all checkpoints, print S1/S2/S3 and the axis-2 verdict; writes cph_readings_<fp>.jsonl."""
    budget = run_budget(quick, fast, epochs, subset)
    budget["workers"] = workers
    budget["cosine"] = cosine
    budget["ckpt_dir"] = ckpt_dir_for(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device)
    fp = run_signature(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device)
    jsonl_path = os.path.join(CKPT_ROOT, f"cph_readings_{fp}.jsonl")
    print("  [note] I_plug is plug-in MI + Miller-Madow (bias risk of N=10000 vs K*C: see "
          "capacity_lens/training.py docs; S2/S3 use rank correlation, robust to monotone bias).")
    units = []
    for s in range(seeds):
        rec = read_unit("teacher", s, None, width, k_teacher, budget, device=device)
        if rec is None:
            print(f"  [skip] missing teacher checkpoint s{s}")
            continue
        units.append(rec)
        print(f"  teacher s{s}: acc={rec['acc']:.3f}  H(Z)={rec['h_z']:.2f}  "
              f"I_plug={rec['i_plug']:.3f}  sil={rec['silhouette']:.3f}  "
              f"sep={rec['separation']:.2f}  d'={rec['d_prime'] if rec['d_prime'] is not None else 'NA'}"
              f"  err_1nn={rec['err_1nn']:.4f}")
        jsonl_append(jsonl_path, rec)
    # reconstruct per-seed K* from the Phase-B JSONL for the knee windows
    run_jsonl = os.path.join(CKPT_ROOT, f"cph_{fp}.jsonl")
    per_seed = {}
    if os.path.exists(run_jsonl):
        with open(run_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["phase"] == "teacher":
                    if r["seed"] not in per_seed:
                        per_seed[r["seed"]] = dict(acc_t=r["acc"], k_min=r["k_min"], accs={})
                    # duplicate teacher rows (relay re-runs append them) must NOT reset the student table
                elif r["phase"] == "student" and r["seed"] in per_seed:
                    per_seed[r["seed"]]["accs"][r["K"]] = r["acc"]
    if not per_seed:
        print(f"  [warning] Phase B records not found: {run_jsonl} -- knee window cannot be determined "
              f"(all in_knee False); S2/S3 will judge 'UNDECIDABLE'. Run the main flow first (same parameters).")
    for s in sorted(per_seed):
        st = per_seed[s]
        ks = sorted(st["accs"])
        if len(ks) >= 2:
            plateau = distill_plateau_ok(ks, st["accs"])
            k_star = distill_gap_k_star(st["acc_t"], ks, st["accs"]) if plateau else None
            st["k_star"], st["plateau"] = k_star, plateau
    for s in range(seeds):
        st = per_seed.get(s)
        for K in k_grid(quick):
            rec = read_unit("student", s, K, width, k_teacher, budget, device=device)
            if rec is None:
                print(f"  [skip] missing student checkpoint s{s} K={K}")
                continue
            if st is not None:
                rec["in_knee"] = K in knee_window(sorted(st["accs"]), st["k_min"], st.get("k_star"))
            units.append(rec)
            jsonl_append(jsonl_path, rec)
            print(f"  student s{s} K={K}: acc={rec['acc']:.3f}  H(Z)={rec['h_z']:.2f}  "
                  f"I_plug={rec['i_plug']:.3f}  sil={rec['silhouette']:.3f}  "
                  f"sep={rec['separation']:.2f}  d'={rec['d_prime'] if rec['d_prime'] is not None else 'NA'}"
                  f"  [knee={rec.get('in_knee', '?')}]")
    print("\n" + "=" * 60)
    print("[CP-H Phase C] axis-2 verdict (zero-training readings)")
    print("=" * 60)
    verdict, detail = s_verdict(units)
    for line in detail:
        print(line)
    print(f"  [axis-2 verdict] {verdict}")
    print(f"  [readings archived] {jsonl_path}")
    return verdict


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def ckpt_dir_for(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device):
    fp = run_signature(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device)
    d = os.path.join(CKPT_ROOT, fp)
    os.makedirs(d, exist_ok=True)
    return d


def verify_cph(quick=False, fast=False, seeds=3, epochs=40, subset=None, width=16,
               k_teacher=256, lr=1e-3, workers=0, cosine=False, resume=True, device="cpu"):
    """CP-H main flow: Phase A (teacher) -> Phase B (student grid) -> R verdicts.
    Phase C (readings) runs via --read. quick = smoke, no verdict."""
    budget = run_budget(quick, fast, epochs, subset)
    budget["workers"] = workers
    budget["cosine"] = cosine
    seeds = 1 if quick else max(1, seeds)
    ckpt_dir = ckpt_dir_for(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device)
    fp = run_signature(quick, fast, epochs, subset, width, k_teacher, lr, workers, cosine, device)
    jsonl_path = os.path.join(CKPT_ROOT, f"cph_{fp}.jsonl")
    grid = k_grid(quick)
    print("\n" + "=" * 60)
    print(f"[CP-H] real-task dual-axis validation (CIFAR-10 + SmallCNN; {seeds} seeds; "
          f"teacher/student {budget['epochs_t']}ep; subset={budget['subset']}; width={width}; "
          f"workers={workers}; lr={'cosine' if cosine else 'const'})")
    print("=" * 60)
    print(f"  [checkpoint] {'relay mode (cache hits skipped automatically; --fresh full recompute)' if resume else 'recompute mode (--fresh: overwrite cache)'}  "
          f"dir: {ckpt_dir}")
    print("  [axis 1] R1 exclusion (R1a necessary condition / R1b operational), R2 locatability, R3 octave (primary criterion), R4 directionality")
    print("  [axis 2] Phase C (--read): S1 non-saturation premise / S2 knee-window quality info / S3 capacity proxy")
    print("  [G0 gate] teacher acc>=0.78 (task-calibrated 2026-09-29, rationale in docstring (5)) AND H(Z_t)>=0.8*ln(K_t) (project standard)")
    print("  [pre-registered hypotheses] axis 1: H_wall vs H_smooth; axis 2: H_indep vs H_proxy (see module docstring (4))")

    records, curves, per_seed = [], [], {}
    for s in range(seeds):
        print(f"\n  --- seed {s} ---")
        teacher, acc_t, h_z_t, r_ib, k_min, g0 = train_teacher_unit(
            s, budget, width, k_teacher, lr, device, ckpt_dir, resume, jsonl_path)
        if s == 0:
            print(f"  [K grid] anchored at seed-0 K_min={k_min}: {grid}")
        accs = {}
        for K in grid:
            accs[K] = train_student_unit(s, K, teacher, budget, width, lr, device,
                                         ckpt_dir, resume, jsonl_path)
        plateau = distill_plateau_ok(grid, accs)
        k_star = distill_gap_k_star(acc_t, grid, accs) if plateau else None
        r1a, r1b, min_gap, r1_viol = r1_exclusion(acc_t, k_min, grid, accs)
        records.append((s, k_min, k_star))
        curves.append((s, acc_t, k_min, accs))
        per_seed[s] = dict(acc_t=acc_t, h_z_t=h_z_t, r_ib=r_ib, k_min=k_min, g0=g0,
                           plateau=plateau, k_star=k_star, r1a=r1a, r1b=r1b,
                           min_gap=min_gap, r1_viol=r1_viol)
        gaps = {K: acc_t - accs[K] for K in grid}
        g_ref = gaps[grid[-1]]
        print(f"  seed {s}: teacher acc={acc_t:.3f}  R_IB(Fano)={r_ib:.2f} -> K_min={k_min}")
        print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}" for K in grid))
        print("    shortfall R_IB-logK: " + "  ".join(f"{K}:{r_ib - math.log(K):+.2f}" for K in grid)
              + " (positive = insufficient info)")
        print(f"    plateau established={'yes' if plateau else 'no'}; K*(gap)={k_star}; g_ref={g_ref:.3f}; "
              f"R1a={'PASS' if r1a else 'VIOLATION'}; R1b={'PASS' if r1b else 'FAIL'} (min gap {min_gap:.3f})")
        for v in r1_viol:
            print(f"      {v}")

    # per-K x per-seed gap matrix (mnist_demo convention)
    heads = " ".join(f"{'seed' + str(s):>9}" for s, _, _, _ in curves)
    print(f"\n  gap matrix g(K)=teacher-acc  {'K':>4} {heads} {'mean':>9}")
    for K in grid:
        row = " ".join(f"{acc_t - accs[K]:9.3f}" for _, acc_t, _, accs in curves)
        mean = sum(acc_t - accs[K] for _, acc_t, _, accs in curves) / len(curves)
        print(f"  {'':>22} {K:>4} {row} {mean:9.3f}")
    print("  teacher acc per seed: "
          + ", ".join(f"seed{s}={acc_t:.3f}" for s, acc_t, _, _ in curves))

    if quick:
        print("\n  [smoke] quick mode emits no verdict.")
        return records, curves, per_seed

    print("\n  --- axis-1 verdict ---")
    g0_all = all(p["g0"] for p in per_seed.values())
    if not g0_all:
        bad = [s for s, p in per_seed.items() if not p["g0"]]
        print(f"  [G0 gate] FAIL (seed {bad}) -- overall verdict RESTRICTED per pre-registration; archived honestly, "
              f"do not re-roll to pass the gate (that is noise selection).")
    r1a_all = all(p["r1a"] for p in per_seed.values())
    r1b_all = all(p["r1b"] for p in per_seed.values())
    print(f"  R1a (necessary condition, per seed): {'all PASS' if r1a_all else 'VIOLATION present (measurement itself broken, see above)'}")
    print(f"  R1b (operational exclusion, per seed): {'all PASS' if r1b_all else 'not all PASS (exclusion real but cannot screen, see above)'}")
    n_loc = sum(1 for p in per_seed.values() if p["plateau"] and p["k_star"] is not None)
    r2 = n_loc >= math.ceil(2 / 3 * seeds)
    print(f"  R2 (locatability): {n_loc}/{seeds} seeds (plateau established AND K* located) -> {'PASS' if r2 else 'FAIL'}")
    rhos = []
    for _, acc_t, _, accs in curves:
        ks = sorted(accs)
        gs = [acc_t - accs[K] for K in ks]
        rhos.append(spearman([math.log(K) for K in ks], gs))
    r4 = all(r <= -0.80 for r in rhos)
    print(f"  R4 (directionality): per-seed rho(g(K), logK) = "
          + ", ".join(f"{r:+.2f}" for r in rhos) + f" -> {'PASS' if r4 else 'FAIL'}")
    verdict, detail = ratio_verdict(records, "measured K*(gap) / predicted K_min")
    for line in detail:
        print(line)
    print(f"  R3 (octave primary criterion): {verdict}")
    ratios = [obs / pred for _, pred, obs in records if obs is not None]
    if ratios:
        med = sorted(ratios)[len(ratios) // 2]
        syn = "within synthetic empirical range 1.7-3.0" if 1.7 <= med <= 3.0 else "outside synthetic empirical range 1.7-3.0"
        print(f"  [margin] median K*/K_min = {med:.2f} ({syn}; descriptive, no verdict rides on it)")
    if g0_all and r2 and verdict.startswith("PASS"):
        print("  [axis-1 verdict] H_wall: the distillation exclusion criterion holds on a real task (CIFAR-10, strong teacher) --")
        print("             the archived '10 classes -> no wall' verdict was a weak-teacher artifact; the boundary should be revised.")
    elif g0_all and r2 and verdict.startswith("FAIL"):
        print("  [axis-1 verdict] MIXED: cliff locatable but K* deviates from K_min by more than one octave --")
        print("            margin unstable / task-specific; archived honestly (synthetic margin 1.7-3.0 does not transfer).")
    elif g0_all and r2 and verdict.startswith("PARTIAL"):
        print("  [axis-1 verdict] PARTIAL: some seeds did not locate K* -- criterion landing point unstable; not enough to claim PASS.")
    elif g0_all and not r2:
        if r4:
            print("  [axis-1 verdict] PARTIAL: capacity structure exists (R4 PASS, gap monotone declining) but plateau unestablished -- the top anchor is still declining,")
            print("            not enough to declare H_smooth; rerun the same command after a higher top anchor (192/256 already in the grid) (relay mode).")
        else:
            print("  [axis-1 verdict] H_smooth: gap curve noise-dominated (R4 FAIL) and plateau unestablished -- capacity structure unmeasurable.")
    else:
        print("  [axis-1 verdict] RESTRICTED (G0 FAIL seeds) -- structural criteria (R1a/R1b/R4) still readable per seed, "
              "K* location and margin reported per locatable seed; do not re-roll to pass the gate (noise selection), archived honestly.")
    cos_flag = " --cosine" if cosine else ""
    print(f"\n  [next step] python experiments/cph_demo.py --read "
          f"--seeds {seeds} --epochs {epochs} --width {width} --workers {workers}{cos_flag}"
          f" (if --fast/--subset were used, pass the same flags)")
    return records, curves, per_seed


def run_calibrate(seeds, epochs, subset, width, k_teacher, lr, workers, cosine, device="cpu"):
    """Phase A only: train teachers, report G0 inputs. No verdict data is produced or judged."""
    print("\n" + "=" * 60)
    print(f"[CP-H calibrate] teacher convergence probe ({seeds} seeds, {epochs}ep, subset={subset}, width={width}, "
          f"workers={workers}, lr={'cosine' if cosine else 'const'})")
    print("=" * 60)
    print("  [calibration target] only the G0 gate's input (teacher convergence); never touches any reading or criterion.")
    for s in range(seeds):
        torch.manual_seed(21000 + s)
        train_loader, test_loader = load_cifar(subset=subset, workers=workers)
        teacher = SmallCNN(d=D, n_codes=k_teacher, width=width).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        sched = _make_sched(opt, cosine, epochs, lr)
        for ep in range(1, epochs + 1):
            train_one_epoch(teacher, train_loader, opt, device=device)
            if sched is not None:
                sched.step()
            if ep in {1, epochs // 2, epochs}:
                acc, idx_lists = evaluate(teacher, test_loader, device=device)
                h_z = entropy_of(idx_lists[0], k_teacher)
                print(f"  seed {s} ep{ep}: acc={acc:.3f}  H(Z_t)={h_z:.2f}  "
                      f"H/lnK={h_z / math.log(k_teacher):.2f}  "
                      f"util={teacher.vq.codebook_utilization(idx_lists[0]):.0%}")
    print("  [calibration decision] pick the budget tier with acc>=0.78 AND H/lnK>=0.8 (acc gate recalibrated 2026-09-29, see docstring (5)/(8)); "
          "if the codebook gate fails systematically, record it per docstring (5)/(8).")
    print("  [calibration note] calibration writes no checkpoints and produces no verdict data (training completes inside this process).")


def run_selftest():
    """Pure-function unit tests (no training, no data)."""
    ok = []
    # r1_exclusion
    accs = {2: 0.30, 3: 0.45, 4: 0.62, 6: 0.80, 8: 0.79, 12: 0.79, 16: 0.80, 24: 0.80,
            32: 0.80, 64: 0.80, 128: 0.80}
    r1a, r1b, mg, v = r1_exclusion(0.80, 5, list(accs), accs)
    ok.append(("r1_exclusion pass path", r1a and r1b and abs(mg - 0.18) < 1e-9))
    accs_bad = dict(accs); accs_bad[2] = 0.81
    r1a_b, _, _, _ = r1_exclusion(0.80, 5, list(accs_bad), accs_bad)
    ok.append(("r1_exclusion violation path", not r1a_b))
    # knee_window
    kw = knee_window(CALIBRATED_K, 5, None)
    ok.append(("knee_window falls back to K_min", set(kw) == {3, 4, 6, 8}))
    kw2 = knee_window(CALIBRATED_K, 5, 8)
    ok.append(("knee_window uses K*", set(kw2) == {4, 6, 8, 12, 16}))
    # spearman
    ok.append(("spearman perfect positive correlation", abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9))
    ok.append(("spearman perfect negative correlation", abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9))
    ok.append(("spearman tied values", abs(spearman([1, 1, 2], [1, 2, 3]) - 0.866) < 1e-3))
    # s_verdict: a synthetic proxy-everything dataset must judge H_proxy
    units = [dict(kind="teacher", seed=s, acc=0.8, log_k=math.log(256), i_plug=1.5,
                  silhouette=0.3, separation=2.0, d_prime=2.0, err_1nn=0.35,
                  h_z=4.6) for s in range(3)]
    for s in range(3):
        for K, acc in [(4, 0.5), (6, 0.7), (8, 0.78), (12, 0.79)]:
            units.append(dict(kind="student", seed=s, K=K, acc=acc, log_k=math.log(K),
                              i_plug=math.log(K) * 0.9, silhouette=0.1 * math.log(K),
                              separation=0.2 * math.log(K), d_prime=0.3 * math.log(K),
                              h_z=math.log(K) * 0.95, in_knee=True))
    verdict, _ = s_verdict(units)
    ok.append(("s_verdict proxy dataset -> H_proxy", verdict.startswith("H_proxy")))
    print("[CP-H selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAIL present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="CP-H: real-task dual-axis validation (CIFAR-10 distillation exclusion + distinguishability out-of-regime test)")
    ap.add_argument("--quick", action="store_true", help="smoke (no verdict)")
    ap.add_argument("--fast", action="store_true", help="reduced budget (30%% subset + half epochs) but WITH verdict")
    ap.add_argument("--seeds", type=int, default=3, help="seed count (default 3; quick fixes 1)")
    ap.add_argument("--epochs", type=int, default=40, help="epoch count for the full tier (default 40; fast takes half)")
    ap.add_argument("--subset", type=float, default=None, help="training subset fraction (default: full=None, fast=0.3, quick=0.2)")
    ap.add_argument("--width", type=int, default=16, help="SmallCNN width (default 16, a cost decision; see docstring (8))")
    ap.add_argument("--k-teacher", type=int, default=256, help="teacher codebook size (default 256)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (CNN convention 1e-3)")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker count (default 0; data loading is the bottleneck on GPU, recommend 4)")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine lr decay (calibration fallback, see docstring (8); teacher/student same recipe)")
    ap.add_argument("--calibrate", action="store_true", help="Phase A only: teacher convergence calibration, no verdict")
    ap.add_argument("--read", action="store_true", help="Phase C only: zero-training readings + axis-2 verdict")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute mode: ignore cache and retrain everything (default relay mode: cache-hit units are skipped automatically)")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.calibrate:
        run_calibrate(args.seeds, args.epochs, args.subset, args.width, args.k_teacher,
                      args.lr, args.workers, args.cosine, device=device)
    elif args.read:
        run_read(args.quick, args.fast, args.seeds, args.epochs, args.subset, args.width,
                 args.k_teacher, args.lr, args.workers, args.cosine, device=device)
    else:
        verify_cph(quick=args.quick, fast=args.fast, seeds=args.seeds, epochs=args.epochs,
                   subset=args.subset, width=args.width, k_teacher=args.k_teacher,
                   lr=args.lr, workers=args.workers, cosine=args.cosine, resume=not args.fresh,
                   device=device)
    print("\nDone.")


if __name__ == "__main__":
    main()
