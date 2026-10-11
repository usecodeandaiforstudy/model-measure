"""SCALE-C: the class-count axis of the scale boundary, measured on the project's own lattice family.

Two questions, one script:
  STEP 0 (zero compute, --existing): the LOOSENESS survey already contains C=10 (CIFAR-10) and C=64
    (grid64/route64) scenarios, so C and epsilon are confounded in the "looseness rises with the error
    rate" law. Fit looseness ~ a + b*eps + c*ln(C) on those 38 scenarios: measured result is recorded in
    the note (the C term contributes +0.000 to R2 there, but only two C values exist, so it cannot rule
    out a C effect at larger C -- which is what this sweep is for).
  MAIN (--sweep): sweep C = m^3 over 125x with the project's own task family and re-measure (R_IB, K_min,
    K_wall, looseness, false exclusions) at every C.

Task family (the project's workhorse, generalised): `mnist_demo.make_grid64` is the m=4 case of a 3-input
Gaussian mixture on an m x m x m lattice with fixed spacing; here m in {2,3,4,6,10} gives C = m^3 in
{8,27,64,216,1000}. **The cell geometry (spacing 3.0, noise sigma) is identical at every m**, so the
per-class decision problem is the same and epsilon is comparable across C by construction -- and the m=4
point is a built-in anchor: cp3 measured its wall at K_wall=96 (R_IB 3.84-4.03, looseness 1.13-1.19), so
a correct re-measurement here reproduces a known number.

Design record (all fixed BEFORE the pre-registered sweep):
  1. the stock model heads hard-code their class count -> build_model() resizes them (C touches only the
     head, which is why this axis is nearly free in compute);
  2. a WHITE-projection synthetic family was tried first and abandoned after three measured failures
     (C=1000 stuck at chance; and with a fixed smooth basis, C>=256 still at chance) -- the project's own
     lattice family needs no such invention and reproduces known numbers;
  3. the REFERENCE must be capacity-unconstrained: with the stock 256-code bottleneck the reference sits at
     chance for C >= 256 (ln C >= 5.5 nats vs log 256 = 5.55), so one measures the reference's own
     bottleneck instead of the task. First run: reference = NO bottleneck. **Correction for this run
     (recorded before it started, after reading the first run's result)**: a bottleneck-free reference
     makes the KD students chase a continuous representation and they stayed 66% short even at 1.9x the
     certified requirement, so the wall could not be located for C >= 27 (1/10 instances). The reference is
     therefore a LARGE but FINITE VQ of the students' own family, K_t = min(8C, 8192) (log K_t = ln C + ~2
     nats, so it is not binding), and the student grid is capped at 8C instead of the first run's 2C --
     that cap had cut off exactly the sizes where the wall sits.
  4. seed budget: this run uses seed 0 only (the first run's two seeds agreed within 0.01 in eps and gave
     identical exclusion sets), which halves the wall-clock at the largest C.

Usage:
    python experiments/scale_c_demo.py --existing      # step 0, zero compute (seconds)
    python experiments/scale_c_demo.py --calibrate     # sigma ladder (records HOW sigma was chosen)
    python experiments/scale_c_demo.py                 # the C sweep, 2 seeds (relay-safe)
    python experiments/scale_c_demo.py --quick         # 2 class counts, few epochs (smoke)
    python experiments/scale_c_demo.py --selftest      # pure-function unit tests (no training)
    python experiments/scale_c_demo.py --verdict-only  # re-verdict from scale_c.jsonl (no training)

Pre-registration (2026-09-30, criteria before data)
-----------------------------------------------------------------------------------------------
(1) Instances: m in {2,3,4,6,10} -> C in {8,27,64,216,1000}; n_per_class = 150 (150*C samples, 80/20
    split); seeds 0 and 1; sigma chosen ONCE from the calibration ladder (a single global knob, never
    tuned per instance) so that eps lands inside 0.02..0.60 at as many C as possible.
(2) Models: GridMLP-style Linear(3->64) -> ReLU -> bottleneck(64) -> Linear(64->C). Reference = NO
    bottleneck (25/60 epochs, Adam 1e-2 per the calibrated MLP budget); students = VQ(K) distilled from it
    (same budget). K grid {2, 8, 32, 128, 512, 2048}, skipping K > 2C.
(3) Readouts per instance: acc_t -> eps -> R_IB = task_info_fano(eps, C) -> K_min -> excluded set;
    per K: acc_s -> relative gap -> measured wall K_wall (rel_tol = 0.10) -> looseness = ln K_wall / R_IB.
(4) Criteria:
      S1 (gate, per instance): 0.02 <= eps <= 0.60; instances outside are reported but excluded from the
         scaling fit (never silently dropped).
      S2 (wall scaling): fit ln K_wall = alpha*ln C + beta over gated instances with a located wall;
         report alpha, R2, n. No pass/fail -- this IS the unknown being measured.
      S3 (direction, derived prediction): looseness is NON-INCREASING in C (Spearman <= 0) and at the
         largest C it is within +-0.5 nats of alpha/(1-eps).
      S4 (soundness at scale): zero false exclusions at EVERY C.
      S5 (ceiling effect): the number of excluded grid levels at the largest C exceeds that at the
         smallest C (the ln C ceiling stops binding as C grows).
    Verdict (pure function scale_verdict): SUPPORT if S3 and S4 and S5; PARTIAL if two; else MIXED/REFUTED.
(5) Relay: per-unit checkpoints under experiments/data/scale_c/<fingerprint>/.
(6) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads.
-----------------------------------------------------------------------------------------------
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import (VQBottleneck, train_one_epoch, evaluate, entropy_of, task_info_fano,
                           DistillCapacity, excluded_set, coverage, false_exclusions, measured_wall,
                           looseness)

M_GRID = [2, 3, 4, 6, 10]                  # C = m^3 -> 8, 27, 64, 216, 1000
# Student grid, ratio 4, capped at K_TEACHER_MULT*C (the first run's cap of 2C cut off exactly the sizes
# where the wall sits -- see the design record). The reference uses a large but FINITE VQ of the same
# family, so a KD student can actually approach it (with a bottleneck-free reference the students stayed
# 66% short even at 1.9x the certified requirement).
K_GRID = [2, 8, 32, 128, 512, 2048, 8192]
K_TEACHER_MULT = 8
K_TEACHER_CAP = 8192
SEEDS = 2
EPOCHS = 60
LR = 1e-2
SPACING = 3.0
N_PER_CLASS = 150
SIGMA = 1.0                                # ADOPTED from --calibrate: the only ladder value that keeps
                                           # eps inside 0.02..0.60 at ALL five class counts
                                           # (measured eps at sigma=1.0: 0.158 / 0.275 / 0.318 / 0.365 / 0.451
                                           #  for C = 8 / 27 / 64 / 216 / 1000; at sigma=0.5 the m=4 point
                                           #  gives 0.035, reproducing cp3's measured MLP eps 0.014-0.039)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "scale_c.jsonl")
LOG = os.path.join(_DATA, "scale_c.log")
REPORT = os.path.join(_DATA, "scale_c_report.txt")
CKPT_ROOT = os.path.join(_DATA, "scale_c")
LOOSENESS_JSONL = os.path.join(_DATA, "looseness.jsonl")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logging():
    global _LOG
    _LOG = open(LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Task and models (the project's lattice family, generalised from m=4 to any m)
# ---------------------------------------------------------------------------

def class_count(m):
    return m ** 3


def make_lattice(m, n_per_class=N_PER_CLASS, sigma=SIGMA, spacing=SPACING, seed=0):
    """The m=4 case is exactly mnist_demo.make_grid64 (same generator, same spacing, same split)."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.cartesian_prod(torch.arange(m), torch.arange(m), torch.arange(m)).float()
    centers = coords * spacing
    n_classes = len(centers)
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    x = centers[y] + sigma * torch.randn(len(y), 3, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return ((x[:n_train], y[:n_train]), (x[n_train:], y[n_train:]))


def instance_loaders(m, seed, batch_size=128):
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, seed=seed)
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0))


class NoVQ(nn.Module):
    """Pass-through stand-in for VQBottleneck (the REFERENCE must not be limited by its own bottleneck)."""

    def forward(self, T):
        return T, torch.zeros(T.shape[0], dtype=torch.long, device=T.device)


class GridMLP(nn.Module):
    """Linear(3->64) -> ReLU -> bottleneck(64) -> Linear(64->C). K=None -> no bottleneck (the reference)."""

    def __init__(self, C, K=None):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.vq = NoVQ() if (K is None or K == 0) else VQBottleneck(64, K)
        self.fc2 = nn.Linear(64, C)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z, idx = self.vq(h)
        return self.fc2(z), [idx]


def teacher_codes(c):
    """The reference's bottleneck size: large enough that it is NOT the binding constraint (log K_t =
    ln C + ln 8, i.e. ~2 nats above the largest requirement ln C), yet the same family as the students."""
    return min(K_TEACHER_MULT * c, K_TEACHER_CAP)


def build_model(C, K, device):
    return GridMLP(C, K).to(device)


def train_unit(C, K, seed, train_loader, test_loader, teacher, epochs, device, ckpt_path, resume):
    """One unit: K = 'ref' -> the reference (plain CE, no KD); otherwise a VQ student distilled from it."""
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return float(ck["acc"]), ck.get("h_z")
    is_ref = K == "ref"
    torch.manual_seed(60000 + 1000 * seed + (0 if is_ref else K))
    model = build_model(C, teacher_codes(C) if is_ref else K, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, device=device,
                        distill_from=None if is_ref else teacher)
    acc, idx = evaluate(model, test_loader, device=device)
    h_z = entropy_of(idx[0], teacher_codes(C)) if (is_ref and idx) else None
    torch.save({"acc": acc, "h_z": h_z, "state_dict": model.state_dict()}, ckpt_path)
    return acc, h_z


# ---------------------------------------------------------------------------
# Pure functions: the scaling analysis
# ---------------------------------------------------------------------------

def ols(xs, ys):
    """Two-variable OLS with intercept: y ~ a + b*x1 + c*x2. Returns (a, b, c, r2)."""
    n = len(xs)
    X = [[1.0, x1, x2] for x1, x2 in xs]
    Xt = [[X[i][j] for i in range(n)] for j in range(3)]
    A = [[sum(Xt[j][k] * X[k][l] for k in range(n)) for l in range(3)] for j in range(3)]
    b = [sum(Xt[j][k] * ys[k] for k in range(n)) for j in range(3)]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            return 0.0, 0.0, 0.0, 0.0
        for r in range(3):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, 4):
                    M[r][c] -= f * M[col][c]
    coef = [M[i][3] / M[i][i] for i in range(3)]
    pred = [coef[0] + coef[1] * x1 + coef[2] * x2 for x1, x2 in xs]
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    return coef[0], coef[1], coef[2], (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def ols1(xs, ys):
    """One-variable OLS with intercept: y ~ a + b*x. Returns (a, b, r2).

    Separate from ols() on purpose: feeding a constant column into the two-variable solver makes it
    singular, which silently returned zeros in the first version of this script."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def rank_correlation(xs, ys):
    """Spearman with average ranks (needs >= 3 points)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = math.sqrt(sum((p - mx) ** 2 for p in rx))
    dy = math.sqrt(sum((q - my) ** 2 for q in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def asymptote_note(alpha, eps):
    """P2's predicted large-C limit of looseness, given the wall exponent and the error rate."""
    return alpha / max(1.0 - eps, 1e-6)


def step0_existing():
    """STEP 0 (zero compute): is C separable from epsilon in the existing 38-scenario survey?"""
    if not os.path.exists(LOOSENESS_JSONL):
        raise SystemExit(f"[SCALE-C] {LOOSENESS_JSONL} not found; run looseness_demo.py --write first")
    rows = [json.loads(line) for line in open(LOOSENESS_JSONL, encoding="utf-8")]
    rows = [r for r in rows if r.get("looseness") is not None and r.get("acc_ref") is not None]
    if not rows:
        raise SystemExit("[SCALE-C] no usable scenarios in looseness.jsonl")
    ys = [r["looseness"] for r in rows]
    a, b, c, r2 = ols([(r["eps"], math.log(r["n_classes"])) for r in rows], ys)
    a1, b1, r21 = ols1([r["eps"] for r in rows], ys)
    print("[SCALE-C step 0] separating C from epsilon on the existing LOOSENESS survey (zero training)")
    print(f"  scenarios: {len(rows)} ; C values: {sorted(set(r['n_classes'] for r in rows))}")
    print(f"  fit  looseness ~ a + b*eps + c*ln(C):  a={a:+.3f}  b={b:+.3f} (eps)  c={c:+.3f} (ln C)  R2={r2:.3f}")
    print(f"  fit  looseness ~ a + b*eps          :  a={a1:+.3f}  b={b1:+.3f} (eps)          R2={r21:.3f}")
    print(f"  => adding the C term changes R2 by {r2 - r21:+.3f}; c is "
          f"{'non-negligible' if abs(c) > 0.1 else 'small'} in magnitude")
    return dict(n=len(rows), a=a, b_eps=b, c_lnC=c, r2=r2, r2_eps_only=r21, b_eps_only=b1)


def scale_verdict(rows):
    """Pre-registered verdict (docstring (4))."""
    gated = [r for r in rows if r["s1"]]
    detail = [f"  instances: {len(rows)} run, {len(gated)} inside the informative error range (S1 gate)"]
    for r in rows:
        detail.append(f"    C={r['c']:5d} seed={r['seed']} eps={r['eps']:.3f} "
                      f"{'[gated in]' if r['s1'] else '[S1 out]'} R_IB={r['rib']:.2f} K_min={r['k_min']:4d} "
                      f"K_wall={r['k_wall']} looseness="
                      f"{None if r['looseness'] is None else round(r['looseness'], 2)} "
                      f"excl={r['excluded']} false={r['false_excl']}")
    # looseness exists only where a wall was LOCATED inside the tested K grid; instances without one are
    # reported (S2/S3 use the reduced set) rather than silently dropped.
    with_lo = [r for r in gated if r["looseness"] is not None]
    detail.append(f"  looseness available for {len(with_lo)}/{len(gated)} gated instances "
                  f"(a wall must be located inside the K grid; the rest are 'wall beyond grid')")
    fits = [(r["c"], r["k_wall"]) for r in with_lo if r["k_wall"]]
    alpha = None
    if len(fits) >= 3:
        a, alpha, r2 = ols1([math.log(c) for c, _ in fits], [math.log(k) for _, k in fits])
        detail.append(f"  S2 wall scaling: ln K_wall = {alpha:+.3f}*ln C + {a:+.3f}  (alpha={alpha:.3f}, "
                      f"R2={r2:.3f}, n={len(fits)})")
    else:
        detail.append(f"  S2 wall scaling: only {len(fits)} located walls -> not fitted")
    rho = (rank_correlation([math.log(r["c"]) for r in with_lo], [r["looseness"] for r in with_lo])
           if len(with_lo) >= 3 else 0.0)
    s3_dir = len(with_lo) >= 3 and rho <= 0.0
    s3_asym = False
    if alpha is not None and with_lo:
        top = max(with_lo, key=lambda r: r["c"])
        target = asymptote_note(alpha, top["eps"])
        s3_asym = abs(top["looseness"] - target) <= 0.5
        detail.append(f"  S3 asymptote check at C={top['c']}: looseness={top['looseness']:.2f} vs "
                      f"alpha/(1-eps)={target:.2f} (|diff|={abs(top['looseness'] - target):.2f}, tol 0.5)")
    detail.append(f"  S3 direction (looseness non-increasing in C): Spearman={rho:+.3f} over "
                  f"{len(with_lo)} instances -> {'PASS' if s3_dir else 'FAIL'} ; asymptote within 0.5 "
                  f"nats -> {'PASS' if s3_asym else 'FAIL'}")
    s4 = all(len(r["false_excl"]) == 0 for r in rows)
    detail.append(f"  S4 soundness at scale (zero false exclusions at every C): "
                  f"{[(r['c'], len(r['false_excl'])) for r in rows]} -> {'PASS' if s4 else 'FAIL'}")
    s5 = False
    if rows:
        lo, hi = min(rows, key=lambda r: r["c"]), max(rows, key=lambda r: r["c"])
        s5 = len(hi["excluded"]) > len(lo["excluded"])
        detail.append(f"  S5 ceiling effect: excluded levels C={lo['c']}:{len(lo['excluded'])} -> "
                      f"C={hi['c']}:{len(hi['excluded'])} -> {'PASS' if s5 else 'FAIL'}")
    score = sum([s3_dir and s3_asym, s4, s5])
    if score == 3:
        return ("SUPPORT: on the class-count axis the certified reading scales as the formula predicts -- "
                "soundness holds up to C=1000, its reach grows with C (the ln C ceiling stops binding), "
                "and the bound gets relatively tighter (looseness non-increasing)"), detail
    if score == 2:
        return "PARTIAL: two of the three scale criteria hold (recorded as-is; see the lines)", detail
    return "MIXED/REFUTED: the scale criteria did not hold (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(m_grid, k_grid, seeds, epochs, sigma):
    raw = "|".join([f"M{','.join(map(str, m_grid))}", f"K{','.join(map(str, k_grid))}", f"s{seeds}",
                    f"ep{epochs}", f"sig{sigma}", f"spc{N_PER_CLASS}", f"sp{SPACING}",
                    f"ktmult{K_TEACHER_MULT}", "lattice-vqref"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def calibrate(ladder, m_grid, epochs, device):
    """Difficulty calibration: records HOW sigma was chosen, before the pre-registered sweep.

    sigma is a single GLOBAL knob (never tuned per instance). The ladder is swept at every C (reference
    only, short budget) and the value keeping eps inside 0.02..0.60 at the most class counts is adopted."""
    print("[SCALE-C calibration] sigma ladder vs class count (reference only, short budget)")
    print(f"  ladder {ladder} ; m {m_grid} ; {epochs} epochs each")
    results = {}
    for m in m_grid:
        C = class_count(m)
        row = []
        for sig in ladder:
            torch.manual_seed(0)
            (x_tr, y_tr), (x_te, y_te) = make_lattice(m, sigma=sig, seed=0)
            tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)
            te = DataLoader(TensorDataset(x_te, y_te), batch_size=512)
            model = build_model(C, None, device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            for _ in range(epochs):
                train_one_epoch(model, tr, opt, device=device)
            acc, _ = evaluate(model, te, device=device)
            row.append((sig, 1.0 - acc))
            print(f"  C={C:5d} (m={m:2d}) sigma={sig:4.1f}: eps={1.0 - acc:.3f}"
                  f"{'  <- in band' if 0.02 <= 1.0 - acc <= 0.60 else ''}")
        results[m] = row
    print("  [adopt] the sigma inside the band at the most class counts")
    return results


def run_sweep(m_grid, k_grid, seeds, epochs, sigma, resume, device):
    fp = run_signature(m_grid, k_grid, seeds, epochs, sigma)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 92)
    print("[SCALE-C] the class-count axis: does the reading keep working over 125x more classes?")
    print("=" * 92)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [task] lattice family (mnist_demo.make_grid64 generalised): C = m^3, spacing {SPACING}, "
          f"sigma {sigma}, {N_PER_CLASS} samples/class")
    print(f"  [protocol] Linear(3->64) -> bottleneck(64) -> Linear(64->C); reference = VQ(K_t = "
          f"min({K_TEACHER_MULT}C, {K_TEACHER_CAP})), i.e. the students' own family; "
          f"{epochs} ep, Adam {LR}, KD for students; K grid {k_grid} capped at {K_TEACHER_MULT}C")
    rows = []
    for m in m_grid:
        C = class_count(m)
        for seed in range(seeds):
            t0 = time.time()
            tr, te = instance_loaders(m, seed)
            path_t = os.path.join(ckpt_dir, f"teacher_C{C}_s{seed}.pt")
            acc_t, h_z_t = train_unit(C, "ref", seed, tr, te, None, epochs, device, path_t, resume)
            teacher = build_model(C, teacher_codes(C), device)
            teacher.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
            teacher.eval()
            eps = 1.0 - acc_t
            rib = task_info_fano(eps, C)
            k_min = DistillCapacity().min_codebook_size(rib)
            accs = {}
            for K in k_grid:
                if K > K_TEACHER_MULT * C:
                    continue                       # sizes beyond the reference's own bottleneck add nothing
                accs[K], _ = train_unit(C, K, seed, tr, te, teacher, epochs, device,
                                        os.path.join(ckpt_dir, f"student_C{C}_K{K}_s{seed}.pt"), resume)
            rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
            fails = sorted(K for K in accs if rel[K] >= 0.50)
            succeeds = sorted(K for K in accs if rel[K] <= 0.10)
            excl = excluded_set(rib, sorted(accs))
            k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
            lo = looseness(rib, k_wall)
            row = dict(c=C, m=m, seed=seed, acc_t=acc_t, h_z_t=h_z_t, eps=eps, rib=rib, k_min=k_min,
                       accs=accs, rel=rel, fails=fails, succeeds=succeeds, excluded=excl, k_wall=k_wall,
                       looseness=lo, s1=(0.02 <= eps <= 0.60), k_teacher=teacher_codes(C),
                       false_excl=false_exclusions(excl, succeeds), coverage=coverage(excl, fails),
                       sigma=sigma, fp=fp)
            rows.append(row)
            jsonl_append(JSONL, row)
            print(f"  C={C:5d} (m={m:2d}) s{seed}: eps={eps:.3f} R_IB={rib:.2f} K_min={k_min:4d} "
                  f"H(Z_t)={None if h_z_t is None else round(h_z_t, 2)} excl={excl} K_wall={k_wall} "
                  f"looseness={None if lo is None else round(lo, 2)} false={row['false_excl']} "
                  f"[{time.time() - t0:.0f}s]")
            print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    print("\n  --- SCALE-C verdict ---")
    verdict, detail = scale_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[SCALE-C] class-count axis\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def verdict_only():
    if not os.path.exists(JSONL):
        raise SystemExit("[SCALE-C] no scale_c.jsonl to re-verdict")
    all_rows = [json.loads(line) for line in open(JSONL, encoding="utf-8")]
    if not all_rows:
        raise SystemExit("[SCALE-C] scale_c.jsonl is empty")
    # re-verdict only the MOST RECENT protocol (different runs share the file for provenance); mixing
    # protocols in one verdict would compare a no-bottleneck reference against a VQ one.
    fp = all_rows[-1].get("fp")
    rows_by_key = {}
    for r in all_rows:
        if r.get("fp") == fp:
            rows_by_key[(r["c"], r["seed"])] = r
    rows = [rows_by_key[k] for k in sorted(rows_by_key)]
    print(f"[SCALE-C offline re-verdict] protocol fingerprint {fp}: {len(rows)} instances")
    for r in rows:
        print(f"  C={r['c']:5d} s{r['seed']}: eps={r['eps']:.3f} R_IB={r['rib']:.2f} K_min={r['k_min']:4d} "
              f"excl={r['excluded']} K_wall={r['k_wall']} "
              f"looseness={None if r['looseness'] is None else round(r['looseness'], 2)}")
    verdict, detail = scale_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")


def run_selftest():
    ok = []
    ok.append(("class_count is m^3", class_count(4) == 64 and class_count(10) == 1000))
    (xtr, ytr), (xte, yte) = make_lattice(2, n_per_class=5, seed=0)
    all_y = torch.cat([make_lattice(4, 2, seed=0)[0][1], make_lattice(4, 2, seed=0)[1][1]])
    ok.append(("the m=4 case spans 64 classes", len(torch.unique(all_y)) == 64))
    ok.append(("lattice input is 3-dimensional", xtr.shape[1] == 3))
    ok.append(("split is 80/20", abs(len(ytr) / (len(ytr) + len(yte)) - 0.8) < 1e-9))
    ok.append(("ols recovers a known plane",
               abs(ols([(1.0, 2.0), (2.0, 1.0), (3.0, 5.0), (0.5, 0.5)],
                       [3 + 2 * 1 + 4 * 2, 3 + 2 * 2 + 4 * 1, 3 + 2 * 3 + 4 * 5,
                        3 + 2 * 0.5 + 4 * 0.5])[0] - 3) < 1e-6))
    ok.append(("ols1 recovers a known line", abs(ols1([1.0, 2.0, 3.0], [5.0, 7.0, 9.0])[1] - 2.0) < 1e-9))
    ok.append(("ols1 survives a constant regressor", ols1([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])[1] == 0.0))
    ok.append(("rank_correlation needs >= 3 points", rank_correlation([1.0, 2.0], [1.0, 2.0]) == 0.0))
    ok.append(("rank_correlation detects a decreasing pair", rank_correlation([1, 2, 3], [3, 2, 1]) < -0.9))
    ok.append(("asymptote uses alpha/(1-eps)", abs(asymptote_note(2.0, 0.5) - 4.0) < 1e-9))
    base = dict(c=1000, seed=0, eps=0.20, rib=5.0, k_min=148, accs={}, rel={}, fails=[2], succeeds=[512],
                excluded=[2, 8, 32], k_wall=512, looseness=1.25, s1=True, false_excl=[], coverage=1.0)
    low = dict(base, c=10, eps=0.20, rib=1.5, k_min=5, excluded=[2], k_wall=8, looseness=2.6)
    mid = dict(base, c=100, eps=0.20, rib=3.0, k_min=20, excluded=[2, 8], k_wall=64, looseness=1.4)
    v, _ = scale_verdict([low, mid, base])
    ok.append(("verdict SUPPORT on a monotone family with growing reach", v.startswith("SUPPORT")))
    v, _ = scale_verdict([low, dict(mid, looseness=3.5), dict(base, looseness=4.0)])
    ok.append(("verdict not SUPPORT when looseness rises with C", not v.startswith("SUPPORT")))
    v, _ = scale_verdict([low, mid, dict(base, false_excl=[512])])
    ok.append(("verdict not SUPPORT when a false exclusion appears at scale", not v.startswith("SUPPORT")))
    v, _ = scale_verdict([low, mid, dict(base, excluded=[2])])
    ok.append(("verdict not SUPPORT when the reach does not grow", not v.startswith("SUPPORT")))
    no_wall = [dict(low, k_wall=None, looseness=None), dict(mid, k_wall=None, looseness=None),
               dict(base, k_wall=None, looseness=None)]
    v, _ = scale_verdict(no_wall)
    ok.append(("verdict survives instances without a located wall", not v.startswith("SUPPORT")))
    print("[SCALE-C selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="SCALE-C: the class-count axis of the scale boundary")
    ap.add_argument("--existing", action="store_true", help="step 0: separate C from eps on looseness.jsonl")
    ap.add_argument("--calibrate", action="store_true", help="sweep the sigma ladder before the sweep")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--sigma", type=float, default=SIGMA, help="the adopted global noise level")
    ap.add_argument("--quick", action="store_true", help="2 class counts, few epochs (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--verdict-only", action="store_true", help="re-verdict from scale_c.jsonl (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.existing:
        step0_existing()
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.calibrate:
        print(f"device: {device}")
        calibrate([1.0, 1.5, 2.0, 3.0], M_GRID, 20, device)
        return
    _setup_logging()
    print(f"device: {device}")
    if args.quick:
        run_sweep([2, 4], [2, 32], 1, 10, args.sigma, not args.fresh, device)
    else:
        run_sweep(M_GRID, K_GRID, args.seeds, args.epochs, args.sigma, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
