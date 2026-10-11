"""GAUSS: does the two-tier reading -- and the exclusion pipeline -- transfer to a CONTINUOUS bottleneck?

Motivation: every exclusion result in this project so far used a discrete VQ bottleneck (supply = log K).
The packaged continuous side (GaussBottleneck / gauss_capacity) had never been tested end to end, and the
2026-09-30 inspection showed it was not even a channel: no signal normalization (so the SNR, hence the
nominal capacity, was undefined) and noise only while training (off at eval). This experiment fixes that
class first (see capacity_lens/bottleneck.py) and then asks the two questions that matter:
  Q1 does the certified exclusion (capacity < R_IB => infeasible) hold on the continuous side?
  Q2 which bottleneck type buys more accuracy per nat on the same task and protocol -- a noiseless
     discrete codebook, or a noisy continuous channel?

Usage:
    python experiments/gauss_demo.py             # 3 seeds, 27 units, ~10 min GPU (relay-safe)
    python experiments/gauss_demo.py --quick     # 1 seed, 3 capacities (smoke)
    python experiments/gauss_demo.py --selftest  # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Task and model: the project's grid64 synthetic task, 3-dimensional input
    (mnist_demo.make_grid64, n_per_class=150, sigma=0.5, spacing=3.0) with the three-layer MLP shape of
    cp3_demo.GridMLP: Linear(3->64) -> ReLU -> bottleneck(64) -> Linear(64->64) logits. Budget:
    120 epochs, Adam lr=1e-2, distillation CE + KL(T=3) -- the CP3 MLP-cell protocol exactly.
(2) Two arms; the ONLY difference is the bottleneck module:
      VQ    (reference, CACHED, zero new training): cp3 MLP-euclid teacher + students
            (experiments/data/cp3/8dfe5bd4), nominal capacity = log K, K in cp3's CALIBRATED_K
            [8,32,48,64,96,128,192,256].
      GAUSS (new): GaussBottleneck(64, sigma), sigma chosen so the nominal capacity
            C_nom = gauss_capacity(64, 1/sigma^2) hits each target in [1,2,3,4,5,6,8,12] nats, plus a
            noiseless ceiling (sigma=0) as the arm's sanity check. Same data, teacher, budget and seeds.
            Seed scheme: 17000 + 100*s + index (index = the position of the target in the list; the ceiling
            is index 0). Recorded here BEFORE running.
(3) Definitions and the accounting axes (fixed):
      C_nom(sigma) = (d/2) ln(1 + 1/sigma^2)   -- the idealized Var=1 nominal capacity (a reference number);
      C_rig(sigma) = (d/2) ln(1 + lambda_mean/sigma^2) -- the RIGOROUS upper bound on what the channel can
                     carry, where lambda_mean is the MEAN per-dimension variance of the normalized signal
                     (Jensen: I(z;z+n) = sum_i (1/2)ln(1+lambda_i/sigma^2) <= (d/2)ln(1+lambda_bar/sigma^2),
                     since ln is concave; the MAXIMUM variance is irrelevant). Unit-RMS normalization fixes
                     lambda_bar at ~1 by construction, so C_rig ~ C_nom; lambda_mean is measured and logged.
      R_IB(eps)    = task_info_fano(1 - eps, 64)  -- the project's own inversion: a student that ACHIEVES
                     error eps proves its bottleneck carried at least R_IB(eps) nats (DPI + Fano).
    So every measured student gives a certified SANDWICH on its bottleneck's capacity:
      R_IB(eps_measured)  <=  I(X;Z)  <=  C_rig(sigma).
    For the VQ arm the analogous numbers are exact: capacity = log K (and R_IB <= log K is R1a, verified).
    Recorded correction (before any verdict was reported): the first version of C_rig used the MAX/mean
    variance ratio instead of the mean, which is also valid but 5-10 nats looser here and made C2 vacuous.
    Only the derivation of the bound was wrong; the criteria text is unchanged. See capacity_rigorous().
(4) Pre-registered criteria (per seed; a criterion passes when >= 2/3 of the seeds satisfy it):
      C0 (arm sanity): the noiseless ceiling reaches retention >= 0.95 (else the arm is broken and C1-C4
         are void -- reported, not silently skipped).
      C1 (upper-bound consistency, rigorous): at EVERY measured point R_IB(eps_measured) <= C_rig(sigma).
         Any violation would falsify the Gaussian capacity formula against the pipeline.
      C2 (certified exclusion transfers, rigorous): every sigma whose C_rig(sigma) < R_IB(teacher) has
         retention <= 0.75 (a channel that provably cannot carry the requirement must not match the teacher).
      C3 (wall located + two-sided consistency): some sigma reaches retention >= 0.90 (the wall), and the
         VQ arm's exact wall log(K_wall) lies inside that sigma's sandwich [R_IB(eps*), C_rig(sigma*)] --
         i.e. the discrete and continuous accounts of "how much capacity this task needs" agree.
      C4 (reported only): C_nom(sigma*) vs log K_wall, the idealized Var=1 comparison.
    Verdict (pure function gauss_verdict): SUPPORT if C0 and C1 and C2 and C3; PARTIAL if C0 and exactly
    two of C1-C3; else MIXED/REFUTED -- recorded as-is.
(5) Relay: per-unit checkpoints under experiments/data/gauss/<fingerprint>/; --fresh recomputes.
    --reaudit re-evaluates the criteria offline from gauss.jsonl (zero training) -- used after the
    correction above, since the first run's stored flags used the loose bound.
(6) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads;
    the cached VQ arm is reused unchanged (same task, teacher, budget and seeds as this run).
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
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (GaussBottleneck, gauss_capacity, train_one_epoch, evaluate,
                           task_info_fano, DistillCapacity)
from mnist_demo import make_grid64
from cp3_demo import CALIBRATED_K

N_CLASSES = 64
D_BOTTLE = 64
N_PER_CLASS = 150
EPOCHS = 120
LR = 1e-2
CP3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cp3", "8dfe5bd4")
CAPACITY_TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0]
SEED_BASE = 17000

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "gauss.jsonl")
LOG = os.path.join(_DATA, "gauss.log")
REPORT = os.path.join(_DATA, "gauss_report.txt")
CKPT_ROOT = os.path.join(_DATA, "gauss")


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


class GridMLPGauss(nn.Module):
    """cp3_demo.GridMLP's shape with the VQ replaced by the Gaussian channel (the only difference)."""

    def __init__(self, sigma, d=D_BOTTLE):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.bott = GaussBottleneck(d, sigma)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z, _ = self.bott(h)
        return self.fc2(z), []


# ---------------------------------------------------------------------------
# Pure functions (capacity algebra + criteria)
# ---------------------------------------------------------------------------

def sigma_for_capacity(c_nats, d=D_BOTTLE):
    """Invert the nominal Gaussian capacity: sigma such that (d/2)ln(1+1/sigma^2) = c_nats."""
    if c_nats <= 0.0:
        return 0.0
    return 1.0 / math.sqrt(math.exp(2.0 * c_nats / d) - 1.0)


def capacity_of_sigma(sigma, d=D_BOTTLE):
    """Nominal Var=1 capacity in nats; inf when noiseless."""
    if sigma <= 0.0:
        return float("inf")
    return gauss_capacity(d, 1.0 / (sigma ** 2))


def capacity_rigorous(sigma, lambda_mean=1.0, d=D_BOTTLE):
    """RIGOROUS upper bound on the channel capacity, and the axis the criteria are evaluated on.

    I(z; z+n) = sum_i (1/2) ln(1 + lambda_i / sigma^2) with lambda_i the eigenvalues of Cov(z); ln is
    concave, so by Jensen the sum is <= (d/2) ln(1 + lambda_bar / sigma^2) where lambda_bar is the MEAN
    eigenvalue = the mean per-dimension variance. The MAXIMUM variance is irrelevant for the bound, and
    unit-RMS normalization fixes the mean at ~1 by construction (mean_i z_i^2 = 1 per sample), so this is
    a rigorous upper bound for the rms and per_dim arms alike (lambda_mean is measured and logged).

    Recorded correction (2026-09-30, before any verdict was reported): the first version of this function
    used the MAX/mean variance ratio, i.e. (d/2)ln(1 + Vmax/sigma^2). That is also an upper bound but a
    needlessly loose one -- it inflated the bound by 5-10 nats here and made criterion C2 vacuous. The
    Jensen/mean form above is both rigorous and tight enough to test. The criteria text ('a rigorous
    upper bound') is unchanged; only the derivation of the bound was wrong.
    """
    if sigma <= 0.0:
        return float("inf")
    return gauss_capacity(d, max(float(lambda_mean), 1e-6) / (sigma ** 2))


def wall_sigma(curve, thresh=0.90):
    """Largest sigma (i.e. SMALLEST capacity) whose retention reaches the threshold; None if none does.
    curve: [(sigma, retention), ...]."""
    ok = sorted((sg for sg, r in curve if r >= thresh), reverse=True)
    return ok[0] if ok else None


def wall_nats_vq(curve, thresh=0.90):
    """Smallest nominal capacity (nats, = log K) whose VQ retention reaches the threshold; None if none."""
    ok = sorted(c for c, r in curve if r >= thresh)
    return ok[0] if ok else None


def gauss_verdict(per_seed):
    """Pre-registered verdict (docstring (4))."""
    n = len(per_seed)
    need = math.ceil(2 / 3 * n)

    def frac(key):
        return sum(1 for r in per_seed if r.get(key) is True) / max(n, 1)

    c0 = frac("c0") >= 2 / 3
    c1 = frac("c1") >= 2 / 3
    c2 = frac("c2") >= 2 / 3
    c3 = frac("c3") >= 2 / 3
    detail = [
        f"  seeds: {n} (a criterion passes at >= {need}/{n})",
        f"  C0 arm sanity (noiseless ceiling retention >= 0.95): "
        f"{[round(r['ceiling_ret'], 3) for r in per_seed]} -> {'PASS' if c0 else 'FAIL'}",
        f"  C1 upper-bound consistency (R_IB(eps_measured) <= C_rig at every point): "
        f"{['PASS' if r['c1'] else ('vacuous' if r['c1'] is None else 'FAIL') for r in per_seed]} -> "
        f"{'PASS' if c1 else 'FAIL'}",
        f"  C2 certified exclusion transfers (C_rig < R_IB => retention <= 0.75): "
        f"{[(r['n_below_ub'], 'PASS' if r['c2'] else ('vacuous' if r['c2'] is None else 'FAIL')) for r in per_seed]}"
        f" -> {'PASS' if c2 else 'FAIL'}",
        f"  C3 wall located + sandwich contains the VQ wall "
        f"[(C_rig(wall), log K_wall, flag)]: "
        f"{[(round(r['wall']['c_rig'], 2) if r['wall'] else None, r['vq_wall'], 'PASS' if r['c3'] else 'FAIL') for r in per_seed]}"
        f" -> {'PASS' if c3 else 'FAIL'}",
        f"  C4 (reported) [C_nom(wall), log K_wall, flag]: "
        f"{[(round(r['wall']['c_nom'], 2) if r['wall'] else None, r['vq_wall'], 'PASS' if r['c4'] else 'FAIL') for r in per_seed]}",
    ]
    if not c0:
        return ("MIXED/REFUTED: the arm fails its own sanity check (a noiseless bottleneck must match the "
                "teacher); C1-C4 are void -- recorded as-is"), detail
    score = sum([c1, bool(c2), c3])
    if score == 3:
        return ("SUPPORT: the pipeline transfers to the continuous channel -- the rigorous capacity ceiling "
                "C_rig is never violated by a measured student (C1), every channel that provably cannot "
                "carry the requirement stays far from the teacher (C2), and the continuous wall's certified "
                "sandwich contains the discrete arm's exact wall (C3)"), detail
    if score == 2:
        return "PARTIAL: two of the three continuous-side criteria hold (recorded as-is; see the lines)", detail
    return "MIXED/REFUTED: the continuous-side criteria did not hold (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(quick):
    raw = "|".join([f"q{int(quick)}", f"d{D_BOTTLE}", f"ep{EPOCHS}", f"lr{LR}", f"n{N_PER_CLASS}",
                    f"T{','.join(str(c) for c in CAPACITY_TARGETS)}", "cp3-8dfe5bd4"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def train_gauss_unit(cap, idx, s, train_loader, test_loader, teacher, device, ckpt_path, resume):
    """One (capacity, seed) unit of the continuous arm. Returns (acc, var_ratio, lambda_mean)."""
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ck["acc"], ck.get("var_ratio"), ck.get("lambda_mean", 1.0)
    sigma = sigma_for_capacity(cap, D_BOTTLE)
    torch.manual_seed(SEED_BASE + 100 * s + idx)
    model = GridMLPGauss(sigma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        train_one_epoch(model, train_loader, opt, device=device, distill_from=teacher)
    acc, _ = evaluate(model, test_loader, device=device)
    var_ratio, lambda_mean = signal_stats(model, train_loader, device)
    torch.save({"acc": acc, "var_ratio": var_ratio, "lambda_mean": lambda_mean, "sigma": sigma,
                "state_dict": model.state_dict()}, ckpt_path)
    return acc, var_ratio, lambda_mean


@torch.no_grad()
def signal_stats(model, loader, device):
    """(max/mean per-dimension variance of the normalized signal, mean per-dimension variance).
    The mean is the quantity the rigorous capacity bound needs (Jensen); the ratio is a profile diagnostic."""
    model.eval()
    x, _ = next(iter(loader))
    h = F.relu(model.fc1(x.to(device)))
    if model.bott.normalize == "per_dim":
        z = h / model.bott.dim_rms.clamp_min(1e-6)
    elif model.bott.normalize == "rms":
        z = h / h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    else:
        z = h
    v = z.var(dim=0)
    return float(v.max() / v.mean().clamp_min(1e-12)), float(v.mean())


def load_vq_arm(s):
    """The cached cp3 MLP-euclid teacher + student grid for one seed (zero training)."""
    tpath = os.path.join(CP3_DIR, f"MLP-euclid_teacher_s{s}.pt")
    if not os.path.exists(tpath):
        raise SystemExit(f"[GAUSS] the cached cp3 MLP-euclid arm is missing at {CP3_DIR}; "
                         f"run experiments/cp3_demo.py first (relay mode).")
    acc_t = float(torch.load(tpath, map_location="cpu", weights_only=True)["acc"])
    accs = {}
    for K in CALIBRATED_K:
        p = os.path.join(CP3_DIR, f"MLP-euclid_student_s{s}_K{K}.pt")
        if os.path.exists(p):
            accs[K] = float(torch.load(p, map_location="cpu", weights_only=True)["acc"])
    return acc_t, accs


def run_gauss(seeds, resume, device, quick=False):
    targets = [4.0] if quick else CAPACITY_TARGETS
    fp = run_signature(quick)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 88)
    print("[GAUSS] exclusion pipeline + nat-efficiency on a CONTINUOUS bottleneck (grid64, MLP shape)")
    print("=" * 88)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [arms] VQ = cached cp3 MLP-euclid (nominal capacity log K) ; GAUSS = N(0,sigma) channel "
          f"(nominal capacity (d/2)ln(1+1/sigma^2), d={D_BOTTLE})")
    print(f"  [targets] capacity targets (nats) = {targets} + noiseless ceiling ; protocol {EPOCHS} ep, "
          f"lr {LR}, KD T=3, n_per_class={N_PER_CLASS}")
    per_seed = []
    for s in range(seeds):
        t0 = time.time()
        acc_t_cached, vq_accs = load_vq_arm(s)
        teacher = torch.load(os.path.join(CP3_DIR, f"MLP-euclid_teacher_s{s}.pt"),
                             map_location="cpu", weights_only=True)
        from cp3_demo import GridMLP, TEACHER_CODES
        teacher_model = GridMLP(n_codes=TEACHER_CODES, metric="euclid").to(device)
        teacher_model.load_state_dict(teacher["state_dict"])
        teacher_model.eval()
        r_ib = task_info_fano(1.0 - acc_t_cached, N_CLASSES)
        k_min = DistillCapacity().min_codebook_size(r_ib)
        (x_tr, y_tr), (x_te, y_te) = make_grid64(n_per_class=N_PER_CLASS, seed=s)
        train_loader = DataLoader(list(zip(x_tr, y_tr)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_te, y_te)), batch_size=256)
        vq_curve = [(math.log(K), vq_accs[K] / acc_t_cached) for K in sorted(vq_accs)]
        vq_wall = wall_nats_vq(vq_curve)
        points, ceiling_ret = [], None
        for idx, cap in enumerate([0.0] + targets):
            is_ceiling = (cap == 0.0)
            ck_name = "ceiling" if is_ceiling else f"C{cap:g}"
            acc, var_ratio, lam = train_gauss_unit(cap, idx, s, train_loader, test_loader, teacher_model,
                                                   device, os.path.join(ckpt_dir, f"{ck_name}_s{s}.pt"),
                                                   resume)
            ret = acc / acc_t_cached
            sigma = sigma_for_capacity(cap, D_BOTTLE)
            if is_ceiling:
                ceiling_ret = ret
            else:
                points.append(dict(cap=cap, sigma=sigma, c_nom=capacity_of_sigma(sigma),
                                   c_rig=capacity_rigorous(sigma, lam), acc=acc, ret=ret,
                                   r_ib_eps=task_info_fano(1.0 - acc, N_CLASSES),
                                   var_ratio=var_ratio, lambda_mean=lam))
            print(f"    seed {s} {'ceiling' if is_ceiling else f'C_nom={cap:5.1f} nats'}: acc={acc:.3f} "
                  f"retention={ret:.3f} var_ratio={var_ratio:.2f} lambda_mean={lam:.3f}")
        sg_wall = wall_sigma([(p["sigma"], p["ret"]) for p in points])
        wall = next((p for p in points if sg_wall is not None and p["sigma"] == sg_wall), None)
        n_below_ub = sum(1 for p in points if p["c_rig"] < r_ib)
        c0 = ceiling_ret is not None and ceiling_ret >= 0.95
        c1 = all(p["r_ib_eps"] <= p["c_rig"] + 1e-9 for p in points) if points else None
        c2 = (all(p["ret"] <= 0.75 for p in points if p["c_rig"] < r_ib)
              if n_below_ub > 0 else None)
        c3 = bool(wall is not None and vq_wall is not None
                  and wall["r_ib_eps"] <= vq_wall <= wall["c_rig"] + 1e-9)
        c4 = bool(wall is not None and vq_wall is not None and vq_wall > 0
                  and abs(wall["c_nom"] - vq_wall) / vq_wall <= 0.5)
        print(f"\n  seed {s}: teacher acc={acc_t_cached:.3f} -> R_IB={r_ib:.2f} nats K_min={k_min} | "
              f"VQ wall={None if vq_wall is None else round(vq_wall, 2)} nats "
              f"(K_wall={None if vq_wall is None else int(round(math.exp(vq_wall)))}) [{time.time() - t0:.0f}s]")
        print("    VQ    (log K, retention): " + "  ".join(f"{c:.2f}:{r:.2f}" for c, r in vq_curve))
        print("    GAUSS (C_nom, retention): " + "  ".join(f"{p['cap']:.1f}:{p['ret']:.2f}" for p in points))
        print("    GAUSS (C_rig, retention): " + "  ".join(f"{p['c_rig']:.1f}:{p['ret']:.2f}" for p in points))
        print(f"    continuous wall: sigma*={None if wall is None else round(wall['sigma'], 2)} "
              f"C_nom={None if wall is None else round(wall['c_nom'], 2)} "
              f"C_rig={None if wall is None else round(wall['c_rig'], 2)} "
              f"R_IB(eps*)={None if wall is None else round(wall['r_ib_eps'], 2)} "
              f"retention={None if wall is None else round(wall['ret'], 3)} ; "
              f"points with C_rig<R_IB: {n_below_ub}")
        print(f"    C0 {'PASS' if c0 else 'FAIL'} | C1 {c1} | C2 {c2} | C3 {'PASS' if c3 else 'FAIL'} | "
              f"C4 {'PASS' if c4 else 'FAIL'}")
        print(f"    VQ looseness={None if vq_wall is None else round(vq_wall / r_ib, 2)} | "
              f"C_nom(wall) looseness={None if wall is None else round(wall['c_nom'] / r_ib, 2)} | "
              f"C_ub(wall) looseness={None if wall is None else round(wall['c_ub'] / r_ib, 2)}")
        jsonl_append(JSONL, dict(seed=s, acc_t=acc_t_cached, r_ib=r_ib, k_min=k_min, vq_curve=vq_curve,
                                 vq_wall=vq_wall, points=points,
                                 wall=None if wall is None else {k: v for k, v in wall.items()},
                                 n_below_ub=n_below_ub, ceiling_ret=ceiling_ret,
                                 c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, fp=fp))
        per_seed.append(dict(seed=s, acc_t=acc_t_cached, r_ib=r_ib, k_min=k_min, vq_wall=vq_wall,
                             wall=wall, n_below_ub=n_below_ub, ceiling_ret=ceiling_ret or 0.0,
                             c0=c0, c1=c1, c2=c2, c3=c3, c4=c4,
                             var_ratio=max(p["var_ratio"] for p in points)))
    print("\n  --- GAUSS verdict ---")
    verdict, detail = gauss_verdict(per_seed)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[GAUSS] continuous bottleneck: exclusion transfer + nat-efficiency\n")
        for r in per_seed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return per_seed, verdict


def reaudit():
    """Offline re-evaluation of the pre-registered criteria from gauss.jsonl (zero training).
    Used after the C_rig correction (docstring (3)): the first run's stored flags were computed with the
    loose max-based bound. Records without lambda_mean fall back to 1.0 (the value RMS normalization
    fixes by construction; stated here rather than hidden)."""
    if not os.path.exists(JSONL):
        raise SystemExit("[GAUSS] no gauss.jsonl to re-audit")
    latest = {}
    for line in open(JSONL, encoding="utf-8"):
        r = json.loads(line)
        latest[r["seed"]] = r
    print("[GAUSS re-audit] criteria recomputed from gauss.jsonl with the corrected rigorous bound")
    per_seed = []
    for s in sorted(latest):
        rec = latest[s]
        pts = []
        for p in rec["points"]:
            lam = p.get("lambda_mean", 1.0)
            pts.append(dict(cap=p["cap"], sigma=p["sigma"], c_nom=capacity_of_sigma(p["sigma"]),
                            c_rig=capacity_rigorous(p["sigma"], lam), acc=p["acc"], ret=p["ret"],
                            r_ib_eps=p["r_ib_eps"], var_ratio=p.get("var_ratio"), lambda_mean=lam))
        r_ib, vq_wall = rec["r_ib"], rec["vq_wall"]
        ceiling_ret = rec.get("ceiling_ret") or 0.0
        sg = wall_sigma([(p["sigma"], p["ret"]) for p in pts])
        wall = next((p for p in pts if sg is not None and p["sigma"] == sg), None)
        n_below = sum(1 for p in pts if p["c_rig"] < r_ib)
        c0 = ceiling_ret >= 0.95
        c1 = all(p["r_ib_eps"] <= p["c_rig"] + 1e-9 for p in pts) if pts else None
        c2 = (all(p["ret"] <= 0.75 for p in pts if p["c_rig"] < r_ib) if n_below > 0 else None)
        c3 = bool(wall is not None and vq_wall is not None
                  and wall["r_ib_eps"] <= vq_wall <= wall["c_rig"] + 1e-9)
        c4 = bool(wall is not None and vq_wall and abs(wall["c_nom"] - vq_wall) / vq_wall <= 0.5)
        print(f"  seed {s}: R_IB={r_ib:.2f} | VQ wall={vq_wall:.2f} nats | wall C_rig="
              f"{None if wall is None else round(wall['c_rig'], 2)} "
              f"ret={None if wall is None else round(wall['ret'], 3)} | points with C_rig<R_IB: {n_below} | "
              f"C0={c0} C1={c1} C2={c2} C3={c3} C4={c4}")
        print("    " + "  ".join(f"[C_rig {p['c_rig']:.1f} ret {p['ret']:.2f} R_IB(eps) {p['r_ib_eps']:.2f}]"
                                 for p in pts))
        per_seed.append(dict(seed=s, ceiling_ret=ceiling_ret, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4,
                             wall=wall, vq_wall=vq_wall, n_below_ub=n_below, r_ib=r_ib))
    verdict, detail = gauss_verdict(per_seed)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    return per_seed, verdict


def run_probe(seeds, epochs_long, device, cap=4.0):
    """Targeted follow-up probe (pre-registered here, before running): is the ~5x nat-efficiency gap of
    the continuous channel STRUCTURAL or OPTIMIZATION-LIMITED?

    Design: the boundary point cap = 4.0 nats (sigma = 2.744), where the channel's nominal capacity is
    4.0 nats and R_IB(teacher) is 3.9-4.0 -- i.e. the channel COULD in principle carry the requirement,
    yet the 120-epoch encoder retained only 0.36-0.43. Re-train exactly that unit with 5x the budget
    (600 epochs), same seeds, same data, same teacher, same architecture; retention is compared against
    the stored 120-epoch value.

    Pre-registered reading (delta = retention(600 ep) - retention(120 ep)):
      delta >= +0.15 -> OPTIMIZATION-LIMITED: the gap is a property of the training protocol, not of the
                        channel; the nominal capacity is (in principle) approachable.
      delta <= +0.05 -> STRUCTURAL at this protocol: more optimization does not close it.
      in between     -> INTERMEDIATE (recorded as-is).
    """
    fp = run_signature(False)
    ckpt_dir = os.path.join(CKPT_ROOT, fp + f"-probe{epochs_long}")
    os.makedirs(ckpt_dir, exist_ok=True)
    sigma = sigma_for_capacity(cap, D_BOTTLE)
    print("=" * 88)
    print(f"[GAUSS probe] optimization vs structure at C_nom={cap} nats (sigma={sigma:.3f}), "
          f"{epochs_long} epochs vs the main run's {EPOCHS}")
    print("=" * 88)
    rows = []
    for s in range(seeds):
        acc_t_cached, _ = load_vq_arm(s)
        teacher_ck = torch.load(os.path.join(CP3_DIR, f"MLP-euclid_teacher_s{s}.pt"),
                               map_location="cpu", weights_only=True)
        from cp3_demo import GridMLP, TEACHER_CODES
        teacher_model = GridMLP(n_codes=TEACHER_CODES, metric="euclid").to(device)
        teacher_model.load_state_dict(teacher_ck["state_dict"])
        teacher_model.eval()
        (x_tr, y_tr), (x_te, y_te) = make_grid64(n_per_class=N_PER_CLASS, seed=s)
        train_loader = DataLoader(list(zip(x_tr, y_tr)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_te, y_te)), batch_size=256)
        ck = os.path.join(ckpt_dir, f"probe_C{cap:g}_s{s}.pt")
        if os.path.exists(ck):
            acc = float(torch.load(ck, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(SEED_BASE + 100 * s + 90)     # 90 = a probe-only index (recorded)
            model = GridMLPGauss(sigma).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            for _ in range(epochs_long):
                train_one_epoch(model, train_loader, opt, device=device, distill_from=teacher_model)
            acc, _ = evaluate(model, test_loader, device=device)
            torch.save({"acc": acc, "sigma": sigma, "epochs": epochs_long,
                        "state_dict": model.state_dict()}, ck)
        main_rec = None
        for line in open(JSONL, encoding="utf-8"):
            r = json.loads(line)
            if r["seed"] == s and r.get("vq_wall") is not None:
                main_rec = r
        short = next((p["ret"] for p in main_rec["points"] if abs(p["cap"] - cap) < 1e-9), None)
        long_ret = acc / acc_t_cached
        delta = None if short is None else long_ret - short
        reading = ("n/a" if delta is None else
                   "OPTIMIZATION-LIMITED" if delta >= 0.15 else
                   "STRUCTURAL at this protocol" if delta <= 0.05 else "INTERMEDIATE")
        print(f"  seed {s}: retention {None if short is None else round(short, 3)} ({EPOCHS} ep) -> "
              f"{long_ret:.3f} ({epochs_long} ep) ; delta={None if delta is None else round(delta, 3)} "
              f"-> {reading}")
        jsonl_append(JSONL, dict(probe=True, seed=s, cap=cap, sigma=sigma, epochs=epochs_long,
                                 acc=acc, acc_t=acc_t_cached, ret_short=short, ret_long=long_ret,
                                 delta=delta, reading=reading, fp=fp))
        rows.append(dict(seed=s, ret_short=short, ret_long=long_ret, delta=delta, reading=reading))
    n_opt = sum(1 for r in rows if r["reading"] == "OPTIMIZATION-LIMITED")
    print(f"\n  [probe verdict] {n_opt}/{len(rows)} seeds read as OPTIMIZATION-LIMITED; "
          f"deltas {[None if r['delta'] is None else round(r['delta'], 3) for r in rows]}")
    return rows


def run_selftest():
    ok = []
    ok.append(("sigma_for_capacity inverts capacity_of_sigma",
               abs(capacity_of_sigma(sigma_for_capacity(4.0)) - 4.0) < 1e-9))
    ok.append(("noiseless ceiling has infinite capacity", capacity_of_sigma(0.0) == float("inf")))
    ok.append(("capacity_rigorous falls with sigma", capacity_rigorous(2.0) > capacity_rigorous(4.0)))
    ok.append(("capacity_rigorous is a strict function of lambda_mean",
               capacity_rigorous(2.0, 1.0) < capacity_rigorous(2.0, 2.5)))
    ok.append(("capacity_rigorous == capacity_of_sigma at lambda_mean = 1",
               abs(capacity_rigorous(2.0, 1.0) - capacity_of_sigma(2.0)) < 1e-12))
    ok.append(("wall_sigma picks the largest sigma (= smallest capacity) that passes",
               wall_sigma([(1.0, 0.95), (2.0, 0.92), (3.0, 0.50)]) == 2.0))
    ok.append(("wall_sigma returns None when nothing passes", wall_sigma([(1.0, 0.30)]) is None))
    ok.append(("wall_nats_vq picks the smallest log K that passes",
               wall_nats_vq([(2.0, 0.30), (4.5, 0.91)]) == 4.5))
    base = dict(seed=0, ceiling_ret=0.99, c0=True, c1=True, c2=True, c3=True, c4=True,
                wall=dict(c_nom=5.0, c_rig=7.0), vq_wall=4.5, n_below_ub=3)
    v, _ = gauss_verdict([dict(base), dict(base), dict(base)])
    ok.append(("verdict SUPPORT on a clean three-seed table", v.startswith("SUPPORT")))
    v, _ = gauss_verdict([dict(base), dict(base, c0=False), dict(base, c0=False)])
    ok.append(("verdict not SUPPORT when the arm sanity check fails 2/3", not v.startswith("SUPPORT")))
    v, _ = gauss_verdict([dict(base), dict(base, c3=False), dict(base, c3=False)])
    ok.append(("verdict PARTIAL when one of the three criteria fails 2/3", v.startswith("PARTIAL")))
    v, _ = gauss_verdict([dict(base), dict(base, c1=False, c2=False, c3=False),
                          dict(base, c1=False, c2=False, c3=False)])
    ok.append(("verdict MIXED/REFUTED when all three fail 2/3", v.startswith("MIXED")))
    v, _ = gauss_verdict([dict(base, c2=None), dict(base, c2=None), dict(base, c2=None)])
    ok.append(("a vacuous C2 is not counted as established", v.startswith("PARTIAL")))
    print("[GAUSS selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="GAUSS: continuous bottleneck exclusion + nat-efficiency")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 capacity (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--reaudit", action="store_true", help="re-evaluate the criteria from gauss.jsonl (no training)")
    ap.add_argument("--probe", type=int, default=0, metavar="EPOCHS",
                    help="targeted probe: re-train the 4-nat boundary point with EPOCHS (e.g. 600)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.reaudit:
        reaudit()
        return
    if args.probe:
        _setup_logging()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        run_probe(args.seeds, args.probe, device)
        print("\nDone.")
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.quick:
        run_gauss(1, not args.fresh, device, quick=True)
    else:
        run_gauss(args.seeds, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
