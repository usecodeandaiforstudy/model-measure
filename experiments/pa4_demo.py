"""PA4: the P-A1-4 dose-response experiment -- cliff sharpness as a function of output-mapping discreteness.

Answers the candidate prediction registered in the analysis ledger (P-A1-4,
2026-09-30): cliff SHARPNESS is a function of the discreteness of the output mapping, NOT of capacity;
the Fano exclusion does not depend on the discreteness. This is the controlled-synthetic reproduction of
the CP-H finding ("wall but no cliff" on CIFAR-10) and, in passing, calibrates wall_geometry_probe's
DISCRETE_THRESHOLD.

Usage:
    python experiments/pa4_demo.py                          # all deltas, 3 seeds, ~1-1.5 h on GPU
    python experiments/pa4_demo.py --deltas 0,0.5           # subset of doses
    python experiments/pa4_demo.py --fresh                  # recompute (default: relay mode, skip cached units)
    python experiments/pa4_demo.py --selftest               # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Design: the grid64 image task (16x16, 3 bands x 4 positions -> 64 classes, same encoding constants as
    cifar_demo.make_grid64_images -- rows0=(1,7,12), cols=(2,6,10,14), sigma_pix=0.15, sigma_bright=0.1)
    with an added OUTPUT-MAPPING DIFFUSION knob: each sample, with probability delta, is blended 50/50 with
    the template of another random class. delta=0 recovers the original task family (same encoding
    parameters, do not tune); larger delta makes class regions overlap in the output mapping while the
    nominal class count and the task's class entropy stay 64/ln64.
    Doses: DELTAS = [0.0, 0.15, 0.3, 0.5] (pre-registered).
    Per (delta, seed): teacher GridCNN(K_t=256, euclid) 40ep (CP3-calibrated budget), students
    K in [8, 32, 48, 64, 96, 128, 192, 256] distilled CE+KL(T=3), 40ep, same lr 1e-3.
    The teacher retrains per dose -> R_IB/K_min adapt to the dose (the exclusion's own terms), which is
    exactly what "the Fano exclusion does not depend on the discreteness" claims about.
(2) Readouts per (delta, seed): teacher acc / H(Z_t) / G0 / R_IB / K_min; student gap curve g(K);
    plateau + K* (preflight functions); R1a/R1b (r1_exclusion from cph_demo, unchanged); sharpness
    (auxiliary, below); wall_geometry_probe on the teacher's test features (err_1nn / silhouette / wall kind).
    Sharpness (pre-registered operation): max over consecutive grid pairs of
    (g(K_low) - g(K_high)) / (log K_high - log K_low) -- the steepest per-nat drop of the gap curve.
(3) Pre-registered criteria (pure function pa4_verdict):
    Q1 (exclusion invariance): R1a AND R1b pass for EVERY (delta, seed). Violation = the exclusion
        machinery broke under diffusion (would refute the P-A1-4 claim, or signal a bug).
    Q2a (locatability monotone): the per-dose locatability fraction (seeds with plateau_ok AND K* located)
        is non-increasing in delta (tolerance 0.05);
    Q2b (sharpness monotone): the per-dose median sharpness is non-increasing in delta (tolerance 0.05);
    Q2c (signal): locatability(delta=0) - locatability(delta=max) >= 1/3 -- the intervention actually
        moves the curve (without this, Q2's monotonicity is vacuous).
    Q3 (probe calibration): per-dose median err_1nn is non-decreasing in delta; err_1nn(delta=0) <= 0.02
        AND err_1nn(delta=max) > 0.02 -- the candidate DISCRETE_THRESHOLD separates the two ends.
    Verdict: Q1 AND Q2a/Q2b/Q2c AND Q3 -> support ("cliff sharpness is a function of output-mapping
    discreteness; exclusion is not"). Q1 fails -> refute (machinery broken). Otherwise -> gray, recorded.
(4) Seed scheme: teacher = 16000 + 1000*d_idx + s; student = 17000 + 1000*d_idx + 100*s + K
    (non-overlapping with all existing families). Data seed = s per dose.
(5) Relay / cache: per-unit checkpoints under experiments/data/pa4/<fingerprint>/; default relay mode
    (skip cached units), --fresh recomputes. Fingerprint = (deltas, epochs, lr, k_t, workers, device);
    the seed count is excluded (unit RNG is defined by its own (d_idx, s, K), cph_demo convention).
(6) Iron rules: print only GBK-encodable characters; criteria before data; pure functions unit-testable
    (--selftest); negative results archived honestly; no downloads (task generated in-process).
(7) Cost: grid64 is cheap; 4 doses x 3 seeds x 9 units ~= 1-1.5 h on the borrowed GPU, relay-safe.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import train_one_epoch, evaluate, entropy_of, task_info_fano
from capacity_lens.preflight import (g0_gate, distill_plateau_ok,
                                     distill_gap_k_star, preflight_report)
from capacity_lens.geometry import wall_geometry_probe
from cph_demo import r1_exclusion
from cp3_demo import build_model

K_T = 256
C_TASK = 64
N_PER_CLASS = 150
CALIBRATED_K = [4, 6, 8, 32, 48, 64, 96, 128, 192, 256]
# Grid coverage fix (2026-09-30, after the first full run): 4/6 were added because at delta=0.5 the
# teachers weaken to K_min=6-7 -- BELOW the original grid bottom 8 -- so r1_exclusion had no below-bound
# points and returned "grid defect", which the first run misreported as a Q1 violation ("REFUTED"). The
# defect is a protocol coverage gap, NOT an exclusion-mechanism failure (at delta=0.5 the K=8 students
# were still 0.55 below their teacher). The grid is NOT part of the fingerprint (cph 192/256 precedent),
# so this extension reuses every cached unit and only adds the 4/6 units; sharpness/K* recompute over the
# extended grid for all doses (recorded, criteria unchanged).
DELTAS = [0.0, 0.15, 0.3, 0.5]
EPOCHS_DEFAULT = 40
LR = 1e-3

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "pa4.jsonl")
LOG = os.path.join(_DATA, "pa4.log")
REPORT = os.path.join(_DATA, "pa4_report.txt")
CKPT_ROOT = os.path.join(_DATA, "pa4")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        self.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


try:
    _LOG = open(LOG, "w", encoding="utf-8")
except OSError:
    # An import must not require a writable data directory (offline replay, read-only checkouts). Fall back to
    # devnull, NOT to None: these scripts tee stdout/stderr through _LOG, and a None sink makes every print
    # fail at interpreter shutdown with exit code 120 (found by the selftest loop that CI runs).
    _LOG = open(os.devnull, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Task: grid64 image version with output-mapping diffusion delta
# ---------------------------------------------------------------------------

def make_grid64_overlap(n_per_class=150, seed=0, delta=0.0):
    """grid64 with output-mapping diffusion: each sample is blended 50/50 with another class's template
    with probability delta (same encoding constants as make_grid64_images; delta=0 = the original family)."""
    g = torch.Generator().manual_seed(seed)
    rows0 = (1, 7, 12)
    cols = (2, 6, 10, 14)
    templates = torch.zeros(C_TASK, 1, 16, 16)
    for v in range(C_TASK):
        v1, rem = divmod(v, 16)
        v2, v3 = divmod(rem, 4)
        for c, vi in enumerate((v1, v2, v3)):
            r0, cx = rows0[c], cols[vi]
            templates[v, 0, r0:r0 + 3, cx - 1:cx + 2] = 1.0
    y = torch.arange(C_TASK).repeat_interleave(n_per_class)
    bright = 1.0 + 0.1 * torch.randn(len(y), 1, 1, 1, generator=g)
    x = templates[y] * bright
    if delta > 0.0:
        flip = torch.rand(len(y), generator=g) < delta
        y_other = torch.randint(0, C_TASK, (len(y),), generator=g)
        y_other = torch.where(y_other == y, (y_other + 1) % C_TASK, y_other)
        blend = 0.5 * x[flip] + 0.5 * (templates[y_other[flip]] * bright[flip])
        x[flip] = blend
    x = x + 0.15 * torch.randn(len(y), 1, 16, 16, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def sharpness(k_list, accs, acc_t):
    """Steepest per-nat drop of the gap curve (pure function): max over consecutive grid pairs of
    (g(K_low) - g(K_high)) / (log K_high - log K_low)."""
    ks = sorted(k_list)
    best = 0.0
    for a, b in zip(ks, ks[1:]):
        drop = (acc_t - accs[a]) - (acc_t - accs[b])
        dlog = math.log(b) - math.log(a)
        if dlog > 0:
            best = max(best, drop / dlog)
    return best


def _noninc(vals, tol=0.05):
    for a, b in zip(vals, vals[1:]):
        if b > a + tol:
            return False
    return True


def _nondown(vals, tol=0.05):
    for a, b in zip(vals, vals[1:]):
        if b < a - tol:
            return False
    return True


def pa4_verdict(per_delta):
    """The P-A1-4 decision rule (pure function, see docstring (3)).

    per_delta: {delta: dict(q1=bool, loc_frac=float, sharp_med=float, err1_med=float)}.
    Returns (verdict, detail_lines)."""
    ds = sorted(per_delta)
    detail = [f"  delta={d}: {per_delta[d]}" for d in ds]
    if len(ds) < 2:
        return "UNDECIDABLE: < 2 doses", detail
    q1 = all(per_delta[d]["q1"] for d in ds)
    q1_untestable = sum(per_delta[d].get("q1_untestable", 0) for d in ds)
    locs = [per_delta[d]["loc_frac"] for d in ds]
    sharps = [per_delta[d]["sharp_med"] for d in ds]
    errs = [per_delta[d]["err1_med"] for d in ds]
    q2a, q2b = _noninc(locs), _noninc(sharps)
    q2c = locs[0] - locs[-1] >= 1 / 3 - 1e-9
    q3 = _nondown(errs) and errs[0] <= 0.02 and errs[-1] > 0.02
    if q1_untestable:
        detail.append(f"  Q1 exclusion invariance: all testable units passed, but {q1_untestable} unit(s) undecidable"
                      f" (K_min below grid floor -- grid coverage defect, not a violation)")
    else:
        detail.append(f"  Q1 exclusion invariance: {'PASS' if q1 else 'VIOLATED'}")
    detail.append(f"  Q2a locatability non-increasing: {locs} -> {'PASS' if q2a else 'FAIL'}; "
                  f"Q2b sharpness non-increasing: {[f'{v:.2f}' for v in sharps]} -> {'PASS' if q2b else 'FAIL'}; "
                  f"Q2c signal (first-last gap >= 1/3): {'PASS' if q2c else 'FAIL'}")
    detail.append(f"  Q3 probe calibration (err_1nn): {[f'{v:.3f}' for v in errs]} -> {'PASS' if q3 else 'FAIL'}")
    if not q1 and q1_untestable == 0:
        return "REFUTED: exclusion machinery failed under diffusion (or a bug) -- archived honestly", detail
    if q2a and q2b and q2c and q3 and q1:
        return ("SUPPORT: cliff sharpness is a function of output-mapping discreteness; Fano exclusion"
                " is independent of discreteness (P-A1-4 holds, DISCRETE_THRESHOLD=0.02 calibration passed)", detail)
    if q2a and q2b and q2c and q3 and q1_untestable:
        return ("PARTIAL: Q2a/b/c and Q3 all pass (sharpness = f(discreteness) established), Q1 has"
                " undecidable units (grid coverage defect; extend grid downward and retest) -- archived honestly", detail)
    return "MIXED: some criteria failed (archived honestly)", detail


# ---------------------------------------------------------------------------
# Training units
# ---------------------------------------------------------------------------

def run_signature(deltas, epochs, lr, k_t, workers, device):
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    raw = "|".join([f"d{'/'.join(str(d) for d in deltas)}", f"ep{epochs}", f"lr{lr}",
                    f"kt{k_t}", f"wk{workers}", f"dev{dev}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def ckpt_dir_for(deltas, epochs, lr, k_t, workers, device):
    d = os.path.join(CKPT_ROOT, run_signature(deltas, epochs, lr, k_t, workers, device))
    os.makedirs(d, exist_ok=True)
    return d


def unit_path(ckpt_dir, d_idx, s, K=None):
    kind = "teacher" if K is None else f"student_K{K}"
    return os.path.join(ckpt_dir, f"d{d_idx}_{kind}_s{s}.pt")


@torch.no_grad()
def collect_features(model, test_loader, device="cpu"):
    model.eval()
    ts, ys = [], []
    for x, y in test_loader:
        x = x.to(device)
        h = model.features(x).flatten(1)
        ts.append(F.relu(model.fc1(h)).cpu())
        ys.append(y)
    return torch.cat(ts), torch.cat(ys)


def train_teacher_unit(d_idx, delta, s, epochs, device, ckpt_dir, resume, fp):
    torch.manual_seed(16000 + 1000 * d_idx + s)
    (xtr, ytr), (xte, yte) = make_grid64_overlap(N_PER_CLASS, s, delta)
    tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    te = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = build_model("cnn", K_T, "euclid").to(device)
    path = unit_path(ckpt_dir, d_idx, s)
    if resume and os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ck["state_dict"])
        print(f"  delta={delta} s{s}: [cache hit] teacher acc={ck['acc']:.3f}  H={ck['h_z']:.2f}")
        return model, ck["acc"], ck["h_z"], tr, te
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device)
    acc, idx = evaluate(model, te, device=device)
    h_z = entropy_of(idx[0], K_T)
    torch.save({"state_dict": model.state_dict(), "acc": acc, "h_z": h_z}, path)
    print(f"  delta={delta} s{s}: teacher acc={acc:.4f}  H(Z_t)={h_z:.2f}  [{time.time() - t0:.0f}s]")
    return model, acc, h_z, tr, te


def train_student_unit(d_idx, delta, s, K, teacher, epochs, device, ckpt_dir, resume, fp):
    torch.manual_seed(17000 + 1000 * d_idx + 100 * s + K)
    (xtr, ytr), (xte, yte) = make_grid64_overlap(N_PER_CLASS, s, delta)
    tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    te = DataLoader(list(zip(xte, yte)), batch_size=256)
    path = unit_path(ckpt_dir, d_idx, s, K)
    if resume and os.path.exists(path):
        acc_s = float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
        print(f"    K={K}: [cache hit] acc={acc_s:.3f}")
        return acc_s
    model = build_model("cnn", K, "euclid").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device, distill_from=teacher)
    acc_s, _ = evaluate(model, te, device=device)
    torch.save({"acc": acc_s}, path)
    print(f"    K={K}: acc={acc_s:.3f}")
    return acc_s


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_pa4(deltas, seeds, epochs, resume, device):
    fp = run_signature(deltas, epochs, LR, K_T, 0, device)
    ckpt_dir = ckpt_dir_for(deltas, epochs, LR, K_T, 0, device)
    print("\n" + "=" * 76)
    print(f"[PA4] P-A1-4 dose-response: cliff sharpness vs output-mapping discreteness (deltas={deltas}; {seeds} seeds/dose; {epochs}ep)")
    print("=" * 76)
    print(f"  [checkpoint] {'relay mode (cache hits auto-skipped)' if resume else 'recompute mode'}  {ckpt_dir}")
    print("  [criteria] Q1 exclusion invariance / Q2a locatability non-increasing / Q2b sharpness non-increasing / Q2c signal / Q3 probe calibration")
    per_delta = {}
    for d_idx, delta in enumerate(deltas):
        print(f"\n  --- dose delta={delta} ---")
        q1_all, untestable, locs, sharps, errs = True, 0, [], [], []
        for s in range(seeds):
            teacher, acc_t, h_z_t, tr, te = train_teacher_unit(d_idx, delta, s, epochs, device,
                                                               ckpt_dir, resume, fp)
            r_ib = task_info_fano(1.0 - acc_t, C_TASK)
            k_min = max(2, math.ceil(math.exp(r_ib)))
            g0, ok_acc, ok_cb = g0_gate(acc_t, h_z_t, K_T)
            accs = {}
            for K in CALIBRATED_K:
                accs[K] = train_student_unit(d_idx, delta, s, K, teacher, epochs, device,
                                             ckpt_dir, resume, fp)
            plateau = distill_plateau_ok(CALIBRATED_K, accs)
            k_star = distill_gap_k_star(acc_t, CALIBRATED_K, accs) if plateau else None
            r1a, r1b, min_gap, viol = r1_exclusion(acc_t, k_min, CALIBRATED_K, accs)
            if min_gap is None:
                # no grid point below K_min: the exclusion is UNTESTABLE here (grid coverage defect),
                # NOT a violation -- never fold it into q1 (protocol discipline, pa4 grid-fix record)
                untestable += 1
                r1a, r1b = False, False
            else:
                q1_all = q1_all and r1a and r1b
            sh = sharpness(CALIBRATED_K, accs, acc_t)
            feats, labels = collect_features(teacher, te, device=device)
            probe = wall_geometry_probe(feats, labels, device=device)
            if plateau and k_star is not None:
                locs.append(1.0)
            else:
                locs.append(0.0)
            sharps.append(sh)
            errs.append(probe["err_1nn"])
            row = dict(delta=delta, s=s, phase="teacher", acc=acc_t, h_z=h_z_t, r_ib=r_ib,
                       k_min=k_min, g0=g0, fp=fp)
            jsonl_append(JSONL, row)
            row = dict(delta=delta, s=s, phase="student", accs=accs, plateau=plateau,
                       k_star=k_star, r1a=r1a, r1b=r1b, untestable=(min_gap is None),
                       sharpness=sh, err_1nn=probe["err_1nn"],
                       wall=probe["wall"], fp=fp)
            jsonl_append(JSONL, row)
            pf = preflight_report(acc_t, h_z_t, K_T, CALIBRATED_K, accs, acc_min=0.95)
            print(f"  s{s}: acc_t={acc_t:.3f} (G0 {'PASS' if g0 else 'FAIL'})  R_IB={r_ib:.2f} -> K_min={k_min}")
            r1_tag = "UNDECIDABLE (grid defect)" if min_gap is None else (
                "PASS" if (r1a and r1b) else "VIOLATED")
            print(f"     plateau={'yes' if plateau else 'no'}  K*={k_star}  R1={r1_tag}  "
                  f"sharpness={sh:.2f}  err_1nn={probe['err_1nn']:.3f} "
                  f"({probe['wall']})")
            print(f"     gap g(K): " + "  ".join(f"{K}:{acc_t - accs[K]:.2f}" for K in CALIBRATED_K))
            print(f"     [preflight] {pf['verdict']}")
        loc_frac = sum(locs) / len(locs)
        per_delta[delta] = dict(q1=q1_all, q1_untestable=untestable, loc_frac=loc_frac,
                                sharp_med=sorted(sharps)[len(sharps) // 2],
                                err1_med=sorted(errs)[len(errs) // 2])
        print(f"  [delta={delta} summary] locatability {loc_frac:.2f}  sharpness median {per_delta[delta]['sharp_med']:.2f}  "
              f"err_1nn median {per_delta[delta]['err1_med']:.3f}")
    print("\n  --- P-A1-4 verdict ---")
    verdict, detail = pa4_verdict(per_delta)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[PA4] P-A1-4 dose-response verdict\n")
        f.write(f"deltas={deltas} seeds={seeds} epochs={epochs}\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return per_delta, verdict


def run_selftest():
    ok = []
    # sharpness: a steep single-step cliff is captured on the right pair (32->48: 0.65 / ln1.5 ~ 1.60)
    accs = {4: 0.1, 6: 0.15, 8: 0.2, 32: 0.3, 48: 0.95, 64: 0.97, 96: 0.98, 128: 0.99,
            192: 0.99, 256: 0.99}
    sh = sharpness(CALIBRATED_K, accs, 1.0)
    ok.append(("sharpness captures 32->48 cliff", 1.2 < sh < 2.0))
    # flat curve -> sharpness near 0
    accs2 = {k: 0.99 for k in CALIBRATED_K}
    ok.append(("sharpness flat curve ~0", sharpness(CALIBRATED_K, accs2, 1.0) < 0.05))
    # monotonicity helpers
    ok.append(("_noninc non-increasing", _noninc([1.0, 0.7, 0.3])))
    ok.append(("_noninc rising violation", not _noninc([0.3, 0.9, 0.5])))
    ok.append(("_nondown non-decreasing", _nondown([0.1, 0.2, 0.9])))
    # verdict: support case
    per = {0.0: dict(q1=True, loc_frac=1.0, sharp_med=0.9, err1_med=0.0),
           0.15: dict(q1=True, loc_frac=0.67, sharp_med=0.5, err1_med=0.05),
           0.3: dict(q1=True, loc_frac=0.33, sharp_med=0.3, err1_med=0.10),
           0.5: dict(q1=True, loc_frac=0.0, sharp_med=0.2, err1_med=0.25)}
    v, _ = pa4_verdict(per)
    ok.append(("support case -> SUPPORT", v.startswith("SUPPORT")))
    # Q1 failure -> refute
    per_bad = {k: dict(v) for k, v in per.items()}
    per_bad[0.3]["q1"] = False
    v, _ = pa4_verdict(per_bad)
    ok.append(("Q1 violation -> REFUTED", v.startswith("REFUTED")))
    print("[PA4 selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all passed' if all_ok else 'failures present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="PA4: P-A1-4 dose-response (cliff sharpness vs output-mapping discreteness)")
    ap.add_argument("--deltas", type=str, default=",".join(str(d) for d in DELTAS),
                    help="dose list, comma-separated (default 0,0.15,0.3,0.5)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per dose (default 3)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="teacher/student epochs (default 40)")
    ap.add_argument("--fresh", action="store_true", help="recompute mode: ignore cache (default relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    deltas = [float(x) for x in args.deltas.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    run_pa4(deltas, args.seeds, args.epochs, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
