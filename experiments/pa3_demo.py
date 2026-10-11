"""PA3: A-003's open predictions -- P-A3-2 (dual-caliber discreteness, zero training) and
P-A3-1 (wall-probe prospective validation, MNIST positive arm).

Usage:
    python experiments/pa3_demo.py --p2            # P-A3-2: zero-training, reads existing checkpoints
    python experiments/pa3_demo.py --p1            # P-A3-1: MNIST prospective validation (trains, ~5 min GPU)
    python experiments/pa3_demo.py --selftest      # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) P-A3-2 (dual-caliber discreteness; the probe measures the REPRESENTATION caliber, the proposition
    is about the OUTPUT mapping): compute, per unit, both err_1nn on the pre-VQ features AND err_1nn on
    the logits. Pool = CP-H units (3 teachers + 39 students, CIFAR-10) + PA4 teachers (12, grid64 four
    doses). Criteria: (a) Spearman rho(feat_err, logit_err) >= 0.8 over the pool; (b) wall classification
    agreement at the calibrated threshold 0.02 = 100% (both calibers put every unit on the same side).
(2) P-A3-1 (prospective validation, POSITIVE arm only -- MNIST was never used to calibrate the probe;
    the diffuse arm needs new data, which is outside this workspace):
    Procedure: train MNIST teachers (Net d=16 K=256, 3ep, 3 seeds -- the v2 protocol budget), run
    wall_geometry_probe on each teacher -> the probe's PREDICTION; then run the distillation grid
    (Fano-convention K_min, grid anchored at seed-0 K_min, the v2 full-style grid) and measure the
    ACTUAL "meaningful cliff":
        meaningful K* := plateau_ok AND k_star is not None AND k_star > k_list[0]
        (the grid-bottom guard: on wall-less tasks everything saturates and K* locates trivially at the
        bottom -- that is NOT a cliff and must not count as one).
    Comparison (pre-registered): prediction correct iff
        (probe=discrete AND >=2/3 seeds have a meaningful K*) OR (probe=diffuse AND <2/3 have one).
(3) Iron rules: print only GBK-encodable characters; criteria before data; pure functions unit-testable;
    no downloads (MNIST is in experiments/data/MNIST/raw).
"""

import argparse
import hashlib
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import train_one_epoch, evaluate, task_info_fano, err_1nn, wall_geometry_probe
from capacity_lens.preflight import distill_plateau_ok, distill_gap_k_star
from cph_demo import collect_unit, load_cifar, run_signature as cph_run_signature, spearman
from cifar_demo import SmallCNN
from cp3_demo import make_data, build_model
from mnist_demo import get_mnist, Net

K_TEACHER = 256
CIFAR_FP = None   # computed lazily (cph fingerprint of the locked config)
PA4_FP = None


def _cifar_fp():
    global CIFAR_FP
    if CIFAR_FP is None:
        CIFAR_FP = cph_run_signature(False, False, 120, None, 96, K_TEACHER, 1e-3, 4, True, "cuda")
    return CIFAR_FP


def _pa4_fp():
    global PA4_FP
    if PA4_FP is None:
        raw = "|".join(["d0.0/0.15/0.3/0.5", "ep40", "lr0.001", "kt256", "wk0", "devcuda"])
        PA4_FP = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return PA4_FP


def _both_err1nn(feats, logits, labels, device="cpu"):
    return err_1nn(feats, labels, device=device), err_1nn(logits, labels, device=device)


def meaningful_k_star(k_list, accs, acc_t):
    """The P-A3-1 "meaningful cliff" guard (pure function): plateau ok, K* located, and K* is NOT the
    grid bottom (a bottom K* on a wall-less task is trivial saturation, not a cliff)."""
    ks = sorted(k_list)
    plateau = distill_plateau_ok(ks, accs)
    k_star = distill_gap_k_star(acc_t, ks, accs) if plateau else None
    return bool(plateau and k_star is not None and k_star > ks[0]), plateau, k_star


def pa31_verdict(pred_wall, meaningful_flags):
    """P-A3-1 comparison (pure function). pred_wall: the probe's per-seed wall; meaningful_flags:
    per-seed bool. Correct iff (discrete AND >=2/3 meaningful) OR (diffuse AND <2/3 meaningful)."""
    import collections
    n = len(meaningful_flags)
    need = math.ceil(2 / 3 * n)
    frac = sum(meaningful_flags)
    walls = collections.Counter(pred_wall)
    if len(walls) == 1:
        wall = list(walls)[0]
    else:
        wall = "mixed"
    detail = [f"  probe predicted: {dict(walls)}; measured meaningful K*: {frac}/{n} seeds (threshold {need})"]
    if wall == "discrete" and frac >= need:
        return "SUPPORT: probe predicted a discrete wall and a meaningful cliff was measured -- prospective positive arm PASSES", detail
    if wall == "diffuse" and frac < need:
        return "SUPPORT: probe predicted a diffuse wall and no meaningful cliff was measured -- prospective positive arm PASSES", detail
    if wall == "mixed":
        return "UNDECIDABLE: the three seeds disagree (recorded as-is)", detail
    return "REFUTED: probe prediction contradicts measurement (recorded as-is)", detail


# ---------------------------------------------------------------------------
# P-A3-2: dual-caliber (zero training)
# ---------------------------------------------------------------------------

def run_p2(device="cpu"):
    print("=" * 76)
    print("[PA3 P-A3-2] dual-caliber discreteness check (zero training: CP-H checkpoints + PA4 checkpoints)")
    print("=" * 76)
    feats_errs, logit_errs, notes = [], [], []
    fp = _cifar_fp()
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cph", fp)
    _, test_loader = load_cifar(subset=None, workers=0)
    width, k_t = 96, K_TEACHER
    for s in range(3):
        tpath = os.path.join(ckpt_dir, f"cifar_teacher_s{s}.pt")
        if not os.path.exists(tpath):
            continue
        ck = torch.load(tpath, map_location="cpu", weights_only=True)
        model = SmallCNN(d=64, n_codes=k_t, width=width).to(device)
        model.load_state_dict(ck["state_dict"])
        t, logits, _, y = collect_unit(model, test_loader, device=device)
        fe, le = _both_err1nn(t, logits, y, device=device)
        feats_errs.append(fe)
        logit_errs.append(le)
        notes.append(f"cph_teacher_s{s}")
        print(f"  cph teacher s{s}: feat_err1nn={fe:.3f}  logit_err1nn={le:.3f}")
    for s in range(3):
        for K in [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 192, 256]:
            spath = os.path.join(ckpt_dir, f"cifar_student_s{s}_K{K}.pt")
            if not os.path.exists(spath):
                continue
            ck = torch.load(spath, map_location="cpu", weights_only=True)
            model = SmallCNN(d=64, n_codes=K, width=width).to(device)
            model.load_state_dict(ck["state_dict"])
            t, logits, _, y = collect_unit(model, test_loader, device=device)
            fe, le = _both_err1nn(t, logits, y, device=device)
            feats_errs.append(fe)
            logit_errs.append(le)
            notes.append(f"cph_s{s}_K{K}")
    # PA4 teachers (grid64, four doses). NOTE: pa4's task is make_grid64_overlap (delta-blended); importing
    # pa4_demo hijacks sys.stdout via its Tee, so guard-and-restore around the import.
    import sys as _sys
    _so, _se = _sys.stdout, _sys.stderr
    from pa4_demo import make_grid64_overlap  # noqa: E402  (module-level Tee side effect)
    _sys.stdout, _sys.stderr = _so, _se
    fp4 = _pa4_fp()
    ckpt4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pa4", fp4)
    for d_idx, delta in enumerate([0.0, 0.15, 0.3, 0.5]):
        for s in range(3):
            tpath = os.path.join(ckpt4, f"d{d_idx}_teacher_s{s}.pt")
            if not os.path.exists(tpath):
                continue
            ck = torch.load(tpath, map_location="cpu", weights_only=True)
            model = build_model("cnn", K_TEACHER, "euclid").to(device)
            model.load_state_dict(ck["state_dict"])
            (xtr, ytr), (xte, yte) = make_grid64_overlap(150, s, delta)
            from torch.utils.data import DataLoader
            te = DataLoader(list(zip(xte, yte)), batch_size=256)
            t, logits, _, y = collect_unit(model, te, device=device)
            fe, le = _both_err1nn(t, logits, y, device=device)
            feats_errs.append(fe)
            logit_errs.append(le)
            notes.append(f"pa4_d{delta}_s{s}")
            print(f"  pa4 delta={delta} s{s}: feat_err1nn={fe:.3f}  logit_err1nn={le:.3f}")
    n = len(feats_errs)
    if n < 8:
        print(f"  [UNDECIDABLE] valid units {n} < 8")
        return
    rho = spearman(feats_errs, logit_errs)
    agree = sum(1 for fe, le in zip(feats_errs, logit_errs)
                if (fe <= 0.02) == (le <= 0.02))
    print(f"\n  valid units: {n}")
    print(f"  (a) rho(feat_err, logit_err) = {rho:+.3f} (pre-registered threshold >= 0.8) -> {'PASS' if rho >= 0.8 else 'FAIL'}")
    print(f"  (b) threshold 0.02 side agreement = {agree}/{n} (pre-registered 100%) -> {'PASS' if agree == n else 'FAIL'}")
    if rho >= 0.8 and agree == n:
        print("  [P-A3-2 verdict] SUPPORT: output-caliber and representation-caliber discreteness align; the probe proxy is valid.")
    else:
        print("  [P-A3-2 verdict] PARTIAL/REFUTED: calibers do not fully align (recorded as-is).")
    print(f"  [note] first 10 unit labels: {notes[:10]}")


# ---------------------------------------------------------------------------
# P-A3-1: MNIST prospective validation (positive arm)
# ---------------------------------------------------------------------------

def run_p1(device="cpu"):
    print("=" * 76)
    print("[PA3 P-A3-1] wall-geometry probe prospective validation (MNIST positive arm; 3 seeds, v2 budget 3ep)")
    print("=" * 76)
    print("  [pre-registered] probe first for the prediction, then run the grid and compare with the measurement; 'meaningful K*' needs plateau to hold and K* above the grid bottom")
    print("         (without a wall, K* lands trivially at the grid bottom -- not a cliff).")
    train_loader, test_loader = get_mnist(subset=None)
    pred_walls, flags = [], []
    k_list = None
    for s in range(3):
        torch.manual_seed(1000 + s)
        teacher = Net(d=16, n_codes=K_TEACHER).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for _ in range(3):
            train_one_epoch(teacher, train_loader, opt, device=device)
        acc_t, _ = evaluate(teacher, test_loader, device=device)
        # probe on the teacher's pre-VQ features
        feats, ys = [], []
        teacher.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                # Net.forward flattens before fc1 (x.view(B, -1)); replicate it for the pre-VQ features
                feats.append(F.relu(teacher.fc1(x.view(x.size(0), -1))).cpu())
                ys.append(y)
        feats = torch.cat(feats)
        labels = torch.cat(ys)
        probe = wall_geometry_probe(feats, labels, device=device)
        pred_walls.append(probe["wall"])
        r_ib = task_info_fano(1.0 - acc_t, 10)
        k_min = max(2, math.ceil(math.exp(r_ib)))
        if k_list is None:
            k_list = sorted({32, max(2, k_min - 2), k_min, min(k_min + 2, K_TEACHER),
                             min(192, K_TEACHER), K_TEACHER})
            print(f"  [K grid] anchored at seed0 K_min={k_min}: {k_list}")
        accs = {}
        for K in k_list:
            torch.manual_seed(2000 + 100 * s + K)
            student = Net(d=16, n_codes=K).to(device)
            opt = torch.optim.Adam(student.parameters(), lr=1e-3)
            for _ in range(3):
                train_one_epoch(student, train_loader, opt, device=device, distill_from=teacher)
            acc_s, _ = evaluate(student, test_loader, device=device)
            accs[K] = acc_s
        meaningful, plateau, k_star = meaningful_k_star(k_list, accs, acc_t)
        flags.append(meaningful)
        print(f"  s{s}: teacher acc={acc_t:.3f}  err_1nn={probe['err_1nn']:.3f} -> predicted[{probe['wall']}]")
        print(f"      plateau={plateau}  K*={k_star}  meaningful={'yes' if meaningful else 'no'}"
              f"  gap g(K): " + "  ".join(f"{K}:{acc_t - accs[K]:.2f}" for K in k_list))
    verdict, detail = pa31_verdict(pred_walls, flags)
    print("\n  --- P-A3-1 verdict ---")
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")


def run_selftest():
    ok = []
    # meaningful_k_star: bottom K* is not meaningful
    accs = {7: 0.96, 9: 0.97, 11: 0.97, 32: 0.97, 192: 0.97, 256: 0.97}
    m, _, ks = meaningful_k_star([7, 9, 11, 32, 192, 256], accs, 0.97)
    ok.append(("bottom K* is not meaningful", not m and ks == 7))
    # a mid-grid K* with plateau is meaningful (g(192)=0.01 <= g_ref+0.02=0.025 -> K*=192)
    accs2 = {7: 0.2, 9: 0.4, 11: 0.6, 32: 0.9, 192: 0.96, 256: 0.965}
    m2, _, ks2 = meaningful_k_star([7, 9, 11, 32, 192, 256], accs2, 0.97)
    ok.append(("mid-grid K* is meaningful", m2 and ks2 == 192))
    # pa31_verdict: discrete + 3/3 -> support
    v, _ = pa31_verdict(["discrete"] * 3, [True, True, True])
    ok.append(("discrete+3/3 -> SUPPORT", v.startswith("SUPPORT")))
    v, _ = pa31_verdict(["diffuse"] * 3, [False, False, True])
    ok.append(("diffuse+1/3 -> SUPPORT", v.startswith("SUPPORT")))
    v, _ = pa31_verdict(["diffuse"] * 3, [True, True, False])
    ok.append(("diffuse+2/3 -> REFUTED", v.startswith("REFUTED")))
    print("[PA3 selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all passed' if all_ok else 'some failed (code bug)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="PA3: A-003 open predictions (P-A3-2 dual-caliber / P-A3-1 MNIST prospective)")
    ap.add_argument("--p2", action="store_true", help="P-A3-2: dual-caliber check (zero training)")
    ap.add_argument("--p1", action="store_true", help="P-A3-1: MNIST prospective validation (trains ~5 min GPU)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.p2:
        run_p2(device=device)
    elif args.p1:
        run_p1(device=device)
    else:
        print("specify --p2 or --p1 (or --selftest).")
    print("\nDone.")


if __name__ == "__main__":
    main()
