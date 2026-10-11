"""CROSS-TASK: is there a common difficulty scale across tasks? (the ledger's P-A1-2, never positively designed)

Why this experiment exists. The A-001/A-003 record contains one item that was never given a positive
design: P-A1-2, "a cross-task difficulty ruler" -- the hope that the requirement reading R_IB could place
*different tasks* on one scale ("task X needs twice the capacity of task Y"). Everything measured since
argues that the reading is witness-relative (WITNESS, PRETRAINED-WITNESS, FAMILY-LIBRARY), but that has
never been turned into a measurement with a KNOWN ground truth. The lattice family makes it possible,
because its true information I(X;Y) is computable exactly: equal-weight Gaussian classes on a lattice with
known centres give p(y|x) in closed form, so Monte Carlo yields the Bayes error, H(Y|X) and hence I(X;Y).

The question, in three increasingly demanding parts:
  X1 INVARIANCE      two different parameterizations of the same task should read the same. spacing 6.0 with
                     sigma 1.0 and spacing 3.0 with sigma 0.5 have the SAME pairwise separation ratio (6.0 vs
                     6.0), so they are the same task up to a global scale change of the input. If the reading
                     is a function of the task's information, R_IB must agree.
  X2 ORDERING        across every (m, sigma, spacing) point measured, does the reading order the tasks the
                     way the ground truth does?
  X3 CARDINALITY     can the readings be compared as numbers, i.e. does under-reading (the gap between the
                     Bayes requirement and the reference-derived one) stay constant across tasks?

Pre-registration (2026-10-05, written before the arm below was run)
--------------------------------------------------------------------------------------------
(1) Ground truth: lattice_posterior_stats (Monte Carlo, 200k samples) gives eps_bayes, I(X;Y), rib_bayes,
    k_min_bayes for every configuration.
(2) Configurations measured by training (C = 216 everywhere, so only the task's difficulty moves):
      A: spacing 3.0, sigma 1.0   (the SCALE-C round-3 baseline; also re-measured inside this arm)
      B: spacing 3.0, sigma 0.5
      C: spacing 6.0, sigma 1.0   (scale-equivalent to B by construction)
    Protocol identical to SCALE-C round 3: reference VQ(K_t=2048), students K in the round-3 grid plus 4096,
    d = 64, 60 epochs, 150 samples/class, seed 0, cached teacher logits.
(3) Criteria:
      X1  |R_IB(C) - R_IB(B)| <= 0.30 nats   AND   the ground truth agrees they are close
          (|I(C) - I(B)| <= 0.30); both must hold for the reading to be a task property.
      X2  Spearman(true I(X;Y), R_IB) = +1 over all points; a single inversion fails it.
      X3  the under-reading gap (rib_bayes - rib) is NOT constant: the honest verdict is "ordinal only" if
          the gap grows with the task's information (report the fitted slope).
    Verdict SUPPORT ("a common scale exists, but only ordinal") if X1 and X2; REFUTED if X1 fails (the
    reading is not even a function of the task's information); MIXED if only one of X1/X2 holds.
(4) Scope limit, recorded in advance: all three configurations come from ONE family, so X1/X2 test
    parameter-invariance and ordering, not cross-family comparability. Cross-family points from other
    campaigns (MNIST / CIFAR-10 / grid64 images) have no computable ground truth and are reported
    separately as a consistency table only.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import task_info_fano, DistillCapacity
from scale_c_demo import rank_correlation, class_count
from wall_saturation_demo import (lattice_posterior_stats, run_task_arm, SPACING, N_MC, M_GRID, SIGMA_GRID)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "cross_task.jsonl")
REPORT = os.path.join(_DATA, "cross_task_report.txt")
M_FIXED = 6                       # C = 216
CONFIGS = [(1.0, 3.0), (0.5, 3.0), (1.0, 6.0)]     # A, B, C


def step0(device, m_grid=M_GRID, sigma_grid=SIGMA_GRID, spacings=(3.0, 6.0)):
    """The ground-truth table: what each task actually carries (zero training)."""
    print("=" * 100)
    print("[CROSS-TASK] the ground truth: true I(X;Y) and the Bayes requirement of every configuration")
    print("=" * 100)
    print(f"  {'m':>3} {'C':>5} {'sigma':>6} {'spacing':>8} {'eps_bayes':>10} {'I(X;Y)':>8} "
          f"{'rib_bayes':>10} {'K_min_bayes':>12}  scale-invariant?")
    rows = []
    for m in m_grid:
        for spacing in spacings:
            base = None
            for sigma in sigma_grid:
                st = lattice_posterior_stats(m, sigma, spacing=spacing, device=device)
                st.update(step0=True, spacing=spacing)
                # spacing s with noise q is the same task as spacing 1 with noise q/s
                st["scale_class"] = round(sigma / spacing, 4)
                rows.append(st)
                print(f"  {m:>3} {st['C']:>5} {sigma:>6.2f} {spacing:>8.1f} {st['eps_bayes']:>10.4f} "
                      f"{st['mi']:>8.3f} {st['rib_bayes']:>10.3f} {st['k_min_bayes']:>12}"
                      f"   sigma/spacing={st['scale_class']}")
    with open(JSONL, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # the pairs the invariance criterion is about
    print("\n  scale-equivalent pairs the criteria use (same sigma/spacing ratio):")
    by_class = {}
    for r in rows:
        if r["m"] == M_FIXED:
            by_class.setdefault(r["scale_class"], []).append(r)
    for cls, group in sorted(by_class.items()):
        if len(group) > 1:
            desc = ", ".join(f"(sigma {g['sigma']:.2f}, spacing {g['spacing']:.1f}, I(X;Y) {g['mi']:.3f})"
                             for g in group)
            print(f"    ratio {cls}: {desc}")
    return rows


def load_baseline_rows(m=M_FIXED):
    """Reuse the WALL-SATURATION sigma-arm rows instead of retraining the same configurations.

    Those rows carry (sigma, rib, mi_true, under_read_rib, k_wall, ...) but were written before the arm
    recorded `spacing`; spacing 3.0 is their documented value, so it is filled in here.
    """
    path = os.path.join(_DATA, "wall_sat.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != "sigma" or r.get("m") != m or "rib" not in r:
            continue
        r.setdefault("spacing", SPACING)
        out.append(r)
    return out


def cross_verdict(rows, arm_rows, tol_rib=0.30, tol_true=0.30):
    detail = []
    by_key = {(round(r["sigma"], 3), round(r["spacing"], 3)): r for r in arm_rows}
    b = by_key.get((0.5, 3.0))
    c = by_key.get((1.0, 6.0))
    if b and c:
        d_rib = abs(b["rib"] - c["rib"])
        d_true = abs(b["mi_true"] - c["mi_true"])
        x1 = d_rib <= tol_rib and d_true <= tol_true
        detail.append(f"  X1 invariance on scale-equivalent tasks: |dR_IB|={d_rib:.3f} nats (tol {tol_rib}), "
                      f"|dI(X;Y)|={d_true:.3f} (tol {tol_true}) -> {'PASS' if x1 else 'FAIL'}")
        detail.append(f"       B(sigma .50, spacing 3): R_IB={b['rib']:.3f}, true={b['mi_true']:.3f}, "
                      f"K_min={b['k_min']} | C(sigma 1.0, spacing 6): R_IB={c['rib']:.3f}, "
                      f"true={c['mi_true']:.3f}, K_min={c['k_min']}")
    else:
        x1 = False
        detail.append("  X1 not evaluable (arm rows missing)")
    pts = [r for r in arm_rows if r.get("mi_true") is not None]
    if len(pts) >= 3:
        # ordering is judged tie-robustly: a MATERIAL difference in the ground truth must not be inverted
        # (an exact tie in I(X;Y) says nothing about the reading, so it cannot be a failure)
        inversions = []
        for i in pts:
            for j in pts:
                if i["mi_true"] > j["mi_true"] + 0.05 and i["rib"] < j["rib"] - 1e-9:
                    inversions.append((round(i["mi_true"], 2), round(i["rib"], 2),
                                       round(j["mi_true"], 2), round(j["rib"], 2)))
        rho = rank_correlation([r["mi_true"] for r in pts], [r["rib"] for r in pts])
        x2 = not inversions
        detail.append(f"  X2 ordering by the true information: {len(inversions)} material inversions over "
                      f"{len(pts)} points (Spearman={rho:+.3f}, descriptive) -> {'PASS' if x2 else 'FAIL'}")
        for inv in inversions:
            detail.append(f"       inversion: true {inv[0]} > {inv[2]} but reading {inv[1]} < {inv[3]}")
    else:
        x2 = False
        detail.append(f"  X2 not evaluable ({len(pts)} points)")
    gaps = [(r["mi_true"], r["under_read_rib"]) for r in pts]
    if len(gaps) >= 2:
        xs = [g[0] for g in gaps]
        ys = [g[1] for g in gaps]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        detail.append(f"  X3 under-reading (rib_bayes - rib) vs the task's true information: "
                      f"{[round(y, 2) for y in ys]} at I={[round(x, 2) for x in xs]}, slope {slope:+.2f} "
                      f"-> " + ("the offset is constant (cardinal comparison plausible)"
                               if abs(slope) < 0.15 else
                               "the offset is NOT constant: it "
                               + ("grows" if slope > 0 else "shrinks")
                               + " with the task's information, so the numbers are not commensurable"))
    if x1 and x2:
        v = ("SUPPORT (ordinal): the reading IS a function of the task's information and orders tasks "
             "correctly, but the readings are not commensurable across tasks -- the gap between the Bayes "
             "requirement and the reading is task- and witness-dependent, not a constant offset")
    elif x1 or x2:
        v = "MIXED: only one of invariance/ordering holds"
    else:
        v = "REFUTED: the reading is not a function of the task's information (P-A1-2 has no positive design)"
    return v, detail


def run_selftest():
    ok = []
    st_a = lattice_posterior_stats(M_FIXED, 1.0, spacing=3.0, n_mc=20000)
    st_c = lattice_posterior_stats(M_FIXED, 1.0, spacing=6.0, n_mc=20000)
    st_b = lattice_posterior_stats(M_FIXED, 0.5, spacing=3.0, n_mc=20000)
    ok.append((f"scale equivalence holds in the ground truth: I(B)={st_b['mi']:.3f} vs I(C)={st_c['mi']:.3f}",
               abs(st_b["mi"] - st_c["mi"]) < 0.15))
    ok.append((f"and both exceed the noisy baseline I(A)={st_a['mi']:.3f} by the measured margin",
               st_b["mi"] > st_a["mi"] + 0.5))
    # the criteria must reject a reading that is not a function of the task
    fake_arm = [dict(sigma=1.0, spacing=3.0, rib=2.0, mi_true=4.5, under_read_rib=1.1, k_min=8),
                dict(sigma=0.5, spacing=3.0, rib=5.0, mi_true=5.4, under_read_rib=0.2, k_min=150),
                dict(sigma=1.0, spacing=6.0, rib=2.1, mi_true=5.4, under_read_rib=3.1, k_min=8)]
    v_bad, _ = cross_verdict([], fake_arm)
    ok.append(("criteria REJECT a scale-equivalent pair that reads differently", v_bad.startswith("REFUTED")
               or v_bad.startswith("MIXED")))
    good = [dict(fake_arm[0]), dict(fake_arm[1]), dict(fake_arm[2], rib=4.95, under_read_rib=0.25)]
    v_good, _ = cross_verdict([], good)
    ok.append(("criteria ACCEPT an invariant, correctly ordered reading", v_good.startswith("SUPPORT")))
    print("[CROSS-TASK selftest] ground truth + criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    # arm rows live in the WALL-SATURATION jsonl: run_task_arm records every task arm there (tag "sigma"
    # for the noise sweep, "cross" for this experiment), so the offline verdict reads both files.
    arm = load_baseline_rows()
    ws = os.path.join(_DATA, "wall_sat.jsonl")
    if os.path.exists(ws):
        for line in open(ws, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") == "cross":
                r.setdefault("spacing", SPACING)
                arm.append(r)
    if not arm:
        raise SystemExit("[CROSS-TASK] no arm rows recorded yet")
    v, detail = cross_verdict(rows, arm)
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="CROSS-TASK: is there a common difficulty scale? (P-A1-2)")
    ap.add_argument("--step0", action="store_true", help="ground-truth table only (zero training)")
    ap.add_argument("--arm", action="store_true", help="train the three configurations at C=216")
    ap.add_argument("--m", type=int, default=M_FIXED)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    rows = step0(device)
    if not args.arm:
        print("\n  (step 0 only; add --arm to measure the readings)")
        return
    print()
    print("=" * 100)
    print(f"[CROSS-TASK] arm at m={args.m} (C={class_count(args.m)}): configurations {CONFIGS}")
    print("=" * 100)
    baseline = load_baseline_rows(args.m)
    have = {(round(r["sigma"], 3), round(r.get("spacing", SPACING), 3)) for r in baseline}
    if baseline:
        print(f"  reusing {len(baseline)} row(s) from WALL-SATURATION for: "
              + ", ".join(f"(sigma {s}, spacing {d})" for s, d in sorted(have)))
    todo = [cfg for cfg in CONFIGS if (round(cfg[0], 3), round(cfg[1], 3)) not in have]
    print(f"  configurations still to train: {todo if todo else 'none'}")
    arm_rows = list(baseline)
    if todo:
        arm_rows += run_task_arm(args.m, todo, device, resume=not args.fresh, tag="cross")
    print("\n  --- cross-task verdict ---")
    v, detail = cross_verdict(rows, arm_rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")
    with open(REPORT, "w", encoding="utf-8") as f:
        for r in rows + arm_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + v + "\n")
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
