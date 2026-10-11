"""LOOSENESS RE-AUDIT: the corrected pre-registration for the falsified criterion, and the ground-truth split.

Background. The 38-scenario survey (`looseness_demo.py`, zero training, reading cached readouts) pre-registered
three criteria; **D2 -- "whenever the reference error is large (eps >= 0.20), looseness >= 3" -- was falsified**
and the round was recorded as MIXED with the falsification kept in place. The note also recorded corrected
epsilon bands, explicitly marked "for a future pre-registration, not changing this round's verdict". This script
is that future pre-registration, run on the same 38 scenarios, plus the two checks the first round was missing:

  (1) HOLDOUT. A relation fitted on the same points that suggested it proves nothing. D2' is therefore fitted on
      half the scenarios (a fixed, mechanical split: the scenarios are sorted by (source, name) and alternate)
      and scored on the other half, with the split rule fixed in advance.
  (2) GROUND TRUTH. On the lattice family the true I(X;Y) is computable (WALL-SATURATION step 0), so the
      reported looseness -- which divides by the WITNESS-RELATIVE requirement -- can be compared against the
      looseness computed against the BAYES requirement. That separates "the bound is loose" from "the witness is
      weak", which the first round could not do.

Pre-registration (2026-10-05, fixed before the numbers below were produced)
--------------------------------------------------------------------------------------------
D2' (replaces D2): looseness is an affine function of the reference error rate, looseness ~ a + b*eps with b > 0;
    the class count adds nothing beyond eps. (D2's threshold form is dropped: it was a step function fitted to a
    continuous relation and it failed on exactly the points that straddle its step.)
Criteria, all on the 38 recorded scenarios, zero training:
    C1  the fit on the FIT half has b > 0 and R^2 >= 0.5;
    C2  on the HELD-OUT half, the median absolute prediction error is <= 0.5 looseness units, and the sign of
        the relation is preserved (no held-out scenario with a larger eps predicted below one with a smaller eps
        by more than 0.5);
    C3  adding c*ln(n_classes) to the fit raises R^2 by <= 0.05 (eps is the whole story).
    Verdict: SUPPORT if C1 and C2; PARTIAL if exactly one; REFUTED if b <= 0.
Ground-truth extension (criteria fixed here too):
    G1  for every lattice instance with a located wall, report looseness_reported = ln(K_wall)/R_IB(witness) and
        looseness_true = ln(K_wall)/R_IB(Bayes), and the ratio;
    G2  the two agree within 10% on instances whose reference is at the Bayes error (the sanity direction);
        otherwise the reported looseness is inflated by the witness's weakness, and the size of that inflation
        is the number this extension exists to produce.
Iron rules: no training (the survey reads caches only), the falsification of D2 stays in the record, and D2' is
labelled post-hoc-fitted on this dataset until it is confirmed on a fresh one.
--------------------------------------------------------------------------------------------
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import looseness
from looseness_demo import loos_verdict

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "looseness.jsonl")
REPORT = os.path.join(_DATA, "looseness_reaudit_report.txt")


# ---------------------------------------------------------------------------
# pure functions (the criteria)
# ---------------------------------------------------------------------------

def ols(xs, ys):
    """Least squares y = a + b x; returns (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def mechanical_split(rows):
    """The split rule, fixed in advance: sort by (source, name) and alternate fit/test."""
    ordered = sorted(rows, key=lambda r: (r.get("source", ""), r.get("name", "")))
    return ordered[0::2], ordered[1::2]


def reaudit(rows):
    """D2' with the holdout and the class-count control; returns (verdict, detail lines, numbers)."""
    fit, test = mechanical_split(rows)
    a, b, r2 = ols([r["eps"] for r in fit], [r["looseness"] for r in fit])
    # class-count control: regress looseness on eps and ln C with the same fit half
    xs = [(r["eps"], math.log(r["n_classes"])) for r in fit]
    ys = [r["looseness"] for r in fit]
    n = len(xs)
    mx1 = sum(x[0] for x in xs) / n
    mx2 = sum(x[1] for x in xs) / n
    my = sum(ys) / n
    s11 = sum((x[0] - mx1) ** 2 for x in xs)
    s22 = sum((x[1] - mx2) ** 2 for x in xs)
    s12 = sum((x[0] - mx1) * (x[1] - mx2) for x in xs)
    s1y = sum((x[0] - mx1) * (y - my) for x, y in zip(xs, ys))
    s2y = sum((x[1] - mx2) * (y - my) for x, y in zip(xs, ys))
    det = s11 * s22 - s12 ** 2
    a2, b1, b2c, r2_c = 0.0, 0.0, 0.0, r2          # degenerate default: the extra regressor adds nothing
    if abs(det) > 1e-12 and s22 > 1e-12:
        b1 = (s22 * s1y - s12 * s2y) / det
        b2c = (s11 * s2y - s12 * s1y) / det
        a2 = my - b1 * mx1 - b2c * mx2
        ss_res = sum((y - (a2 + b1 * x[0] + b2c * x[1])) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2_c = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    errs = [abs(r["looseness"] - (a + b * r["eps"])) for r in test]
    med_err = sorted(errs)[len(errs) // 2] if errs else float("inf")
    # sign preservation on the held-out half
    inversions = 0
    for i in test:
        for j in test:
            if i["eps"] > j["eps"] + 0.05 and (a + b * i["eps"]) < (a + b * j["eps"]) - 0.5:
                inversions += 1
    c1 = b > 0 and r2 >= 0.5
    c2 = med_err <= 0.5 and inversions == 0
    c3 = (r2_c - r2) <= 0.05
    detail = [
        f"  fit half: n={len(fit)}, looseness ~ {a:.3f} + {b:.3f}*eps  (R^2={r2:.3f})",
        f"  C1 positive slope and R^2 >= 0.5 -> {'PASS' if c1 else 'FAIL'}",
        f"  held-out half: n={len(test)}, median |error| = {med_err:.3f} looseness units, "
        f"sign inversions = {inversions} -> C2 {'PASS' if c2 else 'FAIL'}",
        f"  C3 adding c*ln(n_classes) raises R^2 by {r2_c - r2:+.3f} (<= 0.05 required) "
        f"[c = {b2c:+.3f}] -> {'PASS' if c3 else 'FAIL'}",
    ]
    # precedence exactly as pre-registered: a non-positive slope is a refutation regardless of the holdout,
    # because "looseness does not even rise with the reference error" is the hypothesis being tested
    if b <= 0:
        v = "REFUTED: looseness is not an affine function of the reference error on this data"
    elif c1 and c2:
        v = "SUPPORT: D2' holds with a positive slope and generalises to the held-out half"
    elif c1 or c2:
        v = "PARTIAL: exactly one of the fit and the held-out generalisation holds"
    else:
        v = "REFUTED: neither the fit nor its generalisation holds"
    return v, detail, dict(a=a, b=b, r2=r2, r2_class=r2_c, med_err=med_err, inversions=inversions,
                           n_fit=len(fit), n_test=len(test), c3=c3)


def ground_truth_extension():
    """Reported looseness (witness-relative) vs looseness against the Bayes requirement."""
    truth = {}
    ct = os.path.join(_DATA, "cross_task.jsonl")
    if os.path.exists(ct):
        for line in open(ct, encoding="utf-8"):
            r = json.loads(line)
            if r.get("step0"):
                truth[(r["m"], round(r["sigma"], 3), round(r.get("spacing", 3.0), 2))] = r
    ws = os.path.join(_DATA, "wall_sat.jsonl")
    rows = []
    if os.path.exists(ws):
        for line in open(ws, encoding="utf-8"):
            r = json.loads(line)
            # arm "sigma" at the official configuration only: the m=3 two-epoch diagnostic also wrote
            # arm="sigma" rows, and those do not belong in a table compared against round 3. The rows of that
            # arm predate the `epochs` field, so m is the discriminator (the official arm is m=6 everywhere).
            if r.get("arm") == "sigma" and r.get("k_wall") and r.get("m") == 6:
                key = (r["m"], round(r["sigma"], 3), round(r.get("spacing", 3.0), 2))
                t = truth.get(key)
                if not t:
                    continue
                lo_rep = looseness(r["rib"], r["k_wall"])
                lo_true = looseness(t["rib_bayes"], r["k_wall"])
                rows.append(dict(c=r["c"], sigma=r["sigma"], k_wall=r["k_wall"], rib=r["rib"],
                                 rib_bayes=t["rib_bayes"], eps=r["eps"], eps_bayes=t["eps_bayes"],
                                 looseness_reported=lo_rep, looseness_true=lo_true,
                                 share_witness=1.0 - (lo_true / lo_rep if lo_rep else 0.0)))
    sc = os.path.join(_DATA, "scale_c_clean.jsonl")
    if os.path.exists(sc):
        # the jsonl also holds the smoke run's rows (a different fingerprint, same keys), so keep only the
        # LAST row per class count -- the official full run -- exactly as wall_saturation_demo.load_clean_rows does
        latest = {}
        for line in open(sc, encoding="utf-8"):
            r = json.loads(line)
            if r.get("k_wall"):
                latest[r["c"]] = r
        for r in latest.values():
            key = (r["m"], 1.0, 3.0)
            t = truth.get(key)
            if not t:
                continue
            lo_rep = looseness(r["rib"], r["k_wall"])
            lo_true = looseness(t["rib_bayes"], r["k_wall"])
            rows.append(dict(c=r["c"], sigma=1.0, k_wall=r["k_wall"], rib=r["rib"],
                             rib_bayes=t["rib_bayes"], eps=r["eps"], eps_bayes=t["eps_bayes"],
                             looseness_reported=lo_rep, looseness_true=lo_true,
                             share_witness=1.0 - (lo_true / lo_rep if lo_rep else 0.0)))
    # drop exact duplicates (the C=216, sigma=1.0 instance exists in both SCALE-C-3 and the sigma arm)
    seen, unique = set(), []
    for r in rows:
        key = (r["c"], round(r["sigma"], 3), r["k_wall"], round(r["rib"], 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def run_selftest():
    ok = []
    a, b, r2 = ols([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    ok.append((f"ols recovers an exact line (a={a:.2f}, b={b:.2f}, R^2={r2:.3f})",
               abs(a - 1.0) < 1e-9 and abs(b - 2.0) < 1e-9 and abs(r2 - 1.0) < 1e-9))
    rows = [dict(source="a", name=f"n{i}", eps=i / 10.0, looseness=1.0 + 10.0 * i / 10.0, n_classes=10)
            for i in range(10)]
    fit, test = mechanical_split(rows)
    ok.append((f"the split is mechanical and balanced ({len(fit)}/{len(test)})", len(fit) == 5 and len(test) == 5))
    v, _, nums = reaudit(rows)
    ok.append(("an exact relation passes C1/C2 and yields SUPPORT", v.startswith("SUPPORT") and nums["med_err"] < 1e-9))
    noisy = [dict(r, looseness=r["looseness"] + (3.0 if i % 2 else -3.0)) for i, r in enumerate(rows)]
    v2, _, _ = reaudit(noisy)
    ok.append(("a relation that does not generalise is not called SUPPORT", not v2.startswith("SUPPORT")))
    flat = [dict(r, looseness=2.0) for r in rows]
    v3, _, nums3 = reaudit(flat)
    ok.append(("a constant looseness is REFUTED (the pre-registered precedence: b <= 0 refutes)",
               v3.startswith("REFUTED") and nums3["b"] == 0.0))
    partial = [dict(r, looseness=1.0 + 10.0 * r["eps"] + (2.0 if i % 3 == 0 else 0.0))
               for i, r in enumerate(rows)]
    v4, _, _ = reaudit(partial)
    ok.append(("a relation that fits but does not generalise is PARTIAL, not SUPPORT",
               v4.startswith("PARTIAL")))
    print("[LOOSENESS RE-AUDIT selftest] the corrected criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="LOOSENESS RE-AUDIT: corrected pre-registration + ground truth")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    print("=" * 100)
    print(f"[LOOSENESS RE-AUDIT] {len(rows)} recorded scenarios, zero training; D2' is post-hoc-fitted here")
    print("=" * 100)
    v0, d0 = loos_verdict(rows)
    print("  the original round, re-run unchanged (for the record):")
    for line in d0:
        print("   " + line.strip())
    print(f"   [original verdict] {v0}")
    print()
    v, detail, nums = reaudit(rows)
    for line in detail:
        print(line)
    print(f"  [D2' verdict] {v}")
    print()
    gt = ground_truth_extension()
    print("-" * 100)
    print("  G1/G2: reported looseness (witness-relative) vs looseness against the Bayes requirement")
    if not gt:
        print("   no lattice instances with both a located wall and a computed ground truth")
    else:
        print(f"   {'C':>5} {'sigma':>6} {'K_wall':>7} {'eps':>6} {'eps_bayes':>10} {'R_IB':>7} "
              f"{'R_IB_bayes':>10} {'lo_reported':>12} {'lo_true':>8} {'witness share':>14}")
        for r in sorted(gt, key=lambda x: (x["sigma"], x["c"])):
            print(f"   {r['c']:>5} {r['sigma']:>6.2f} {r['k_wall']:>7} {r['eps']:>6.3f} {r['eps_bayes']:>10.3f} "
                  f"{r['rib']:>7.3f} {r['rib_bayes']:>10.3f} {r['looseness_reported']:>12.2f} "
                  f"{r['looseness_true']:>8.2f} {r['share_witness']:>13.0%}")
        shares = [r["share_witness"] for r in gt]
        print(f"\n   median share of the reported looseness attributable to the witness's weakness: "
              f"{sorted(shares)[len(shares) // 2]:.0%} (range {min(shares):.0%}..{max(shares):.0%})")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(f"original verdict (unchanged): {v0}\n")
        f.write(f"D2' verdict: {v}\n")
        for line in detail:
            f.write(line + "\n")
        for r in gt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(f"numbers: {json.dumps(nums)}\n")
    print(f"\n  [report] {REPORT}")


if __name__ == "__main__":
    main()
