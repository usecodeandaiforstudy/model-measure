"""CP-H post: zero-training post-analysis of the axis-2 gray zone (reads cph_readings_*.jsonl only).

**Positioning (must be read first)**: this is a POST-HOC, DESCRIPTIVE analysis -- it computes no verdict
and changes no pre-registered criterion (the pre-registered S1/S2/S3 verdicts live in cph_demo.py and
stand as printed). Its purpose is to squeeze the gray zone one step further with three additional views:
  1. cluster decomposition of the knee window (bottom cluster K<=8 vs top cluster K>=64): the bottom
     cluster is where acc varies a lot (0.10-0.36 across seeds) while log K varies little (0.69-2.08) --
     a reading that still tracks acc there carries quality information that capacity alone cannot order;
  2. partial Spearman rho(reading, acc | logK) on the 14 knee points (rank residuals): the direct
     "beyond-capacity" correlation. n=14 -> DIRECTIONAL ONLY, wide CI;
  3. full-grid context correlations over all student units.
Everything below is labeled descriptive/post-hoc, never a verdict.

Usage:
    python experiments/cph_post.py                        # newest cph_readings_*.jsonl
    python experiments/cph_post.py --jsonl <path>         # explicit file
"""

import argparse
import glob
import json
import math
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
READINGS = ["i_plug", "silhouette", "separation", "d_prime"]


def _ranks(vals):
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
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 0 else 0.0


def partial_spearman(xs, ys, zs):
    """Spearman rho(x, y) with z partialed out (rank residuals, least squares). Directional for small n."""
    n = len(xs)
    if n < 4:
        return 0.0
    rx, ry, rz = _ranks(list(xs)), _ranks(list(ys)), _ranks(list(zs))

    def resid(r):
        m, mz = sum(r) / n, sum(rz) / n
        v = sum((z - mz) ** 2 for z in rz)
        b = sum((z - mz) * (a - m) for z, a in zip(rz, r)) / v if v > 0 else 0.0
        a0 = m - b * mz
        return [a - (a0 + b * z) for z, a in zip(rz, r)]

    ex, ey = resid(rx), resid(ry)
    mx, my = sum(ex) / n, sum(ey) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(ex, ey))
    vx = sum((a - mx) ** 2 for a in ex)
    vy = sum((b - my) ** 2 for b in ey)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 0 else 0.0


def load_units(jsonl_path):
    """Dedupe (kind, seed, K) keeping the LAST row (the --read re-run wrote corrected in_knee flags)."""
    last = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            last[(r["kind"], r["seed"], r["K"])] = r
    return list(last.values())


def col(units, key):
    return [u[key] for u in units]


def corr_line(label, xs, ys, zs=None):
    if zs is None:
        return f"  {label:>14}: rho(acc)={spearman(xs, ys):+.2f}"
    return (f"  {label:>14}: rho(acc)={spearman(xs, ys):+.2f}  "
            f"rho(logK)={spearman(xs, zs):+.2f}  "
            f"**partial rho(acc|logK)={partial_spearman(xs, ys, zs):+.2f}**")


def run(jsonl_path):
    units = load_units(jsonl_path)
    students = [u for u in units if u["kind"] == "student"]
    teachers = [u for u in units if u["kind"] == "teacher"]
    print("=" * 76)
    print(f"[CP-H post] axis-2 gray zone zero-training post-analysis (descriptive, no verdict)")
    print("=" * 76)
    print(f"  data: {jsonl_path}  ({len(teachers)} teachers / {len(students)} students, deduped to last row)")
    if not students:
        print("  [error] no student readings.")
        return
    knee = [u for u in students if u.get("in_knee")]
    bottom = [u for u in knee if u["K"] <= 8]
    top = [u for u in knee if u["K"] >= 64]
    print(f"\n  knee window {len(knee)} pts = bottom cluster (K<=8) {len(bottom)} pts + top cluster (K>=64) {len(top)} pts")

    acc_k, logk_k = col(knee, "acc"), col(knee, "log_k")
    print(f"\n  [1] full-grid context ({len(students)} student units)")
    acc_a, logk_a = col(students, "acc"), col(students, "log_k")
    print(f"  rho(logK, acc) = {spearman(logk_a, acc_a):+.2f}  "
          f"rho(H(Z), acc) = {spearman(col(students, 'h_z'), acc_a):+.2f}")
    print("\n  [2] knee window (recheck against the pre-registered definition)")
    print(corr_line("logK itself", logk_k, acc_k))
    print(corr_line("H(Z) baseline", col(knee, "h_z"), acc_k, logk_k))
    for r in READINGS:
        print(corr_line(r, col(knee, r), acc_k, logk_k))
    print("\n  [3] bottom cluster K<=8 (logK varies little, acc varies a lot across seeds -- a region capacity cannot order)")
    if len(bottom) >= 4:
        acc_b, logk_b = col(bottom, "acc"), col(bottom, "log_k")
        print(f"  rho(logK, acc) = {spearman(logk_b, acc_b):+.2f} (near 0 means the correlations below carry 'beyond capacity' information)")
        print(corr_line("H(Z) baseline", col(bottom, "h_z"), acc_b, logk_b))
        for r in READINGS:
            print(corr_line(r, col(bottom, r), acc_b, logk_b))
    else:
        print(f"  {len(bottom)} pts < 4, skip")
    print("\n  [4] top cluster K>=64 (reading behavior at the KD tail; directional only)")
    if len(top) >= 3:
        acc_t, logk_t = col(top, "acc"), col(top, "log_k")
        print(f"  rho(logK, acc) = {spearman(logk_t, acc_t):+.2f}")
        print(corr_line("H(Z) baseline", col(top, "h_z"), acc_t, logk_t))
        for r in READINGS:
            print(corr_line(r, col(top, r), acc_t, logk_t))
    else:
        print(f"  {len(top)} pts < 3, skip")
    print("\n  [note] all numbers are post-hoc descriptive statistics (n=14; partial-correlation CIs are extremely wide; directional only); "
          "the pre-registered S1/S2/S3 verdicts are unaffected by anything in this section.")
    # write report
    out = os.path.join(DATA, "cph_post_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"[CP-H post] {jsonl_path}\n")
        f.write(f"knee {len(knee)} pts = bottom {len(bottom)} + top {len(top)}\n")
        for r in READINGS:
            f.write(f"{r}: knee rho(acc)={spearman(col(knee, r), acc_k):+.2f} "
                    f"rho(logK)={spearman(col(knee, r), logk_k):+.2f} "
                    f"partial={partial_spearman(col(knee, r), acc_k, logk_k):+.2f}; "
                    f"bottom rho(acc)={spearman(col(bottom, r), col(bottom, 'acc')) if len(bottom) >= 4 else float('nan'):+.2f}\n")
    print(f"  [archived] {out}")


def main():
    ap = argparse.ArgumentParser(description="CP-H axis-2 gray-zone post-hoc analysis (zero-training, reads JSONL only)")
    ap.add_argument("--jsonl", type=str, default=None, help="readings JSONL (default: newest cph_readings_*.jsonl)")
    args = ap.parse_args()
    path = args.jsonl
    if path is None:
        cands = sorted(glob.glob(os.path.join(DATA, "cph", "cph_readings_*.jsonl")),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            print("[error] no cph_readings_*.jsonl found; use --jsonl to specify one.")
            return
        path = cands[0]
    run(path)


if __name__ == "__main__":
    main()
