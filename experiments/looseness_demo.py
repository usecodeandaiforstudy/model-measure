"""LOOSENESS: how much looser is the certified Fano bound than the PRACTICAL wall, across all tasks?

Motivation (2026-09-30): WITNESS showed the exclusion's usefulness is governed by the looseness of the
bound, not by the witness's budget; MOE showed the two-tier flow bound must be anchored at the MEASURED
wall rather than the Fano bound. This script turns both observations into one number, computed from the
caches the project already has -- ZERO TRAINING, read-only by default.

Usage:
    python experiments/looseness_demo.py                 # survey, print only (no files written)
    python experiments/looseness_demo.py --write         # also write looseness.jsonl / looseness_report.txt
    python experiments/looseness_demo.py --selftest      # pure-function unit tests

Pre-registration (2026-09-30, definitions and expectations before the survey was run)
-----------------------------------------------------------------------------------------------
(1) Scenario = one (task/config, reference model, student grid) triple recovered from a cache. Each
    scenario yields (acc_ref, {K: acc_student}, n_classes).
    Sources (discovered by globbing the cache, so later grid extensions are picked up automatically):
      cp3      experiments/data/cp3/*/          4 cells (MLP/CNN x euclid/cosine), grid64 (64 classes)
      cph      experiments/data/cph/*/          CIFAR-10 (10 classes), the calibrated reference
      pa4      experiments/data/pa4/*/          grid64, four output-mapping doses (64 classes)
      moe      experiments/data/moe.jsonl       route64 dense side (64 classes; teacher not checkpointed)
      witness  experiments/data/witness.jsonl   tasks A/B/C (64/64/10 classes; students = acc_W4 - gap)
(2) Fixed definitions (no tuning):
      R_IB        = task_info_fano(1 - acc_ref, n_classes)     (the certified bound, nats)
      K_min       = ceil(e^R_IB)                               (the certified exclusion threshold)
      K_wall      = measured_wall(grid, accs, acc_ref, rel_tol=0.10)   (practical wall; relative gap)
      wall_ratio  = K_wall / K_min                             (how far the wall sits above the bound)
      looseness   = ln(K_wall) / R_IB                          (nats the bound falls short of the wall)
(3) Expectations to test:
      D1  on the deterministic synthetic task with a near-perfect reference (acc_ref >= 0.99, grid64),
          looseness ~ 1 (the certified bound coincides with the practical wall);
      D2  whenever the reference error is large (eps >= 0.20), looseness >= 3;
      D3  looseness rises with the reference error rate across all scenarios (Spearman >= 0.6), i.e. the
          looseness is governed by the error regime rather than by the class count.
    Verdict (pure function): SUPPORT if D1 and D2 hold, and D3's sign is positive; else MIXED/REFUTED
    (recorded as-is). Scenarios where no grid point reaches the wall (K_wall = None) are reported
    separately as "wall beyond grid" and excluded from the medians (never silently dropped).
(4) Iron rules: zero training; read-only unless --write; English-only output; no downloads.
-----------------------------------------------------------------------------------------------
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import task_info_fano, measured_wall, looseness, DistillCapacity

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "looseness.jsonl")
REPORT = os.path.join(_DATA, "looseness_report.txt")
REL_TOL = 0.10


def _acc(path):
    return float(torch.load(path, map_location="cpu", weights_only=True)["acc"])


def _scenario(source, name, acc_ref, accs_students, n_classes, note=""):
    grid = sorted(accs_students)
    rib = task_info_fano(1.0 - acc_ref, n_classes)
    k_min = DistillCapacity().min_codebook_size(rib)
    k_wall, rel = measured_wall(grid, accs_students, acc_ref, rel_tol=REL_TOL)
    lo = looseness(rib, k_wall)
    return dict(source=source, name=name, acc_ref=acc_ref, eps=1.0 - acc_ref, n_classes=n_classes,
                grid=grid, k_min=k_min, k_wall=k_wall, wall_ratio=(k_wall / k_min if k_wall else None),
                looseness=lo, rib=rib, note=note,
                min_rel_gap=(min(rel.values()) if rel else None))


# ---------------------------------------------------------------------------
# Source scanners (globbing the caches; nothing is trained)
# ---------------------------------------------------------------------------

def scan_dirs(root, filename_re, source, n_classes):
    """Discover <cell>_teacher_s<seed>.pt / <cell>_student_s<seed>_K<K>.pt style caches."""
    pat = re.compile(filename_re)
    found = {}
    for path in glob.glob(os.path.join(root, "*", "*.pt")):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        cell = g.get("cell") or "single"          # the cph pattern has no cell group
        found.setdefault((cell, int(g["s"])), {})[g["kind"] if "K" not in g or g["K"] is None
                                                  else f"student_{g['K']}"] = path
    out = []
    for (cell, seed), files in sorted(found.items()):
        if "teacher" not in files:
            continue
        students = {int(k.split("_")[1]): _acc(v) for k, v in files.items() if k.startswith("student_")}
        if not students:
            continue
        out.append(_scenario(source, f"{cell}_s{seed}", _acc(files["teacher"]), students, n_classes))
    return out


def scan_cp3():
    return scan_dirs(os.path.join(_DATA, "cp3"),
                     r"(?P<cell>[A-Za-z]+-[a-z]+)_(?P<kind>teacher|student)_s(?P<s>\d+)(?:_K(?P<K>\d+))?\.pt$",
                     "cp3", 64)


def scan_cph():
    return scan_dirs(os.path.join(_DATA, "cph"),
                     r"cifar_(?P<kind>teacher|student)_s(?P<s>\d+)(?:_K(?P<K>\d+))?\.pt$",
                     "cph", 10)


def scan_pa4():
    root = os.path.join(_DATA, "pa4")
    pat = re.compile(r"d(?P<d>\d+)_(?P<kind>teacher|student_K(?P<K>\d+))_s(?P<s>\d+)\.pt$")
    found = {}
    for path in glob.glob(os.path.join(root, "*", "*.pt")):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        key = (int(g["d"]), int(g["s"]))
        rec = found.setdefault(key, {})
        if g["kind"] == "teacher":
            rec["teacher"] = path
        else:
            rec[f"K{g['K']}"] = path
    doses = ["0.0", "0.15", "0.3", "0.5"]
    out = []
    for (d, seed), files in sorted(found.items()):
        if "teacher" not in files:
            continue
        students = {int(k[1:]): _acc(v) for k, v in files.items() if k.startswith("K")}
        if students:
            out.append(_scenario("pa4", f"delta{doses[d]}_s{seed}", _acc(files["teacher"]), students, 64))
    return out


def scan_moe():
    """route64 dense side: the teacher is not checkpointed, so read moe.jsonl (one scenario per row)."""
    path = os.path.join(_DATA, "moe.jsonl")
    if not os.path.exists(path):
        return []
    latest = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        latest[(r["s"], round(r["acc_t"], 4))] = r          # de-duplicate relay re-runs
    out = []
    for i, r in enumerate(sorted(latest.values(), key=lambda x: x["acc_t"])):
        students = {int(k): float(v) for k, v in r["dense"]["accs"].items()}
        out.append(_scenario("moe", f"route64_dense_row{i}", float(r["acc_t"]), students, 64))
    return out


def scan_witness():
    """Tasks A/B/C: students are reconstructed as acc_W4 - gap (the run records gaps, not accuracies)."""
    path = os.path.join(_DATA, "witness.jsonl")
    if not os.path.exists(path):
        return []
    latest = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        latest[(r["task"], r["seed"])] = r                  # the last row carries the widest grid
    out = []
    for (task, seed), r in sorted(latest.items()):
        acc_ref = float(r["accs"]["W4"])
        students = {int(k): acc_ref - float(v) for k, v in r["gap_by_K"].items()}
        n = 10 if task == "C" else 64
        out.append(_scenario("witness", f"task{task}_s{seed}", acc_ref, students, n))
    return out


SCANNERS = [("cp3", scan_cp3), ("cph", scan_cph), ("pa4", scan_pa4),
            ("moe", scan_moe), ("witness", scan_witness)]


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def rank_correlation(xs, ys):
    """Spearman rank correlation with average ranks for ties (pure function)."""
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
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def median(vals):
    v = sorted(vals)
    return v[len(v) // 2] if v else None


def loos_verdict(scenarios):
    """Pre-registered verdict (docstring (3))."""
    usable = [s for s in scenarios if s["looseness"] is not None]
    beyond = [s for s in scenarios if s["k_wall"] is None]
    det = [s for s in usable if s["n_classes"] == 64 and s["acc_ref"] >= 0.99]
    hi = [s for s in usable if s["eps"] >= 0.20]
    d1 = len(det) == 0 or median([s["looseness"] for s in det]) <= 1.5
    d2 = len(hi) == 0 or all(s["looseness"] >= 3.0 for s in hi)
    rho = rank_correlation([s["eps"] for s in usable], [s["looseness"] for s in usable])
    d3 = rho >= 0.6
    detail = [
        f"  scenarios: {len(scenarios)} total, {len(usable)} with a located wall, {len(beyond)} beyond grid",
        f"  D1 (deterministic, acc_ref>=0.99, 64 classes): n={len(det)}, "
        f"median looseness={median([s['looseness'] for s in det]) if det else None} -> "
        f"{'PASS' if d1 else 'FAIL'}",
        f"  D2 (high-error scenarios eps>=0.20): n={len(hi)}, "
        f"min looseness={min([s['looseness'] for s in hi]) if hi else None} -> {'PASS' if d2 else 'FAIL'}",
        f"  D3 Spearman(eps, looseness) = {rho:+.3f} (>= 0.6) -> {'PASS' if d3 else 'FAIL'}",
    ]
    if d1 and d2 and d3:
        return "SUPPORT: the certified bound's looseness is governed by the error regime; it is ~1 only " \
               "on (near-)deterministic tasks and >= 3 whenever the reference error is large", detail
    if d1 and d2:
        return "PARTIAL: D1/D2 hold but the error-regime trend (D3) is not confirmed", detail
    return "MIXED: the pre-registered expectations are not all met (recorded as-is)", detail


def run_selftest():
    ok = []
    ok.append(("median", median([3.0, 1.0, 2.0]) == 2.0))
    ok.append(("rank_correlation perfect positive", abs(rank_correlation([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9))
    ok.append(("rank_correlation perfect negative",
               abs(rank_correlation([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9))
    ok.append(("rank_correlation ties", rank_correlation([1, 1, 2], [5, 5, 9]) > 0.9))
    d1_only = [dict(n_classes=64, acc_ref=1.00, eps=0.00, looseness=1.0, k_wall=64),
               dict(n_classes=64, acc_ref=0.90, eps=0.10, looseness=2.0, k_wall=96),
               dict(n_classes=10, acc_ref=0.80, eps=0.20, looseness=4.0, k_wall=128)]
    v, _ = loos_verdict(d1_only)
    ok.append(("verdict SUPPORT on a clean three-point set", v.startswith("SUPPORT")))
    ok.append(("rank_correlation needs n>=3 (two points -> 0)", rank_correlation([0.0, 0.2], [1.0, 4.0]) == 0.0))
    bad = [dict(n_classes=64, acc_ref=0.8, eps=0.2, looseness=1.0, k_wall=8)]
    v, _ = loos_verdict(bad)
    ok.append(("verdict not SUPPORT when D2 fails", not v.startswith("SUPPORT")))
    print("[LOOSENESS selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="LOOSENESS: certified Fano bound vs the practical wall")
    ap.add_argument("--write", action="store_true", help="also write looseness.jsonl and the report")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no data)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    print("[LOOSENESS] certified bound vs practical wall (zero training; caches read-only)")
    print(f"  definitions: rel_tol={REL_TOL}; wall_ratio = K_wall/K_min; looseness = ln(K_wall)/R_IB")
    scenarios = []
    for source, fn in SCANNERS:
        got = fn()
        print(f"  [scan] {source}: {len(got)} scenarios")
        scenarios.extend(got)
    if not scenarios:
        print("  [abort] no cached scenario found")
        return
    print(f"\n  {'source':9s} {'scenario':22s} {'acc_ref':>7s} {'eps':>5s} {'C':>3s} "
          f"{'R_IB':>5s} {'K_min':>5s} {'K_wall':>6s} {'ratio':>6s} {'loose':>5s}")
    for s in sorted(scenarios, key=lambda x: (x["looseness"] is None, -(x["looseness"] or 0))):
        kw = str(s["k_wall"]) if s["k_wall"] else ">" + str(max(s["grid"]))
        ratio = f"{s['wall_ratio']:.1f}" if s["wall_ratio"] else "-"
        loose = f"{s['looseness']:.2f}" if s["looseness"] is not None else "-"
        print(f"  {s['source']:9s} {s['name']:22s} {s['acc_ref']:7.3f} {s['eps']:5.3f} "
              f"{s['n_classes']:3d} {s['rib']:5.2f} {s['k_min']:5d} {kw:>6s} {ratio:>6s} {loose:>5s}")
    print("\n  --- LOOSENESS verdict ---")
    verdict, detail = loos_verdict(scenarios)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    if args.write:
        with open(JSONL, "a", encoding="utf-8") as f:
            for s in scenarios:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("[LOOSENESS] certified bound vs practical wall\n")
            for s in scenarios:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            for line in detail:
                f.write(line + "\n")
            f.write("verdict: " + verdict + "\n")
        print(f"  [written] {JSONL} ; {REPORT}")


if __name__ == "__main__":
    main()
