"""BUDGET DOSE-RESPONSE: is the student's retention deficit a power law in the training budget, and is its
exponent the SAME across tasks?

Where the question comes from. WALL-SATURATION measured the deficit at two budgets on two instances and found
a ratio of 0.55 (C=216, sigma=0.5, K=4096, 60 -> 180 epochs) and 0.64 (C=1000, sigma=1.0, K=2048, 60 -> 180);
SCALE-C section 7.4 recorded the second as a confirmation of the first. Two points per instance is a ratio, not
a law. This script turns it into a dose-response curve: several budgets on the same instance, a power-law fit,
and a test of whether one task's exponent predicts another's deficit.

Why it matters for the tool. If the deficit follows a shared power law, then the budget needed to reach the
retention bar is *predictable* rather than something every deployment has to search for -- and, just as
importantly, the wall's protocol-relativity acquires a functional form instead of being a caveat.

Pre-registration (2026-10-05, written before the runs below)
--------------------------------------------------------------------------------------------
Instances (all trained with the cached-target distillation objective so a rung costs one training run):
  I1  m=4  (C=64),  sigma=0.5, K=1024, epochs {60, 120, 240, 480}   -- cheap, four rungs
  I2  m=6  (C=216), sigma=0.5, K=4096, epochs {60, 180, 360}       -- the instance WALL-SATURATION used; the
                                                                      60/180 rungs are reused from its cache
  I3  m=4  (C=64),  sigma=0.5, K=256,  epochs {60, 240}            -- is the exponent K-dependent?
Measurement: rel = (acc_ref - acc_student)/acc_ref at each rung; the reference is trained once per instance
(GridMLP, VQ(2048), 60 epochs, 150 samples/class, seed 0, identical to the earlier campaigns).
Criteria:
  P1 SHAPE      per instance, log(rel) is affine in log(epochs) with R^2 >= 0.90 (a power law, not a floor);
  P2 SHARED b   the exponents of I1 and I2 agree within 0.15;
  P3 PREDICTIVE using I1's exponent and level to predict I2's deficit at I2's largest rung: relative error
                <= 30%;
  P4 BAR        report the budget each instance's fit predicts for reaching rel = 0.10 -- flagged as an
                EXTRAPOLATION, not a measurement.
Verdict: SUPPORT if P1 and (P2 or P3); MIXED if only P1; REFUTED if P1 fails on any instance.
Iron rules: criteria before data; the cached rungs of I2 are reused rather than retrained (identical protocol);
extrapolations are labelled; relay mode and --verdict-only as everywhere else.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import task_info_fano, DistillCapacity, train_one_epoch, evaluate
from scale_c_demo import make_lattice, GridMLP, class_count
from scale_c_clean_demo import train_one_epoch_cached, teacher_logits

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(_DATA, "budget_dose")
JSONL = os.path.join(_DATA, "budget_dose.jsonl")
REPORT = os.path.join(_DATA, "budget_dose_report.txt")
WALL_SAT = os.path.join(_DATA, "wall_sat")

SIGMA = 0.5
SPACING = 3.0
N_PER_CLASS = 150
EPOCHS = 60
LR = 1e-2
SEED = 0
K_TEACHER = 2048

INSTANCES = [
    dict(tag="I1", m=4, k=1024, epochs=[60, 120, 240, 480]),
    dict(tag="I2", m=6, k=4096, epochs=[60, 180, 360]),
    dict(tag="I3", m=4, k=256, epochs=[60, 240]),
]


def fit_power(xs, ys):
    """Fit log(y) = log(a) + b log(x); returns (a, b, r2) with y > 0 required."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    b = sxy / sxx if sxx else 0.0
    lna = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum((y - (lna + b * x)) ** 2 for x, y in zip(lx, ly))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return math.exp(lna), b, r2


def _cached_rung(m, sigma, k, epochs):
    """Reuse a rung that a previous campaign already trained, when its protocol matches exactly."""
    C = class_count(m)
    cands = [
        (os.path.join(WALL_SAT, f"m{m}_C{C}", f"student_sigma{sigma}_K{k}.pt"), EPOCHS),
        (os.path.join(WALL_SAT, f"m{m}_C{C}", f"student_budget{epochs}_sigma{sigma}_K{k}.pt"), epochs),
    ]
    for path, ep in cands:
        if ep == epochs and os.path.exists(path):
            return float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
    return None


def run_dose(instance, device, resume=True):
    m, k = instance["m"], instance["k"]
    C = class_count(m)
    cache = os.path.join(ROOT, f"{instance['tag']}_C{C}_K{k}")
    os.makedirs(cache, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, n_per_class=N_PER_CLASS, sigma=SIGMA, spacing=SPACING, seed=SEED)
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    path_t = os.path.join(cache, "teacher.pt")
    # the reference of this instance: reuse WALL-SATURATION's sigma=0.5 teacher when the task matches exactly
    reuse = os.path.join(WALL_SAT, f"m{m}_C{C}", f"teacher_sigma{SIGMA}.pt")
    if not os.path.exists(path_t) and os.path.exists(reuse):
        torch.save(torch.load(reuse, map_location="cpu", weights_only=True), path_t)
    if resume and os.path.exists(path_t):
        acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    else:
        torch.manual_seed(70000 + C)
        ref = GridMLP(C, K_TEACHER).to(device)
        opt = torch.optim.Adam(ref.parameters(), lr=LR)
        for _ in range(EPOCHS):
            train_one_epoch(ref, tr, opt, device=device)
        acc_t, _ = evaluate(ref, te, device=device)
        torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
    ref = GridMLP(C, K_TEACHER).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    path_c = os.path.join(cache, "targets.pt")
    if resume and os.path.exists(path_c):
        targets = torch.load(path_c, map_location="cpu", weights_only=True).float()
    else:
        targets = teacher_logits(ref, tr, device).float()
        torch.save(targets, path_c)
    eps = 1.0 - acc_t
    rib = task_info_fano(eps, C)
    print(f"\n  [{instance['tag']}] C={C}, sigma={SIGMA}, K={k}: reference acc={acc_t:.3f} (eps={eps:.3f}) "
          f"R_IB={rib:.2f} K_min={DistillCapacity().min_codebook_size(rib)}")
    rows, t0 = [], time.time()
    for ep in instance["epochs"]:
        acc = _cached_rung(m, SIGMA, k, ep) if resume else None
        src = "reused"
        if acc is None:
            path_s = os.path.join(cache, f"student_K{k}_ep{ep}.pt")
            if resume and os.path.exists(path_s):
                acc = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
                src = "cached"
            else:
                torch.manual_seed(71000 + C + k)
                stu = GridMLP(C, k).to(device)
                opt = torch.optim.Adam(stu.parameters(), lr=LR)
                loader = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=128, shuffle=True,
                                    num_workers=0)
                for _ in range(ep):
                    train_one_epoch_cached(stu, loader, opt, device=device)
                acc, _ = evaluate(stu, te, device=device)
                torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
                src = "trained"
        rel = (acc_t - acc) / acc_t
        rows.append(dict(arm="dose", tag=instance["tag"], c=C, m=m, sigma=SIGMA, k=k, epochs=ep,
                         acc_t=acc_t, acc=acc, rel=rel, source=src, rib=rib,
                         k_min=DistillCapacity().min_codebook_size(rib)))
        print(f"    epochs={ep:4d}: acc={acc:.3f} -> deficit {rel:+.4f}   [{src}, {time.time() - t0:.0f}s]")
    return rows


def dose_verdict(rows):
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    fits = {}
    detail = []
    p1 = True
    for tag, rs in sorted(by_tag.items()):
        rs = sorted(rs, key=lambda r: r["epochs"])
        pos = [r for r in rs if r["rel"] > 0]
        if len(pos) < 2:
            detail.append(f"  {tag}: fewer than 2 positive rungs -> cannot fit")
            p1 = False
            continue
        a, b, r2 = fit_power([r["epochs"] for r in pos], [r["rel"] for r in pos])
        fits[tag] = dict(a=a, b=b, r2=r2, n=len(pos),
                         bar_budget=(a / 0.10) ** (1.0 / -b) if b < 0 else None)
        detail.append(f"  {tag} (K={rs[0]['k']}, C={rs[0]['c']}): rel ~ ({a:.4f}) * epochs^({b:+.3f}), "
                      f"R^2={r2:.3f} over {len(pos)} rungs -> {'PASS' if r2 >= 0.90 else 'FAIL'}")
        if r2 < 0.90:
            p1 = False
    p2 = p3 = False
    if "I1" in fits and "I2" in fits:
        db = abs(fits["I1"]["b"] - fits["I2"]["b"])
        p2 = db <= 0.15
        detail.append(f"  P2 shared exponent: I1 b={fits['I1']['b']:+.3f} vs I2 b={fits['I2']['b']:+.3f}, "
                      f"|diff|={db:.3f} (tol 0.15) -> {'PASS' if p2 else 'FAIL'}")
        ep_max = max(r["epochs"] for r in by_tag["I2"])
        rel_obs = [r["rel"] for r in by_tag["I2"] if r["epochs"] == ep_max][0]
        pred = fits["I1"]["a"] * ep_max ** fits["I1"]["b"]
        err = abs(pred - rel_obs) / rel_obs if rel_obs else float("inf")
        p3 = err <= 0.30
        detail.append(f"  P3 predictive: I1's fit predicts I2's deficit at {ep_max} epochs = {pred:.4f} vs "
                      f"observed {rel_obs:.4f} (relative error {err:.1%}, tol 30%) -> "
                      f"{'PASS' if p3 else 'FAIL'}")
    if "I3" in fits:
        detail.append(f"  P4 (extrapolated budget to reach rel=0.10): "
                      + ", ".join(f"{t}: {int(f['bar_budget'])} epochs" if f.get("bar_budget") else f"{t}: n/a"
                                  for t, f in sorted(fits.items()))
                      + "  [EXTRAPOLATION, not measured]")
        if "I1" in fits and "I3" in fits:
            detail.append(f"  K-dependence of the exponent: I1 (K=1024) b={fits['I1']['b']:+.3f} vs "
                          f"I3 (K=256) b={fits['I3']['b']:+.3f}")
    if p1 and (p2 or p3):
        v = "SUPPORT: the deficit is a power law in the budget, with a shared exponent across tasks"
    elif p1:
        v = "MIXED: the deficit is a power law per instance, but the exponents do not transfer"
    else:
        v = "REFUTED: the dose-response is not a power law at this resolution"
    return v, detail, fits


def overfit_check(device, m=6, sigma=SIGMA, k=4096):
    """Why does the deficit regress at 360 epochs? Test the overfitting explanation instead of asserting it.

    The students distil from a FINITE cached sample of the teacher's soft targets (150 samples/class), so a
    longer budget can fit those targets better while generalising worse. The check evaluates the saved
    checkpoints of two rungs on the TRAIN and TEST splits: if train retention improves while test retention
    degrades, overfitting to the cached targets is measured rather than inferred.
    """
    from capacity_lens import evaluate
    C = class_count(m)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, n_per_class=N_PER_CLASS, sigma=sigma, spacing=SPACING, seed=SEED)
    tr_eval = DataLoader(TensorDataset(x_tr, y_tr), batch_size=512, shuffle=False, num_workers=0)
    te_eval = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    out = []
    for ep, path in ((60, os.path.join(WALL_SAT, f"m{m}_C{C}", f"student_sigma{sigma}_K{k}.pt")),
                     (180, os.path.join(WALL_SAT, f"m{m}_C{C}", f"student_budget180_sigma{sigma}_K{k}.pt")),
                     (360, os.path.join(ROOT, f"I2_C{C}_K{k}", f"student_K{k}_ep360.pt"))):
        if not os.path.exists(path):
            print(f"  epochs={ep}: checkpoint missing ({path})")
            continue
        stu = GridMLP(C, k).to(device)
        stu.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
        acc_tr, _ = evaluate(stu, tr_eval, device=device)
        acc_te, _ = evaluate(stu, te_eval, device=device)
        out.append(dict(arm="overfit_check", epochs=ep, acc_train=acc_tr, acc_test=acc_te, k=k, c=C))
        print(f"  epochs={ep:4d}: train acc={acc_tr:.4f}   test acc={acc_te:.4f}")
    if len(out) >= 2:
        best_test = max(out, key=lambda r: r["acc_test"])
        worst_test = min(out, key=lambda r: r["acc_test"])
        v = (f"the longest budget has the highest train accuracy but not the highest test accuracy "
             f"({out[-1]['acc_train']:.4f} vs {best_test['acc_train']:.4f} train; "
             f"{out[-1]['acc_test']:.4f} vs {best_test['acc_test']:.4f} test) -> "
             + ("overfitting to the cached soft targets is MEASURED"
                if out[-1]["acc_test"] < best_test["acc_test"] - 1e-9
                and out[-1]["acc_train"] >= best_test["acc_train"] - 1e-9
                else "the overfitting explanation is NOT supported by this check"))
        print(f"  [check] {v}")
        with open(JSONL, "a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return out, v
    return out, "UNDECIDABLE"


def run_selftest():
    ok = []
    a, b, r2 = fit_power([1, 2, 4, 8], [0.5 * x ** -0.5 for x in (1, 2, 4, 8)])
    ok.append((f"power-law fit exact (a={a:.4f}, b={b:+.3f}, R^2={r2:.3f})",
               abs(b + 0.5) < 1e-9 and abs(r2 - 1.0) < 1e-9))
    rows = [dict(tag="I1", c=64, k=1024, epochs=e, rel=0.4 * e ** -0.5) for e in (60, 120, 240, 480)]
    rows += [dict(tag="I2", c=216, k=4096, epochs=e, rel=0.9 * e ** -0.5) for e in (60, 180, 360)]
    v, detail, fits = dose_verdict(rows)
    ok.append(("identical exponents -> SUPPORT", v.startswith("SUPPORT")))
    ok.append((f"extrapolated bar budget reported ({int(fits['I1']['bar_budget'])} epochs for I1)",
               fits["I1"]["bar_budget"] is not None))
    bad = [dict(r, rel=r["rel"] * (2.0 if r["tag"] == "I2" else 1.0) * (1.0 + 0.5 * i))
           for i, r in enumerate(rows)]
    v2, _, _ = dose_verdict(bad)
    ok.append(("a non-power-law dose-response is not called SUPPORT", not v2.startswith("SUPPORT")))
    flat = [dict(tag="I1", c=64, k=1024, epochs=e, rel=0.2) for e in (60, 120, 240, 480)]
    v3, _, _ = dose_verdict(flat)
    ok.append(("a flat deficit (no dose response) is REFUTED", v3.startswith("REFUTED")))
    print("[BUDGET DOSE-RESPONSE selftest] the power-law criterion:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    rows = [r for r in rows if r.get("arm") == "dose"]
    if not rows:
        raise SystemExit("[BUDGET DOSE] nothing recorded yet")
    v, detail, _ = dose_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="BUDGET DOSE-RESPONSE: power law in the training budget?")
    ap.add_argument("--tag", nargs="+", default=[i["tag"] for i in INSTANCES])
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    ap.add_argument("--overfit-check", action="store_true",
                    help="evaluate the saved rungs on train vs test (why the deficit regresses)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.overfit_check:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        print("=" * 100)
        print("[BUDGET DOSE] overfitting check: does the longest budget fit the cache better but generalise worse?")
        print("=" * 100)
        overfit_check(device)
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for inst in INSTANCES:
        if inst["tag"] not in args.tag:
            continue
        rows = run_dose(inst, device, resume=not args.fresh)
        with open(JSONL, "a", encoding="utf-8") as f:      # append per instance: an interruption keeps them
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    all_rows = [r for r in all_rows if r.get("arm") == "dose"]
    print()
    v, detail, _ = dose_verdict(all_rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")
    with open(REPORT, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + v + "\n")
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
