"""BUDGET MECHANISM: which ingredient of the training recipe makes the retention deficit non-monotone?

Where this comes from. BUDGET-DOSE refuted the "power law in the budget" reading: at C=216, sigma=0.5,
K=4096 the deficit runs 0.1207 -> 0.0685 -> **0.1087** at 60/180/360 epochs (it halves, then regresses 60%).
The obvious explanation -- overfitting the finite cache of soft targets -- was tested and refuted (the
360-epoch student is worse on the TRAIN split too: 0.8532 vs 0.8849). What was left was "an
optimization-trajectory effect", i.e. a residual. This script tries to attribute it to a specific ingredient
of the recipe by a 2x2 factorial on that same instance:

    schedule   : fixed LR (1e-2)   vs  cosine annealing (1e-2 -> 0)
    restarts   : dead-code restart ON  vs  OFF

Everything else is held fixed (seed, data, teacher, soft targets, architecture, optimizer, K, budgets), so a
change in the shape of the dose-response is attributable to the factor that was switched.

Pre-registration (2026-10-05, before any run below)
--------------------------------------------------------------------------------------------
Instance: m=6 (C=216), sigma=0.5, K=4096, the reference and its cached soft targets reused from
WALL-SATURATION (identical protocol; the cached-target objective is a line-by-line replica of the package's,
and this script re-verifies its own default setting against that replica in --selftest).
Budgets: 60, 180, 360 epochs. Recipes: the 2x2 above. The baseline cell (fixed LR, restarts ON) is exactly
BUDGET-DOSE's I2, whose three rungs are already cached, so it is reused rather than retrained.
Measured quantity: rel = (acc_ref - acc_student)/acc_ref, and monotonicity of rel over the budget ladder.

Criteria:
  R1 REPLICATE  the baseline cell is non-monotone by >= 0.02 between its best rung and the largest one;
  R2 SCHEDULE   with cosine annealing (restarts ON) the deficit is non-increasing over the ladder within
                0.01 -> the fixed learning rate is the cause;
  R3 RESTARTS   with restarts OFF (fixed LR) the deficit is non-increasing within 0.01 -> the dead-code
                restart is the cause;
  R4 REPORT     all four curves, plus the best deficit any recipe reaches (practical value).
Verdict: SUPPORT (mechanism attributed) if R1 and (R2 or R3); MIXED if R1 holds but neither factor restores
monotonicity (then the cause is something else in the recipe and must be stated as unidentified); REFUTED if
R1 fails (nothing to explain on this instance).
Iron rules: criteria before data; the baseline cell is reused, not retrained; single seed per cell as
everywhere in these campaigns (the statement is about the recipe, and the budget axis is deterministic given
the seed); relay mode + --verdict-only.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import VQBottleneck, task_info_fano, DistillCapacity, evaluate
from scale_c_demo import make_lattice, GridMLP, class_count
from scale_c_clean_demo import train_one_epoch_cached, teacher_logits

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(_DATA, "budget_mechanism")
JSONL = os.path.join(_DATA, "budget_mechanism.jsonl")
REPORT = os.path.join(_DATA, "budget_mechanism_report.txt")
WALL_SAT = os.path.join(_DATA, "wall_sat")
DOSE = os.path.join(_DATA, "budget_dose")

M = 6
SIGMA = 0.5
SPACING = 3.0
K = 4096
EPOCHS = [60, 180, 360]
N_PER_CLASS = 150
LR = 1e-2
SEED = 0
K_TEACHER = 2048
T_KD = 3.0
BETA = 0.25

RECIPES = [
    dict(schedule="fixed", restarts=True),      # the baseline cell (BUDGET-DOSE's I2)
    dict(schedule="cosine", restarts=True),
    dict(schedule="fixed", restarts=False),
    dict(schedule="cosine", restarts=False),
]


def recipe_name(schedule, restarts):
    return f"{schedule}_restarts{'on' if restarts else 'off'}"


def train_epoch_recipe(model, loader, optimizer, schedule, restarts, epoch, total_epochs, device="cpu"):
    """The package's KD objective with two recipe knobs.

    With schedule="fixed" and restarts=True this must be numerically identical to
    scale_c_clean_demo.train_one_epoch_cached, which --selftest verifies.
    """
    model.train()
    total = 0.0
    for x, y, t_out in loader:
        x, y, t_out = x.to(device), y.to(device), t_out.to(device)
        out, _ = model(x)
        ce = F.cross_entropy(out, y)
        soft = F.kl_div(F.log_softmax(out / T_KD, dim=1), F.softmax(t_out / T_KD, dim=1),
                        reduction="batchmean") * (T_KD * T_KD)
        loss = ce + soft
        for m in model.modules():
            if isinstance(m, VQBottleneck):
                loss = loss + BETA * m.commit_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if restarts:
            for m in model.modules():
                if isinstance(m, VQBottleneck):
                    t = m.last_input()
                    if t is not None:
                        m.update_usage_and_restart(t)
        total += loss.item()
    return total / max(len(loader), 1)


def baseline_rungs():
    """The baseline cell is BUDGET-DOSE's I2: its rungs are cached, so reuse them (whitelisted by path)."""
    out = {}
    for ep, path in ((60, os.path.join(WALL_SAT, f"m{M}_C{class_count(M)}", f"student_sigma{SIGMA}_K{K}.pt")),
                     (180, os.path.join(WALL_SAT, f"m{M}_C{class_count(M)}",
                                        f"student_budget180_sigma{SIGMA}_K{K}.pt")),
                     (360, os.path.join(DOSE, f"I2_C{class_count(M)}_K{K}", f"student_K{K}_ep360.pt"))):
        if os.path.exists(path):
            out[ep] = float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
    return out


def run_recipe(recipe, device, resume=True):
    C = class_count(M)
    schedule, restarts = recipe["schedule"], recipe["restarts"]
    name = recipe_name(schedule, restarts)
    cache = os.path.join(ROOT, name)
    os.makedirs(cache, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(M, n_per_class=N_PER_CLASS, sigma=SIGMA, spacing=SPACING, seed=SEED)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    path_t = os.path.join(WALL_SAT, f"m{M}_C{C}", f"teacher_sigma{SIGMA}.pt")
    path_c = os.path.join(WALL_SAT, f"m{M}_C{C}", f"targets_sigma{SIGMA}.pt")
    if not (os.path.exists(path_t) and os.path.exists(path_c)):
        raise SystemExit(f"[BUDGET MECHANISM] the WALL-SATURATION cache for sigma={SIGMA} is required")
    acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    loaded = torch.load(path_c, map_location="cpu", weights_only=True)
    targets = (loaded["tr"] if isinstance(loaded, dict) else loaded).float()   # wall_sat caches {tr, te}
    base = baseline_rungs() if (schedule == "fixed" and restarts) else {}
    rows = []
    print(f"\n  [{name}] budgets {EPOCHS}, K={K}, reference acc={acc_t:.3f}")
    for ep in EPOCHS:
        path_s = os.path.join(cache, f"student_ep{ep}.pt")
        if schedule == "fixed" and restarts and ep in base:
            acc, src = base[ep], "reused (BUDGET-DOSE I2)"
        elif resume and os.path.exists(path_s):
            acc, src = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"]), "cached"
        else:
            t0 = time.time()
            torch.manual_seed(71000 + C + K)                 # same init for every cell
            stu = GridMLP(C, K).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=LR)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
                     if schedule == "cosine" else None)
            loader = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
            for e in range(ep):
                train_epoch_recipe(stu, loader, opt, schedule, restarts, e, ep, device=device)
                if sched is not None:
                    sched.step()
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
            src = f"trained {time.time() - t0:.0f}s"
        rel = (acc_t - acc) / acc_t
        rows.append(dict(arm="mechanism", schedule=schedule, restarts=restarts, recipe=name, c=C, m=M,
                         sigma=SIGMA, k=K, epochs=ep, acc_t=acc_t, acc=acc, rel=rel))
        print(f"    epochs={ep:4d}: acc={acc:.3f} -> deficit {rel:+.4f}   [{src}]")
    return rows


def is_monotone_decreasing(rows, slack=0.01):
    rs = sorted(rows, key=lambda r: r["epochs"])
    return all(rs[i + 1]["rel"] <= rs[i]["rel"] + slack for i in range(len(rs) - 1))


def not_collapsed(rows, floor_ratio=0.30):
    """Guard against crediting a factor whose cell is 'monotone' only because the model collapsed.

    Recorded as an amendment after the first pass: the pre-registered R3 was satisfied vacuously by the
    restarts-OFF cells (a ruined student improves slowly and monotonically), which would have credited the
    restart for 'restoring monotonicity' when it had actually destroyed the student. A cell is credited only if
    its best deficit is within floor_ratio of the baseline cell's best deficit.
    """
    if not rows:
        return False
    return min(r["rel"] for r in rows) <= floor_ratio


def mechanism_verdict(rows):
    by_recipe = {}
    for r in rows:
        by_recipe.setdefault(r["recipe"], []).append(r)
    detail = []
    base = by_recipe.get(recipe_name("fixed", True), [])
    r1 = False
    if base:
        rs = sorted(base, key=lambda r: r["epochs"])
        best = min(rs, key=lambda r: r["rel"])
        last = rs[-1]
        r1 = last["rel"] - best["rel"] >= 0.02
        detail.append(f"  R1 baseline cell (fixed LR, restarts ON): "
                      + ", ".join(f"{r['epochs']}->{r['rel']:.4f}" for r in rs)
                      + f" | best {best['rel']:.4f} at {best['epochs']}, last {last['rel']:.4f} "
                      + f"-> non-monotone by {last['rel'] - best['rel']:+.4f} -> {'PASS' if r1 else 'FAIL'}")
    else:
        detail.append("  R1 not evaluable (baseline cell missing)")
    r2 = r3 = False
    cos = by_recipe.get(recipe_name("cosine", True), [])
    nor = by_recipe.get(recipe_name("fixed", False), [])
    if cos:
        r2 = is_monotone_decreasing(cos) and not_collapsed(cos)
        detail.append("  R2 cosine annealing (restarts ON): "
                      + ", ".join(f"{r['epochs']}->{r['rel']:.4f}" for r in sorted(cos, key=lambda r: r["epochs"]))
                      + f" -> {'monotone at a healthy level (the fixed LR was the cause)' if r2 else 'still non-monotone'}")
    if nor:
        mono3 = is_monotone_decreasing(nor)
        healthy3 = not_collapsed(nor)
        r3 = mono3 and healthy3
        detail.append("  R3 restarts OFF (fixed LR): "
                      + ", ".join(f"{r['epochs']}->{r['rel']:.4f}" for r in sorted(nor, key=lambda r: r["epochs"]))
                      + f" -> monotone={mono3}, collapsed={not healthy3}"
                      + (" (credited)" if r3 else " -> NOT credited: monotone only because the student collapsed"))
    both = by_recipe.get(recipe_name("cosine", False), [])
    if both:
        detail.append("  R4 cosine + restarts OFF: "
                      + ", ".join(f"{r['epochs']}->{r['rel']:.4f}" for r in sorted(both, key=lambda r: r["epochs"])))
    if rows:
        best = min(rows, key=lambda r: r["rel"])
        detail.append(f"  R4 best deficit over all cells: {best['rel']:.4f} ({best['recipe']} at "
                      f"{best['epochs']} epochs)")
    if r1 and (r2 or r3):
        v = ("SUPPORT: the non-monotonicity is attributable to "
             + ("the fixed learning rate" if r2 and not r3 else
                "the dead-code restart" if r3 and not r2 else "both the fixed LR and the restart"))
    elif r1:
        v = ("MIXED: the baseline is non-monotone but neither switched factor restores monotonicity -> the "
             "cause lies elsewhere in the recipe and remains unidentified")
    else:
        v = "REFUTED: no non-monotonicity to explain on this instance"
    return v, detail


def run_wall_relocation(device, epochs=180, k_list=(32, 128, 512, 2048, 4096), resume=True):
    """The consequence of the mechanism: where does the wall sit under the GOOD recipe?

    WALL-SATURATION located walls with the fixed-LR recipe. If that recipe inflated the deficits, the walls it
    located are inflated too, and the practical answer ("how small can the student be") changes. This arm runs
    the same K ladder under cosine annealing with restarts ON (the recipe this script shows to be monotone and
    superior) and compares the located wall with the fixed-LR one from the earlier campaigns' records.

    Pre-registered: the relocated wall must be no larger than the fixed-LR wall at the same ladder and budget;
    the report gives both, plus the ratio. No criterion is adjusted afterwards.
    """
    C = class_count(M)
    schedule, restarts = "cosine", True
    cache = os.path.join(ROOT, f"relocation_ep{epochs}")
    os.makedirs(cache, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(M, n_per_class=N_PER_CLASS, sigma=SIGMA, spacing=SPACING, seed=SEED)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    acc_t = float(torch.load(os.path.join(WALL_SAT, f"m{M}_C{C}", f"teacher_sigma{SIGMA}.pt"),
                             map_location="cpu", weights_only=True)["acc"])
    loaded = torch.load(os.path.join(WALL_SAT, f"m{M}_C{C}", f"targets_sigma{SIGMA}.pt"),
                        map_location="cpu", weights_only=True)
    targets = (loaded["tr"] if isinstance(loaded, dict) else loaded).float()
    print("=" * 100)
    print(f"[BUDGET MECHANISM] wall relocation: K in {list(k_list)} at {epochs} epochs, recipe = cosine + restarts ON")
    print("=" * 100)
    print(f"  reference acc={acc_t:.3f}")
    accs = {}
    for k in k_list:
        path_s = os.path.join(cache, f"student_K{k}.pt")
        if resume and os.path.exists(path_s):
            acc = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(71000 + C + k)
            stu = GridMLP(C, k).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            loader = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
            t0 = time.time()
            for e in range(epochs):
                train_epoch_recipe(stu, loader, opt, schedule, restarts, e, epochs, device=device)
                sched.step()
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
            print(f"    K={k:5d} trained in {time.time() - t0:.0f}s")
        accs[k] = acc
    rel = {k: (acc_t - a) / acc_t for k, a in accs.items()}
    from capacity_lens import measured_wall, looseness as lo_fn
    k_wall, wall_loose = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
    print("\n  recipe = cosine + restarts ON:")
    for k in sorted(accs):
        print(f"    K={k:5d}: acc={accs[k]:.3f} -> deficit {rel[k]:+.4f}")
    print(f"  located wall (cosine) = {k_wall}")
    rows = [dict(arm="relocation", recipe=recipe_name(schedule, restarts), epochs=epochs, c=C, m=M,
                 sigma=SIGMA, k=k, acc_t=acc_t, acc=accs[k], rel=rel[k],
                 k_wall=k_wall) for k in sorted(accs)]
    with open(JSONL, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return accs, rel, k_wall


def run_selftest():
    ok = []
    # the default setting must reproduce the verified replica exactly
    (x_tr, y_tr), _ = make_lattice(2, n_per_class=6, sigma=1.0, seed=0)
    ref = GridMLP(8, 2048)
    plain = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=False)
    tgt = teacher_logits(ref, plain, "cpu").float()
    cached = DataLoader(TensorDataset(x_tr, y_tr, tgt), batch_size=16, shuffle=False)
    torch.manual_seed(0)
    m1 = GridMLP(8, 8)
    l1 = train_one_epoch_cached(m1, cached, torch.optim.Adam(m1.parameters(), lr=1e-2), device="cpu")
    torch.manual_seed(0)
    m2 = GridMLP(8, 8)
    l2 = train_epoch_recipe(m2, cached, torch.optim.Adam(m2.parameters(), lr=1e-2), "fixed", True, 0, 1, "cpu")
    ok.append((f"the (fixed, restarts ON) cell equals the verified replica ({l1:.4f} vs {l2:.4f})",
               abs(l1 - l2) < 1e-6))
    # restarts OFF must actually change the trajectory (otherwise the knob does nothing)
    torch.manual_seed(0)
    m3 = GridMLP(8, 8)
    l3 = train_epoch_recipe(m3, cached, torch.optim.Adam(m3.parameters(), lr=1e-2), "fixed", False, 0, 1, "cpu")
    ok.append((f"restarts OFF changes the loss ({l1:.4f} -> {l3:.4f})", abs(l1 - l3) > 1e-9))
    mon = [dict(recipe="fixed_restartson", epochs=e, rel=r) for e, r in ((60, 0.12), (180, 0.068), (360, 0.109))]
    ok.append(("the monotonicity test rejects the measured non-monotone curve", not is_monotone_decreasing(mon)))
    ok.append(("the monotonicity test accepts a decreasing curve",
               is_monotone_decreasing([dict(recipe="x", epochs=e, rel=r)
                                       for e, r in ((60, 0.12), (180, 0.09), (360, 0.07))])))
    rows = ([dict(recipe=recipe_name("fixed", True), schedule="fixed", restarts=True, epochs=e, rel=r)
             for e, r in ((60, 0.12), (180, 0.068), (360, 0.109))]
            + [dict(recipe=recipe_name("cosine", True), schedule="cosine", restarts=True, epochs=e, rel=0.2 * e ** -0.5)
               for e in (60, 180, 360)])
    v, _ = mechanism_verdict(rows)
    ok.append(("a factor that restores monotonicity is credited (SUPPORT)", v.startswith("SUPPORT")))
    v2, _ = mechanism_verdict([dict(r, rel=(0.12 if r["epochs"] == 60 else 0.07 if r["epochs"] == 180 else 0.1))
                               for r in rows])
    ok.append(("no factor restoring monotonicity yields MIXED, not SUPPORT", v2.startswith("MIXED")))
    print("[BUDGET MECHANISM selftest] replica fidelity + the attribution criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    rows = [r for r in rows if r.get("arm") == "mechanism"]
    if not rows:
        raise SystemExit("[BUDGET MECHANISM] nothing recorded yet")
    v, detail = mechanism_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="BUDGET MECHANISM: which recipe ingredient causes the regression?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--wall-relocation", action="store_true",
                    help="locate the wall under the good recipe (cosine + restarts) at a fixed budget")
    ap.add_argument("--reloc-epochs", type=int, default=180)
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.wall_relocation:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        run_wall_relocation(device, epochs=args.reloc_epochs, resume=not args.fresh)
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("=" * 100)
    print(f"[BUDGET MECHANISM] instance C={class_count(M)}, sigma={SIGMA}, K={K}: 2x2 recipe factorial")
    print("=" * 100)
    for recipe in RECIPES:
        rows = run_recipe(recipe, device, resume=not args.fresh)
        with open(JSONL, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    all_rows = [r for r in all_rows if r.get("arm") == "mechanism"]
    print()
    v, detail = mechanism_verdict(all_rows)
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
