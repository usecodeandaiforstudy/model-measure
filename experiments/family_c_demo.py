"""FAMILY C: a third synthetic family -- high-dimensional Gaussian mixtures (d_in = 16).

Why a third family, and why this one. CROSS-TASK closed P-A1-2 as "a common scale exists inside a family
(ordinal), and does not extend across families", on two families: the 3-D lattice (very regular) and random
centres in 3-D (irregular geometry). Both have a **3-dimensional input**, which the project flagged early as a
possible confound: with only three input dimensions, "how many regions can K codes carve out" is constrained by
the input geometry rather than by the label count. This family removes exactly that property: C centres drawn
i.i.d. from N(0, spread^2 I_16) in **16 dimensions**, equal weights, isotropic noise. The likelihood, the Bayes
error and the true I(X;Y) remain computable by the same Monte Carlo, and the exact scale twin (scale every
centre and the noise together) still exists in any dimension.

Pre-registration (2026-10-05, before the arm below was run)
--------------------------------------------------------------------------------------------
(0) CONFIGURATION RULE, fixed before any reading was measured: choose (sigma, spread) so that the family's
    ground-truth I(X;Y) falls inside **3.3-5.4 nats**, the range spanned by the lattice and family-B points --
    otherwise the cross-family comparison has no matched pair to compare. The step-0 table is printed for all
    candidate configurations and the choice is made from it (it is a measurement, not a reading).
(1) Configurations measured by training: 3 of them, C = 216, one of which is the EXACT scale twin of another
    (spread and sigma both scaled by 2). Protocol identical to families A and B, so the readings are
    comparable: reference = the family's own MLP with VQ(2048), 60 epochs, 150 samples/class, seed 0, students
    K in {32, 128, 512, 2048} under the cached-target distillation objective.
(2) Criteria:
      Z1 INVARIANCE  on the exact twin pair: |dR_IB| <= 0.30 nats while |dI(X;Y)| <= 0.30;
      Z2 ORDERING    0 material inversions between the true I(X;Y) and the reading over the family's points;
      Z3 CROSS-FAMILY at matched information (|dI| <= 0.30 nats) compare this family's under-reading with the
                     lattice's and family B's: report both gaps. A third family-specific offset confirms the
                     "within-family only" statement on independent geometry; agreement with one of the two
                     would be a partial counterexample and would be reported as such.
    Verdict: SUPPORT (the ordinal claim extends to a third family) if Z1 and Z2 and the offsets agree within
    0.30 nats; MIXED if Z1/Z2 hold but the offset is family-specific; REFUTED if Z1 or Z2 fails.
(3) The architecture is the family's own (Linear(d_in -> 64) -> ReLU -> VQ(64, K) -> Linear(64 -> C)), i.e. the
    same shape as everywhere else with the input width adapted; nothing else is tuned.
Iron rules: criteria before data; the configuration rule above is applied from the zero-training table, not
from the readings; English-only output; relay mode + --verdict-only.
--------------------------------------------------------------------------------------------
"""

import argparse
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
from capacity_lens import (VQBottleneck, task_info_fano, DistillCapacity, train_one_epoch, evaluate,
                           measured_wall, looseness)
from scale_c_demo import rank_correlation
from scale_c_clean_demo import train_one_epoch_cached, teacher_logits

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(_DATA, "family_c")
JSONL = os.path.join(_DATA, "family_c.jsonl")
REPORT = os.path.join(_DATA, "family_c_report.txt")
D_IN = 16
N_CLASSES = 216
N_PER_CLASS = 150
EPOCHS = 60
LR = 1e-2
SEED = 0
K_TEACHER = 2048
K_GRID = [32, 128, 512, 2048]
N_MC = 200_000
TARGET_BAND = (3.3, 5.4)

# The configuration rule (fixed before any reading) is applied from the step-0 table: keep (sigma, spread)
# pairs whose ground-truth I(X;Y) lies in [3.3, 5.4] nats. From the widened table:
#   C1 (sigma 1.0, spread 1.5): I(X;Y)=5.243, K_min(bayes)=144
#   C2 (sigma 2.0, spread 3.0): the EXACT scale twin of C1 (ground truth identical by construction)
#   C3 (sigma 2.5, spread 3.0): I(X;Y)=4.884, K_min(bayes)=64 -- an ordering point away from the ceiling
CONFIGS = [
    dict(tag="C1", sigma=1.0, spread=1.5),
    dict(tag="C2", sigma=2.0, spread=3.0),      # the exact scale twin of C1
    dict(tag="C3", sigma=2.5, spread=3.0),
]


class HiDimMLP(nn.Module):
    """The family's own reference/student: Linear(d_in -> 64) -> ReLU -> VQ(64, K) -> Linear(64 -> C)."""

    def __init__(self, d_in, n_classes, k_codes, d=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.fc2 = nn.Linear(d, n_classes)

    def forward(self, x):
        z, idx = self.vq(F.relu(self.fc1(x)))
        return self.fc2(z), [idx]


def random_centers(n_classes=N_CLASSES, d_in=D_IN, spread=3.0, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_classes, d_in, generator=g) * spread


def make_mixture(centers, sigma, n_per_class=N_PER_CLASS, seed=SEED):
    g = torch.Generator().manual_seed(seed + 1)
    c = centers.shape[0]
    y = torch.arange(c).repeat_interleave(n_per_class)
    x = centers[y] + sigma * torch.randn(len(y), centers.shape[1], generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


@torch.no_grad()
def mixture_stats(centers, sigma, n_mc=N_MC, seed=0, chunk=10_000, device="cpu"):
    """Monte Carlo over the known mixture (any input dimension): Bayes error, H(Y|X), I(X;Y), Bayes requirement."""
    centers = centers.to(device)
    c = centers.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    err = h_cond = 0.0
    done = 0
    while done < n_mc:
        n = min(chunk, n_mc - done)
        y = torch.randint(0, c, (n,), generator=g)
        x = (centers.cpu()[y] + sigma * torch.randn(n, centers.shape[1], generator=g)).to(device)
        log_post = torch.log_softmax(-(torch.cdist(x, centers) ** 2) / (2.0 * sigma ** 2), dim=1)
        err += (log_post.argmax(dim=1).cpu() != y).sum().item()
        h_cond += -log_post[torch.arange(n), y].sum().item()
        done += n
    eps_bayes = err / n_mc
    mi = math.log(c) - h_cond / n_mc
    rib = task_info_fano(eps_bayes, c)
    return dict(c=c, sigma=sigma, eps_bayes=eps_bayes, mi=mi, rib_bayes=rib,
                k_min_bayes=DistillCapacity().min_codebook_size(rib), n_mc=n_mc, d_in=centers.shape[1])


def geometry_note(centers):
    d = torch.cdist(centers, centers)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return dict(nn_min=float(nn.min()), nn_median=float(nn.median()), nn_max=float(nn.max()),
                spread=float(centers.std()))


def step0(device, sigmas=(0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 6.0), spreads=(1.5, 3.0)):
    """The configuration table: the zero-training ground truth over candidate (sigma, spread) pairs.

    Recorded finding from the first pass: with the original candidate grid (sigma <= 1.0) *every* configuration
    sat at the ceiling (I(X;Y) = ln 216 = 5.375, Bayes error ~ 0). That is distance concentration in 16
    dimensions -- random centres separate like sqrt(2*d)*spread while the noise stays sigma -- and it is itself
    the structural difference this family was chosen for. The grid was therefore widened to sigma up to 6.0
    BEFORE any reading was measured; the pre-registered band rule was not changed.
    """
    centers_cache = {s: random_centers(spread=s) for s in spreads}
    print("=" * 100)
    print(f"[FAMILY C] configuration table: input dim {D_IN}, C={N_CLASSES}, candidate (sigma, spread)")
    print("=" * 100)
    print(f"  the pre-registered rule is to keep configurations whose I(X;Y) lies in "
          f"[{TARGET_BAND[0]}, {TARGET_BAND[1]}] nats, the band spanned by families A and B\n")
    print(f"  {'sigma':>6} {'spread':>7} {'eps_bayes':>10} {'I(X;Y)':>8} {'rib_bayes':>10} "
          f"{'K_min_bayes':>12}  in band?")
    rows = []
    for spread in spreads:
        for sigma in sigmas:
            st = mixture_stats(centers_cache[spread], sigma, device=device)
            in_band = TARGET_BAND[0] <= st["mi"] <= TARGET_BAND[1]
            g = geometry_note(centers_cache[spread])
            rec = dict(st, step0=True, spread=spread, in_band=in_band)
            rec["geometry_spread"] = g.pop("spread")      # the geometry note reports its own spread
            rec.update(g)
            rows.append(rec)
            print(f"  {sigma:>6.2f} {spread:>7.1f} {st['eps_bayes']:>10.4f} {st['mi']:>8.3f} "
                  f"{st['rib_bayes']:>10.3f} {st['k_min_bayes']:>12}  {'YES' if in_band else 'no'}")
    with open(JSONL, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def run_config(cfg, device, resume=True):
    centers = random_centers(spread=cfg["spread"])
    stats = mixture_stats(centers, cfg["sigma"], device=device)
    cache = os.path.join(ROOT, cfg["tag"])
    os.makedirs(cache, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_mixture(centers, cfg["sigma"])
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    t0 = time.time()
    path_t = os.path.join(cache, "teacher.pt")
    if resume and os.path.exists(path_t):
        acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    else:
        torch.manual_seed(70000 + N_CLASSES)
        ref = HiDimMLP(D_IN, N_CLASSES, K_TEACHER).to(device)
        opt = torch.optim.Adam(ref.parameters(), lr=LR)
        for _ in range(EPOCHS):
            train_one_epoch(ref, tr, opt, device=device)
        acc_t, _ = evaluate(ref, te, device=device)
        torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
    ref = HiDimMLP(D_IN, N_CLASSES, K_TEACHER).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    path_c = os.path.join(cache, "targets.pt")
    if resume and os.path.exists(path_c):
        loaded = torch.load(path_c, map_location="cpu", weights_only=True)
        targets = (loaded["tr"] if isinstance(loaded, dict) else loaded).float()
    else:
        targets = teacher_logits(ref, tr, device).float()
        torch.save(targets, path_c)
    accs = {}
    for k in K_GRID:
        path_s = os.path.join(cache, f"student_K{k}.pt")
        if resume and os.path.exists(path_s):
            accs[k] = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
            continue
        torch.manual_seed(71000 + N_CLASSES + k)
        stu = HiDimMLP(D_IN, N_CLASSES, k).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=LR)
        loader = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
        for _ in range(EPOCHS):
            train_one_epoch_cached(stu, loader, opt, device=device)
        acc, _ = evaluate(stu, te, device=device)
        torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
        accs[k] = acc
    eps = 1.0 - acc_t
    rib = task_info_fano(eps, N_CLASSES)
    rel = {k: (acc_t - a) / max(acc_t, 1e-12) for k, a in accs.items()}
    k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
    row = dict(arm="family_c", tag=cfg["tag"], c=N_CLASSES, d_in=D_IN, sigma=cfg["sigma"],
               spread=cfg["spread"], acc_t=acc_t, eps=eps, rib=rib,
               k_min=DistillCapacity().min_codebook_size(rib), accs=accs, rel=rel, k_wall=k_wall,
               looseness=looseness(rib, k_wall), eps_bayes=stats["eps_bayes"], mi_true=stats["mi"],
               rib_bayes=stats["rib_bayes"], k_min_bayes=stats["k_min_bayes"],
               under_read_rib=stats["rib_bayes"] - rib, geometry=geometry_note(centers), t=time.time() - t0)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [{cfg['tag']}] d_in={D_IN} C={N_CLASSES} sigma={cfg['sigma']} spread={cfg['spread']}: "
          f"ref acc={acc_t:.3f} (eps={eps:.3f} vs Bayes {stats['eps_bayes']:.3f}) R_IB={rib:.2f} "
          f"(true I={stats['mi']:.2f}) K_min={row['k_min']} (Bayes {stats['k_min_bayes']}) -> "
          f"K_wall={k_wall}  [{row['t']:.0f}s]")
    print("      rel gaps: " + "  ".join(f"K{k}:{rel[k]:.2f}" for k in sorted(accs)))
    return row


def other_families():
    """The under-readings of families A (lattice) and B (random centres, 3-D) at their measured points."""
    out = []
    ct = os.path.join(_DATA, "cross_task.jsonl")
    ws = os.path.join(_DATA, "wall_sat.jsonl")
    truth = {}
    if os.path.exists(ct):
        for line in open(ct, encoding="utf-8"):
            r = json.loads(line)
            if r.get("step0") and abs(r.get("spacing", 3.0) - 3.0) < 1e-9 and r["m"] == 6:
                truth[(r["m"], round(r["sigma"], 3))] = r
    if os.path.exists(ws):
        for line in open(ws, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") == "sigma" and r.get("m") == 6 and "rib_bayes" in r:
                out.append(dict(family="A (3-D lattice)", mi=r["mi_true"], under=r["rib_bayes"] - r["rib"],
                                tag=f"lattice sigma={r['sigma']:.2f}"))
    fb = os.path.join(_DATA, "family_b.jsonl")
    if os.path.exists(fb):
        for line in open(fb, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") == "family_b":
                out.append(dict(family="B (3-D random centres)", mi=r["mi_true"],
                                under=r["rib_bayes"] - r["rib"], tag=r["tag"]))
    return out


def family_verdict(rows, others=None):
    detail = []
    by_tag = {r["tag"]: r for r in rows}
    z1 = False
    if "C1" in by_tag and "C2" in by_tag:
        d_rib = abs(by_tag["C1"]["rib"] - by_tag["C2"]["rib"])
        d_true = abs(by_tag["C1"]["mi_true"] - by_tag["C2"]["mi_true"])
        z1 = d_rib <= 0.30 and d_true <= 0.30
        detail.append(f"  Z1 invariance on the exact scale twin (spread and sigma both x2): "
                      f"|dR_IB|={d_rib:.3f} nats, |dI(X;Y)|={d_true:.4f} -> {'PASS' if z1 else 'FAIL'}")
    else:
        detail.append("  Z1 not evaluable (C1 or C2 missing)")
    pts = [r for r in rows if r.get("mi_true") is not None]
    inversions = []
    for i in pts:
        for j in pts:
            if i["mi_true"] > j["mi_true"] + 0.05 and i["rib"] < j["rib"] - 1e-9:
                inversions.append((i["tag"], j["tag"]))
    z2 = len(pts) >= 3 and not inversions
    detail.append(f"  Z2 ordering inside family C: {len(inversions)} material inversions over {len(pts)} points "
                  f"(Spearman={rank_correlation([r['mi_true'] for r in pts], [r['rib'] for r in pts]):+.3f}, "
                  f"descriptive) -> {'PASS' if z2 else 'FAIL'}")
    for r in pts:
        detail.append(f"       {r['tag']}: true I(X;Y)={r['mi_true']:.3f}, reading={r['rib']:.3f}, "
                      f"under-reading={r['rib_bayes'] - r['rib']:.3f} nats")
    z3 = None
    if others:
        table = []
        for r in pts:
            near = min(others, key=lambda o: abs(o["mi"] - r["mi_true"]))
            if abs(near["mi"] - r["mi_true"]) <= 0.30:
                table.append((r["tag"], r["mi_true"], r["rib_bayes"] - r["rib"], near))
        if table:
            detail.append("  Z3 cross-family at matched information (|dI| <= 0.30):")
            worst = 0.0
            for tag, mi, under, near in table:
                gap = abs(under - near["under"])
                worst = max(worst, gap)
                detail.append(f"       {tag} (I={mi:.3f}, under-reading {under:.3f}) vs {near['family']} "
                              f"{near['tag']} (I={near['mi']:.3f}, {near['under']:.3f}) -> gap {gap:.3f} nats")
            z3 = worst <= 0.30
            detail.append(f"       largest cross-family gap: {worst:.3f} nats -> "
                          f"{'agrees (ordinal claim extends)' if z3 else 'family-specific (within-family only)'}")
        else:
            detail.append("  Z3 not evaluable (no matched-information point in the other families)")
    if z1 and z2 and z3:
        v = "SUPPORT: the ordinal claim extends to a third family and the offsets agree"
    elif z1 and z2:
        v = ("MIXED: invariance and ordering hold on the third family, but the offset is family-specific -- "
             "the 'within-family only' statement now rests on three structurally different families")
    else:
        v = "REFUTED: the third family breaks invariance or ordering"
    return v, detail


def run_selftest():
    ok = []
    centers = random_centers(n_classes=8, spread=3.0)
    ok.append((f"centre draw is reproducible and {D_IN}-dimensional", centers.shape == (8, D_IN)))
    a = mixture_stats(centers, 0.5, n_mc=8000)
    b = mixture_stats(centers, 2.0, n_mc=8000)
    ok.append((f"noise raises the Bayes error ({a['eps_bayes']:.3f} -> {b['eps_bayes']:.3f})",
               b["eps_bayes"] > a["eps_bayes"]))
    scaled = mixture_stats(centers * 2.0, 1.0, n_mc=8000)
    ok.append((f"the scale twin is exact in {D_IN} dimensions ({a['mi']:.3f} vs {scaled['mi']:.3f})",
               abs(a["mi"] - scaled["mi"]) < 0.05))
    (x_tr, y_tr), _ = make_mixture(centers, 0.5, n_per_class=4)
    ok.append((f"the mixture splits 80/20 with all classes ({len(torch.unique(y_tr))} classes)",
               len(torch.unique(y_tr)) == 8 and x_tr.shape[1] == D_IN))
    rows = [dict(tag="C1", rib=2.5, mi_true=4.5, rib_bayes=3.2),
            dict(tag="C2", rib=2.45, mi_true=4.5, rib_bayes=3.2),
            dict(tag="C3", rib=3.5, mi_true=5.2, rib_bayes=4.0)]
    v, _ = family_verdict(rows, others=[dict(family="A (3-D lattice)", mi=4.5, under=0.70, tag="lattice")])
    ok.append(("invariance + ordering with an agreeing offset -> SUPPORT", v.startswith("SUPPORT")))
    v2, _ = family_verdict(rows, others=[dict(family="A (3-D lattice)", mi=4.5, under=0.10, tag="lattice")])
    ok.append(("a family-specific offset -> MIXED, not SUPPORT", v2.startswith("MIXED")))
    print("[FAMILY C selftest] ground truth + criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    rows = [r for r in rows if r.get("arm") == "family_c"]
    if not rows:
        raise SystemExit("[FAMILY C] nothing recorded yet")
    v, detail = family_verdict(rows, other_families())
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="FAMILY C: high-dimensional (16-D) Gaussian mixtures for P-A1-2")
    ap.add_argument("--step0", action="store_true", help="configuration table only (zero training)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if not args.step0:
        for cfg in CONFIGS:
            run_config(cfg, device, resume=not args.fresh)
    else:
        step0(device)
        return
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    rows = [r for r in rows if r.get("arm") == "family_c"]
    print()
    v, detail = family_verdict(rows, other_families())
    for line in detail:
        print(line)
    print(f"  [verdict] {v}")
    with open(REPORT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + v + "\n")
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
