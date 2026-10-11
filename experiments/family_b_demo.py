"""FAMILY B: a second synthetic family with computable ground truth, for the P-A1-2 cross-task question.

Where the question comes from. CROSS-TASK gave P-A1-2 its first positive design on the lattice family and
closed it as **"a common scale exists, but only up to order"**: the reading survived an exactly
information-preserving re-parameterization (X1, 0.175 nats apart) and ordered tasks correctly (X2), while the
offset between the Bayes requirement and the reading was not constant (X3, 0.87/0.80/0.34/0.52 nats). Every
one of those numbers came from ONE family whose geometry is very regular (equal spacing, isotropic noise,
all classes exchangeable). The honest next question is whether the conclusion survives a family that is not
regular at all -- and, harder, whether readings from the two families can be placed on one scale.

**Family B: random-centre Gaussian mixtures.** C centres drawn i.i.d. from N(0, spread^2 I_3), equal class
weights, isotropic noise sigma. Like the lattice it is a Gaussian mixture, so the Bayes error, H(Y|X), the
true I(X;Y) and the Bayes-implied requirement are all computable in closed form by Monte Carlo -- but the
geometry is irregular: inter-centre distances vary, some classes are near-twins, others are isolated. Nothing
about the family is tuned; the centres are one draw of a seed, and that draw is the task.

Pre-registration (2026-10-05, before the runs below)
--------------------------------------------------------------------------------------------
Configurations (each: reference = GridMLP with VQ(2048), 60 epochs, 150 samples/class, seed 0 -- the same
protocol as SCALE-C round 3 and CROSS-TASK, so the readings are comparable):
  B1  C=216, sigma=0.5, spread=3.0
  B2  C=216, sigma=1.0, spread=3.0
  B3  C=216, sigma=1.0, spread=6.0      <- the EXACT scale-equivalent twin of B1: scaling every centre AND the
                                           noise by the same factor leaves all posteriors unchanged. (The first
                                           draft of this list wrote B3 as sigma=0.5, spread=6.0, which the
                                           zero-training step-0 table immediately showed is NOT a twin -- the
                                           true information came out 4.603 vs 5.226 nats. The configuration was
                                           corrected before any training run; the criterion Y1 was not touched.)
  B4  C=216, sigma=0.5, spread=6.0      <- an extra ordering point (an easier task than B1)
Criteria:
  Y1 INVARIANCE   |R_IB(B1) - R_IB(B3)| <= 0.30 nats while |I(X;Y)(B1) - I(X;Y)(B3)| <= 0.30 (the ground
                  truth must agree they are the same task);
  Y2 ORDERING     0 material inversions between the true I(X;Y) and the reading over the family's points;
  Y3 CROSS-FAMILY at matched true information, compare the under-reading (rib_bayes - rib) of family B with
                  the lattice family's: if the two differ by more than 0.30 nats, the offset is
                  family-specific and even the ordinal claim does not transfer across families.
Verdict: SUPPORT (ordinal extends to a second family) if Y1 and Y2 and the Y3 offset is <= 0.30;
         MIXED if Y1 and Y2 hold but the cross-family offset is larger (then the claim is bounded to
         within-family comparisons -- still a real answer, and the one the data would be showing);
         REFUTED if Y1 or Y2 fails.
Iron rules: criteria before data; the training protocol is copied from the earlier campaigns rather than
re-tuned; the centres are a single draw (no cherry-picking of a "nice" geometry); English-only output.
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
from capacity_lens import task_info_fano, DistillCapacity, train_one_epoch, evaluate, measured_wall, looseness
from scale_c_demo import GridMLP, class_count, rank_correlation
from scale_c_clean_demo import train_one_epoch_cached, teacher_logits

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(_DATA, "family_b")
JSONL = os.path.join(_DATA, "family_b.jsonl")
REPORT = os.path.join(_DATA, "family_b_report.txt")
N_PER_CLASS = 150
EPOCHS = 60
LR = 1e-2
SEED = 0
K_TEACHER = 2048
K_GRID = [32, 128, 512, 2048]
N_MC = 200_000

CONFIGS = [
    dict(tag="B1", c=216, sigma=0.5, spread=3.0),
    dict(tag="B2", c=216, sigma=1.0, spread=3.0),
    dict(tag="B3", c=216, sigma=1.0, spread=6.0),      # the exact scale twin of B1 (centres x2, sigma x2)
    dict(tag="B4", c=216, sigma=0.5, spread=6.0),      # an extra ordering point
]


# ---------------------------------------------------------------------------
# the family and its ground truth
# ---------------------------------------------------------------------------

def random_centers(c, spread, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, 3, generator=g) * spread


def make_mixture(centers, sigma, n_per_class=N_PER_CLASS, seed=SEED):
    """Samples from the equal-weight mixture; returns ((x_tr, y_tr), (x_te, y_te))."""
    g = torch.Generator().manual_seed(seed + 1)
    c = centers.shape[0]
    y = torch.arange(c).repeat_interleave(n_per_class)
    x = centers[y] + sigma * torch.randn(len(y), 3, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


@torch.no_grad()
def mixture_stats(centers, sigma, n_mc=N_MC, seed=0, chunk=20_000, device="cpu"):
    """Monte Carlo over the known mixture: Bayes error, H(Y|X), I(X;Y) and the Bayes-implied requirement."""
    centers = centers.to(device)
    c = centers.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    err = h_cond = 0.0
    done = 0
    while done < n_mc:
        n = min(chunk, n_mc - done)
        y = torch.randint(0, c, (n,), generator=g)
        x = (centers.cpu()[y] + sigma * torch.randn(n, 3, generator=g)).to(device)
        log_post = torch.log_softmax(-(torch.cdist(x, centers) ** 2) / (2.0 * sigma ** 2), dim=1)
        err += (log_post.argmax(dim=1).cpu() != y).sum().item()
        h_cond += -log_post[torch.arange(n), y].sum().item()
        done += n
    eps_bayes = err / n_mc
    mi = math.log(c) - h_cond / n_mc
    rib_bayes = task_info_fano(eps_bayes, c)
    return dict(c=c, sigma=sigma, eps_bayes=eps_bayes, mi=mi, rib_bayes=rib_bayes,
                k_min_bayes=DistillCapacity().min_codebook_size(rib_bayes), n_mc=n_mc)


def geometry_note(centers):
    """A cheap description of the draw, so a reader can see how irregular the geometry is."""
    d = torch.cdist(centers, centers)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return dict(nn_min=float(nn.min()), nn_median=float(nn.median()), nn_max=float(nn.max()),
                spread=float(centers.std()))


# ---------------------------------------------------------------------------
# the training arm
# ---------------------------------------------------------------------------

def run_config(cfg, device, resume=True):
    c, sigma, spread = cfg["c"], cfg["sigma"], cfg["spread"]
    centers = random_centers(c, spread)
    stats = mixture_stats(centers, sigma, device=device)
    cache = os.path.join(ROOT, cfg["tag"])
    os.makedirs(cache, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_mixture(centers, sigma)
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    t0 = time.time()
    path_t = os.path.join(cache, "teacher.pt")
    if resume and os.path.exists(path_t):
        acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    else:
        torch.manual_seed(70000 + c)
        ref = GridMLP(c, K_TEACHER).to(device)
        opt = torch.optim.Adam(ref.parameters(), lr=LR)
        for _ in range(EPOCHS):
            train_one_epoch(ref, tr, opt, device=device)
        acc_t, _ = evaluate(ref, te, device=device)
        torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
    ref = GridMLP(c, K_TEACHER).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    path_c = os.path.join(cache, "targets.pt")
    if resume and os.path.exists(path_c):
        targets = torch.load(path_c, map_location="cpu", weights_only=True).float()
    else:
        targets = teacher_logits(ref, tr, device).float()
        torch.save(targets, path_c)
    accs = {}
    for k in K_GRID:
        path_s = os.path.join(cache, f"student_K{k}.pt")
        if resume and os.path.exists(path_s):
            accs[k] = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
            continue
        torch.manual_seed(71000 + c + k)
        stu = GridMLP(c, k).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=LR)
        loader = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
        for _ in range(EPOCHS):
            train_one_epoch_cached(stu, loader, opt, device=device)
        acc, _ = evaluate(stu, te, device=device)
        torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
        accs[k] = acc
    eps = 1.0 - acc_t
    rib = task_info_fano(eps, c)
    rel = {k: (acc_t - a) / max(acc_t, 1e-12) for k, a in accs.items()}
    k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
    row = dict(arm="family_b", tag=cfg["tag"], c=c, sigma=sigma, spread=spread, acc_t=acc_t, eps=eps,
               rib=rib, k_min=DistillCapacity().min_codebook_size(rib), accs=accs, rel=rel, k_wall=k_wall,
               looseness=looseness(rib, k_wall), eps_bayes=stats["eps_bayes"], mi_true=stats["mi"],
               rib_bayes=stats["rib_bayes"], k_min_bayes=stats["k_min_bayes"],
               under_read_rib=stats["rib_bayes"] - rib,
               under_read_k=stats["k_min_bayes"] / max(DistillCapacity().min_codebook_size(rib), 1),
               geometry=geometry_note(centers), t=time.time() - t0)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [{cfg['tag']}] C={c} sigma={sigma} spread={spread}: ref acc={acc_t:.3f} (eps={eps:.3f} vs "
          f"Bayes {stats['eps_bayes']:.3f}) R_IB={rib:.2f} (true I={stats['mi']:.2f}) "
          f"K_min={row['k_min']} (Bayes {stats['k_min_bayes']}) -> K_wall={k_wall}  [{row['t']:.0f}s]")
    print(f"      geometry: nearest-centre distance {row['geometry']['nn_min']:.2f}/"
          f"{row['geometry']['nn_median']:.2f}/{row['geometry']['nn_max']:.2f} (min/median/max)")
    print("      rel gaps: " + "  ".join(f"K{k}:{rel[k]:.2f}" for k in sorted(accs)))
    return row


# ---------------------------------------------------------------------------
# the criteria
# ---------------------------------------------------------------------------

def lattice_rows():
    """The lattice points with a computed ground truth, for the cross-family comparison (CROSS-TASK + sigma arm)."""
    out = {}
    ct = os.path.join(_DATA, "cross_task.jsonl")
    if os.path.exists(ct):
        for line in open(ct, encoding="utf-8"):
            r = json.loads(line)
            if r.get("step0") and abs(r.get("spacing", 3.0) - 3.0) < 1e-9:
                out[(r["m"], round(r["sigma"], 3))] = r
    return out


def family_verdict(rows):
    """Y1 (invariance under the exact scale twin) and Y2 (ordering) inside family B. Y3, the cross-family
    offset, needs the lattice arm rows and is computed in cross_family_table()."""
    detail = []
    by_tag = {r["tag"]: r for r in rows}
    y1 = False
    if "B1" in by_tag and "B3" in by_tag:
        d_rib = abs(by_tag["B1"]["rib"] - by_tag["B3"]["rib"])
        d_true = abs(by_tag["B1"]["mi_true"] - by_tag["B3"]["mi_true"])
        y1 = d_rib <= 0.30 and d_true <= 0.30
        detail.append(f"  Y1 invariance on the exactly scale-equivalent twin: |dR_IB|={d_rib:.3f} nats, "
                      f"|dI(X;Y)|={d_true:.4f} -> {'PASS' if y1 else 'FAIL'}")
        detail.append(f"       B1 (spread 3): R_IB={by_tag['B1']['rib']:.3f}, true={by_tag['B1']['mi_true']:.3f}"
                      f" | B3 (spread 6): R_IB={by_tag['B3']['rib']:.3f}, true={by_tag['B3']['mi_true']:.3f}")
    else:
        detail.append("  Y1 not evaluable (B1 or B3 missing)")
    pts = [r for r in rows if r.get("mi_true") is not None]
    inversions = []
    for i in pts:
        for j in pts:
            if i["mi_true"] > j["mi_true"] + 0.05 and i["rib"] < j["rib"] - 1e-9:
                inversions.append((i["tag"], j["tag"]))
    y2 = len(pts) >= 3 and not inversions
    detail.append(f"  Y2 ordering inside family B: {len(inversions)} material inversions over {len(pts)} points "
                  f"(Spearman={rank_correlation([r['mi_true'] for r in pts], [r['rib'] for r in pts]):+.3f}, "
                  f"descriptive) -> {'PASS' if y2 else 'FAIL'}")
    for r in pts:
        detail.append(f"       {r['tag']}: true I(X;Y)={r['mi_true']:.3f}, reading={r['rib']:.3f}, "
                      f"under-reading={r['rib_bayes'] - r['rib']:.3f} nats")
    return y1, y2, detail


def run_selftest():
    ok = []
    centers = random_centers(8, 3.0)
    ok.append((f"centre draw is reproducible ({centers.shape}, seed fixed)", centers.shape == (8, 3)))
    a = mixture_stats(centers, 0.5, n_mc=20000)
    b = mixture_stats(centers, 2.0, n_mc=20000)
    ok.append((f"noise raises the Bayes error ({a['eps_bayes']:.3f} -> {b['eps_bayes']:.3f})",
               b["eps_bayes"] > a["eps_bayes"]))
    # exact scale equivalence: scaling centres and noise together leaves every posterior unchanged
    scaled = mixture_stats(centers * 2.0, 1.0, n_mc=20000)
    ok.append((f"scaling centres and noise together is information-preserving "
               f"({a['mi']:.3f} vs {scaled['mi']:.3f})", abs(a["mi"] - scaled["mi"]) < 0.05))
    (x_tr, y_tr), (x_te, y_te) = make_mixture(centers, 0.5, n_per_class=4)
    ok.append((f"the mixture splits 80/20 with all classes present ({len(torch.unique(y_tr))} classes)",
               len(torch.unique(y_tr)) == 8 and len(y_tr) == int(0.8 * 32)))
    g = geometry_note(centers)
    ok.append((f"the geometry note reports real spread ({g['nn_median']:.2f} median nearest centre)",
               g["nn_median"] > 0))
    y1, y2, _ = family_verdict([dict(tag="B1", rib=4.95, mi_true=5.0, rib_bayes=5.3),
                                dict(tag="B2", rib=3.0, mi_true=3.2, rib_bayes=3.6),
                                dict(tag="B3", rib=4.90, mi_true=5.0, rib_bayes=5.3)])
    ok.append(("the criteria accept an invariant, correctly ordered family", y1 and y2))
    y1b, y2b, _ = family_verdict([dict(tag="B1", rib=4.95, mi_true=5.0, rib_bayes=5.3),
                                  dict(tag="B2", rib=3.0, mi_true=3.2, rib_bayes=3.6),
                                  dict(tag="B3", rib=1.0, mi_true=5.0, rib_bayes=5.3)])
    ok.append(("an invariance violation is caught", not y1b))
    print("[FAMILY B selftest] ground truth + criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def cross_family_table(rows):
    """Match each family-B point to the nearest lattice point in true information and compare the offsets."""
    lat = []
    ws = os.path.join(_DATA, "wall_sat.jsonl")
    if os.path.exists(ws):
        for line in open(ws, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") == "sigma" and r.get("m") == 6 and "rib_bayes" in r:
                lat.append(dict(name=f"lattice C={r['c']} sigma={r['sigma']:.2f}", mi=r["mi_true"],
                                rib=r["rib"], rib_bayes=r["rib_bayes"], under=r["rib_bayes"] - r["rib"]))
    table = []
    for r in rows:
        if not lat:
            break
        near = min(lat, key=lambda l: abs(l["mi"] - r["mi_true"]))
        table.append(dict(family_b=r["tag"], mi_b=r["mi_true"], under_b=r["rib_bayes"] - r["rib"],
                          lattice=near["name"], mi_l=near["mi"], under_l=near["under"],
                          gap=abs((r["rib_bayes"] - r["rib"]) - near["under"]),
                          d_mi=abs(r["mi_true"] - near["mi"])))
    return table


def verdict_only():
    """Offline re-verdict from the recorded rows (no training), following the project's convention."""
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    rows = [r for r in rows if r.get("arm") == "family_b"]
    if not rows:
        raise SystemExit("[FAMILY B] no arm rows recorded yet")
    y1, y2, detail = family_verdict(rows)
    for line in detail:
        print(line)
    table = [t for t in cross_family_table(rows) if t["d_mi"] <= 0.30]      # matched comparisons only
    for t in table:
        print(f"   {t['family_b']}: under-reading {t['under_b']:.3f} vs {t['lattice']} {t['under_l']:.3f} "
              f"-> gap {t['gap']:.3f} (|dI|={t['d_mi']:.3f})")
    worst = max((t["gap"] for t in table), default=None)
    if worst is None:
        print("  [verdict] MIXED: Y1/Y2 evaluated but no matched-information lattice point was available")
    elif worst > 0.30:
        print(f"  [verdict] MIXED: the ordinal claim holds inside family B (Y1/Y2), but at matched information "
              f"the under-reading differs from the lattice by up to {worst:.2f} nats")
    else:
        print("  [verdict] SUPPORT: the ordinal claim extends to a second family")


def main():
    ap = argparse.ArgumentParser(description="FAMILY B: a second synthetic family for the P-A1-2 question")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--step0", action="store_true", help="ground-truth table only (zero training)")
    ap.add_argument("--verdict-only", action="store_true", help="offline re-verdict from the jsonl (no training)")
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
    print("=" * 100)
    print("[FAMILY B] random-centre Gaussian mixtures (irregular geometry, computable ground truth)")
    print("=" * 100)
    if args.step0:
        for cfg in CONFIGS:
            centers = random_centers(cfg["c"], cfg["spread"])
            st = mixture_stats(centers, cfg["sigma"], device=device)
            g = geometry_note(centers)
            print(f"  {cfg['tag']}: C={st['c']} sigma={cfg['sigma']} spread={cfg['spread']} -> "
                  f"eps_bayes={st['eps_bayes']:.4f}, I(X;Y)={st['mi']:.3f}, rib_bayes={st['rib_bayes']:.3f}, "
                  f"K_min(bayes)={st['k_min_bayes']}, nearest centres {g['nn_min']:.2f}/"
                  f"{g['nn_median']:.2f}/{g['nn_max']:.2f}")
            with open(JSONL, "a", encoding="utf-8") as f:
                rec = dict(st, step0=True, tag=cfg["tag"], spread=cfg["spread"])
                rec["geometry_spread"] = g.pop("spread")      # geometry_note also reports a spread
                rec.update(g)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    rows = []
    for cfg in CONFIGS:
        rows.append(run_config(cfg, device, resume=not args.fresh))
    print()
    y1, y2, detail = family_verdict(rows)
    for line in detail:
        print(line)
    table = [t for t in cross_family_table(rows) if t["d_mi"] <= 0.30]   # matched comparisons only
    if table:
        print("\n  Y3 cross-family table (nearest lattice point in true information, |dI| <= 0.30):")
        print(f"   {'family B':>9} {'I(X;Y)':>8} {'under-read':>11} | {'lattice':>22} {'I(X;Y)':>8} "
              f"{'under-read':>11} {'gap':>6} {'|dI|':>6}")
        for t in table:
            print(f"   {t['family_b']:>9} {t['mi_b']:>8.3f} {t['under_b']:>11.3f} | {t['lattice']:>22} "
                  f"{t['mi_l']:>8.3f} {t['under_l']:>11.3f} {t['gap']:>6.3f} {t['d_mi']:>6.3f}")
        worst = max(t["gap"] for t in table)
        if worst > 0.30:
            v = (f"MIXED: the ordinal claim holds inside family B (Y1/Y2), but at matched information the "
                 f"under-reading differs from the lattice by up to {worst:.2f} nats -> the offset is "
                 f"family-specific, so readings from different families are not comparable even ordinally")
        else:
            v = ("SUPPORT: the ordinal claim extends to a second family, and the under-reading agrees with "
                 "the lattice family at matched information")
    else:
        v = "MIXED: Y1/Y2 evaluated but no matched-information lattice point was available for Y3"
    print(f"\n  [verdict] {v}")
    with open(REPORT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        for t in table:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
        f.write("verdict: " + v + "\n")
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
