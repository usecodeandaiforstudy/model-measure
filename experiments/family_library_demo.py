"""FAMILY-LIBRARY: can a within-family calibration turn the reading from an AFTER-THE-FACT measurement
into a QUASI-A-PRIORI prediction for a new instance of the same family?

Why this is worth doing NOW (the PW experiment settled it): the reading is witness-relative and a frozen or
cheaply-pretrained witness produces an EMPTY reading -- a task-trained witness is needed (witness_threshold:
acc >= 0.577 just to exclude K=2 on CIFAR-10), and that costs minutes to hours. Amortisation is exactly
that trade: pay seconds per new instance instead of training a witness.

Honesty up front: a family library is a CALIBRATION, not a law. It is valid only (a) inside the family,
(b) for the FIXED witness protocol it was calibrated on (the reading is witness-relative, so changing the
protocol changes what is predicted), and (c) as a PLANNING prior -- never as a certified verdict.

Usage:
    python experiments/family_library_demo.py                 # 6 instances x 12 ep witness, ~8-12 min GPU
    python experiments/family_library_demo.py --quick         # 3 instances, 2 epochs (smoke)
    python experiments/family_library_demo.py --selftest      # pure-function unit tests (no training)
    python experiments/family_library_demo.py --verdict-only  # re-validate from family.jsonl (no training)

Pre-registration (2026-09-30, criteria before data)
-----------------------------------------------------------------------------------------------
(1) Family: CIFAR-10 with the label space restricted to the first C classes of a FIXED permutation
    (torch.randperm seeded 0, pre-registered) for C in {2,3,4,5,6,10}. One instance = one C. 20% train
    subset (the project's fixed subset generator, seed 0) filtered to those classes; test = the CIFAR-10
    test split filtered the same way. Instance order = increasing C.
(2) Two kinds of measurement per instance:
      MEASURED (the target to be predicted): a task-TRAINED witness = cifar_demo.SmallCNN(d=64, K=256,
        width=96), 12 epochs, Adam 1e-3 on the instance's augmented 20% train subset -> acc_t ->
        R_IB = task_info_fano(1 - acc_t, C) -> K_min = ceil(e^R_IB).
      CHEAP DESCRIPTORS (seconds, no training): acc_ridge = closed-form ridge on standardized raw pixels
        (10000 train / full test); acc_randfeat = closed-form ridge on a RANDOM-INIT backbone's pre-VQ
        representation (64-d); C and ln C.
(3) Amortised predictor (fixed before the data), two stages:
      stage 1 calibration (fitted on the training instances only): acc_pred = clip(a + b*acc_ridge, 0, 1);
      stage 2 the project's own formula: R_IB_pred = task_info_fano(1 - acc_pred, C), K_min_pred = ceil(e^R_IB_pred).
    Validation = LEAVE-ONE-OUT: the calibration is refitted without the held-out instance and then used to
    predict it, so no instance ever contributes to its own prediction.
(4) Metrics and pre-registered criteria:
      F-1 (a family law exists): Spearman(acc_ridge, acc_t) >= 0.80 across instances.
      F-2 (the prediction is usable): LOO |R_IB_pred - R_IB_measured| <= 0.30 nats for ALL instances.
      F-3 (SAFETY, the direction that matters): for every instance the predicted exclusion set is a SUBSET
          of the measured one (a predicted over-exclusion tells a user that a workable K is impossible).
      F-4 (contrast, reported): the label-only reading (R_IB = ln C) on the same safety metric -- the
          WITNESS run already showed that reading over-excludes.
    Verdict (pure function lib_verdict): SUPPORT if F-1 and F-2 and F-3; PARTIAL if two of them; else
    MIXED/REFUTED -- recorded as-is.
(5) Relay: descriptors and witness checkpoints cached under experiments/data/family/<fingerprint>/.
(6) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads.
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
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (train_one_epoch, evaluate, task_info_fano, DistillCapacity, excluded_set)
from cifar_demo import SmallCNN
from cph_demo import load_cifar
from witness_demo import cifar_raw_loaders

C_GRID = [2, 3, 4, 5, 6, 10]
D_REP, N_CODES, WIDTH = 64, 256, 96
EPOCHS = 12
TRAIN_SUBSET = 0.20
LR = 1e-3
PROBE_TRAIN = 10000
SEED_SUBSET_PERM = 0
EXCL_GRID = [2, 4, 8, 16, 32, 64, 128, 256]

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "family.jsonl")
LOG = os.path.join(_DATA, "family.log")
REPORT = os.path.join(_DATA, "family_report.txt")
CKPT_ROOT = os.path.join(_DATA, "family")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logging():
    global _LOG
    _LOG = open(LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Family construction
# ---------------------------------------------------------------------------

def class_subset(c, perm_seed=SEED_SUBSET_PERM):
    """The C classes of this instance: the first C of a fixed permutation (pre-registered)."""
    g = torch.Generator().manual_seed(perm_seed)
    return sorted(torch.randperm(10, generator=g)[:c].tolist())


def labels_of(ds):
    """Labels of a CIFAR-10 dataset or a Subset of one (fast: uses .targets, not __getitem__)."""
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if isinstance(ds, Subset):
        base = labels_of(ds.dataset)
        return [base[i] for i in ds.indices]
    raise TypeError(f"cannot read labels from {type(ds)}")


class Remapped(torch.utils.data.Dataset):
    """The samples of `classes`, with labels remapped to 0..C-1 (order = the sorted class list)."""

    def __init__(self, ds, classes):
        self.ds = ds
        self.mapping = {c: i for i, c in enumerate(classes)}
        self.keep = [i for i, y in enumerate(labels_of(ds)) if int(y) in self.mapping]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        x, y = self.ds[self.keep[i]]
        return x, self.mapping[int(y)]


def instance_loaders(classes, train_loader, test_loader, batch_size=128):
    """(shuffled train, test) loaders for one instance, built from already-filtered project loaders."""
    return (DataLoader(Remapped(train_loader.dataset, classes), batch_size=batch_size, shuffle=True),
            DataLoader(Remapped(test_loader.dataset, classes), batch_size=256, shuffle=False))


# ---------------------------------------------------------------------------
# Descriptors and predictors (pure functions where possible)
# ---------------------------------------------------------------------------

def ridge_probe(Xtr, ytr, Xte, yte, n_classes, lam_rel=1e-3):
    """Closed-form ridge on standardized inputs (the project's probe convention)."""
    Xtr, Xte = Xtr.double(), Xte.double()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), n_classes, dtype=torch.float64)
    Y[torch.arange(len(ytr)), ytr.long()] = 1.0
    d = Xtr.shape[1]
    A = Xtr.t() @ Xtr + lam_rel * float((Xtr ** 2).sum()) / d * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    return float(((Xte @ W).argmax(1) != yte.long()).float().mean().item())


def fit_calibration(xs, ys):
    """OLS slope/intercept for acc_t ~ a + b*acc_ridge (pure function)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def predict_rib(acc_ridge, c, a, b):
    """Stage 2: calibrated accuracy -> the project's requirement formula. Returns (R_IB, acc_pred)."""
    acc = min(max(a + b * acc_ridge, 0.0), 1.0)
    return task_info_fano(1.0 - acc, c), acc


def rank_correlation(xs, ys):
    """Spearman with average ranks (pure function; needs >= 3 points)."""
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
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = math.sqrt(sum((p - mx) ** 2 for p in rx))
    dy = math.sqrt(sum((q - my) ** 2 for q in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def lib_verdict(rows):
    """Pre-registered verdict (docstring (4)). rows carry the per-instance LOO results."""
    rho = rank_correlation([r["acc_ridge"] for r in rows], [r["acc_t"] for r in rows])
    f1 = rho >= 0.80
    f2 = all(abs(r["rib_pred"] - r["rib"]) <= 0.30 for r in rows)
    over = [(r["c"], sorted(set(r["excl_pred"]) - set(r["excl_measured"]))) for r in rows]
    f3 = all(len(o[1]) == 0 for o in over)
    detail = [
        f"  F-1 family law (Spearman(acc_ridge, acc_t) >= 0.80): {rho:+.3f} -> {'PASS' if f1 else 'FAIL'}",
        f"  F-2 usable prediction (LOO |dR_IB| <= 0.30 nats for all): "
        f"{[(r['c'], round(r['rib_pred'] - r['rib'], 2)) for r in rows]} -> {'PASS' if f2 else 'FAIL'}",
        f"  F-3 safety (predicted exclusion subset of measured; violations = over-exclusion): {over} -> "
        f"{'PASS' if f3 else 'FAIL'}",
        f"  F-4 contrast (label-only reading ln C, same safety metric): "
        f"{[(r['c'], sorted(set(r['excl_label_only']) - set(r['excl_measured']))) for r in rows]}",
    ]
    score = sum([f1, f2, f3])
    if score == 3:
        return ("SUPPORT: within this family a seconds-cheap descriptor plus a leave-one-out calibration "
                "predicts the witness-relative reading well enough to plan with, and never over-excludes "
                "(the dangerous direction)"), detail
    if score == 2:
        return "PARTIAL: two of the three family-library criteria hold (recorded as-is; see the lines)", detail
    return "MIXED/REFUTED: the family-library criteria did not hold (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(c_grid, epochs, subset, quick):
    raw = "|".join([f"C{','.join(map(str, c_grid))}", f"ep{epochs}", f"sub{subset}", f"w{WIDTH}",
                    f"k{N_CODES}", f"d{D_REP}", f"perm{SEED_SUBSET_PERM}", f"q{int(quick)}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def descriptor_record(c, device, resume, ckpt_dir):
    """Cheap descriptors for one instance (seconds each): raw-pixel ridge and random-feature ridge."""
    path = os.path.join(ckpt_dir, f"desc_C{c}.json")
    if resume and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    classes = class_subset(c)
    tr_raw, te_raw = cifar_raw_loaders()
    tr, te = instance_loaders(classes, tr_raw, te_raw)
    xs, ys = [], []
    for x, y in tr:
        xs.append(x.flatten(1)); ys.append(y)
        if sum(len(t) for t in ys) >= PROBE_TRAIN:
            break
    Xtr, ytr = torch.cat(xs)[:PROBE_TRAIN], torch.cat(ys)[:PROBE_TRAIN]
    xs, ys = [], []
    for x, y in te:
        xs.append(x.flatten(1)); ys.append(y)
    Xte, yte = torch.cat(xs), torch.cat(ys)
    acc_ridge = 1.0 - ridge_probe(Xtr, ytr, Xte, yte, c)
    torch.manual_seed(50000 + c)
    bb = SmallCNN(d=D_REP, n_codes=N_CODES, width=WIDTH).to(device).eval()

    @torch.no_grad()
    def feats(loader, limit=None):
        out_x, out_y = [], []
        for x, y in loader:
            h = F.relu(bb.fc1(bb.features(x.to(device)).flatten(1)))
            out_x.append(h.cpu()); out_y.append(y)
            if limit is not None and sum(len(t) for t in out_y) >= limit:
                break
        X, Y = torch.cat(out_x), torch.cat(out_y)
        return (X[:limit], Y[:limit]) if limit else (X, Y)

    ftr, ftr_y = feats(tr, limit=PROBE_TRAIN)
    fte, fte_y = feats(te)
    acc_randfeat = 1.0 - ridge_probe(ftr, ftr_y, fte, fte_y, c)
    rec = dict(c=c, classes=classes, acc_ridge=acc_ridge, acc_randfeat=acc_randfeat,
               n_train=len(ytr), n_test=len(yte))
    json.dump(rec, open(path, "w", encoding="utf-8"))
    return rec


def measured_record(c, epochs, subset, device, resume, ckpt_dir):
    """The ground truth to be predicted: the reading from a task-TRAINED witness (fixed protocol)."""
    path = os.path.join(ckpt_dir, f"witness_C{c}.pt")
    if resume and os.path.exists(path):
        return float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
    classes = class_subset(c)
    tr_aug, te_aug = load_cifar(batch_size=128, subset=subset, workers=0)
    tr, te = instance_loaders(classes, tr_aug, te_aug)
    torch.manual_seed(51000 + c)
    model = SmallCNN(d=D_REP, n_codes=N_CODES, width=WIDTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device)
    acc, _ = evaluate(model, te, device=device)
    torch.save({"acc": acc, "state_dict": model.state_dict()}, path)
    return acc


def run_library(c_grid, epochs, subset, resume, device, quick=False):
    fp = run_signature(c_grid, epochs, subset, quick)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 92)
    print("[FAMILY-LIBRARY] within-family amortisation of the witness-relative reading")
    print("=" * 92)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [family] CIFAR-10 label subsets C in {c_grid} (fixed permutation seed {SEED_SUBSET_PERM}) ; "
          f"witness protocol = SmallCNN {epochs} ep / train subset {subset} (the library is protocol-relative)")
    rows = []
    for c in c_grid:
        t0 = time.time()
        desc = descriptor_record(c, device, resume, ckpt_dir)
        acc_t = measured_record(c, epochs, subset, device, resume, ckpt_dir)
        rib = task_info_fano(1.0 - acc_t, c)
        k_min = DistillCapacity().min_codebook_size(rib)
        rows.append(dict(c=c, acc_ridge=desc["acc_ridge"], acc_randfeat=desc["acc_randfeat"],
                         acc_t=acc_t, rib=rib, k_min=k_min,
                         excl_measured=excluded_set(rib, EXCL_GRID),
                         excl_label_only=excluded_set(math.log(c), EXCL_GRID)))
        print(f"  C={c:2d}: acc_ridge={desc['acc_ridge']:.3f} acc_randfeat={desc['acc_randfeat']:.3f} "
              f"| trained witness acc={acc_t:.3f} -> R_IB={rib:.3f} K_min={k_min} "
              f"excl={rows[-1]['excl_measured']} [{time.time() - t0:.0f}s]")
    for i, r in enumerate(rows):
        xs = [rows[j]["acc_ridge"] for j in range(len(rows)) if j != i]
        ys = [rows[j]["acc_t"] for j in range(len(rows)) if j != i]
        a, b = fit_calibration(xs, ys)
        rib_pred, acc_pred = predict_rib(r["acc_ridge"], r["c"], a, b)
        r["rib_pred"], r["acc_pred"] = rib_pred, acc_pred
        r["k_min_pred"] = DistillCapacity().min_codebook_size(rib_pred)
        r["excl_pred"] = excluded_set(rib_pred, EXCL_GRID)
        print(f"  LOO C={r['c']:2d}: calibrated acc_pred={acc_pred:.3f} (measured {r['acc_t']:.3f}) -> "
              f"R_IB_pred={rib_pred:.3f} (measured {r['rib']:.3f}) ; K_min_pred={r['k_min_pred']} "
              f"(measured {r['k_min']}) ; excl_pred={r['excl_pred']}")
    print("\n  --- FAMILY-LIBRARY verdict ---")
    verdict, detail = lib_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    for r in rows:
        jsonl_append(JSONL, dict(r, fp=fp, verdict=verdict))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[FAMILY-LIBRARY] within-family amortisation of the witness-relative reading\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def verdict_only():
    if not os.path.exists(JSONL):
        raise SystemExit("[FAMILY] no family.jsonl to re-validate")
    latest = {}
    for line in open(JSONL, encoding="utf-8"):
        r = json.loads(line)
        latest[r["c"]] = r
    rows = [latest[c] for c in sorted(latest)]
    print("[FAMILY-LIBRARY offline re-validation]")
    for r in rows:
        print(f"  C={r['c']:2d}: acc_ridge={r['acc_ridge']:.3f} acc_t={r['acc_t']:.3f} "
              f"R_IB {r['rib']:.2f} -> pred {r['rib_pred']:.2f} | K_min {r['k_min']} -> {r['k_min_pred']} | "
              f"excl {r['excl_measured']} vs pred {r['excl_pred']}")
    verdict, detail = lib_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")


def run_selftest():
    ok = []
    ok.append(("class_subset returns C classes", len(class_subset(4)) == 4))
    ok.append(("class_subset is deterministic", class_subset(4) == class_subset(4)))
    ok.append(("class_subset is a subset of 0..9", set(class_subset(6)) <= set(range(10))))
    a, b = fit_calibration([0.2, 0.4, 0.6], [0.4, 0.6, 0.8])
    ok.append(("fit_calibration recovers a perfect linear law", abs(a - 0.2) < 1e-9 and abs(b - 1.0) < 1e-9))
    rib, acc = predict_rib(0.5, 10, 0.2, 1.0)
    ok.append(("predict_rib applies the project formula", abs(rib - task_info_fano(0.3, 10)) < 1e-9))
    ok.append(("predict_rib clips the accuracy to [0,1]", predict_rib(5.0, 10, 0.0, 1.0)[1] == 1.0))
    ok.append(("rank_correlation == 1 on a monotone family",
               abs(rank_correlation([0.1, 0.3, 0.5], [0.2, 0.4, 0.6]) - 1.0) < 1e-9))
    base = dict(c=10, acc_ridge=0.50, acc_randfeat=0.40, acc_t=0.80, rib=1.40, k_min=4,
                excl_measured=[2], excl_label_only=[2, 4], excl_pred=[2], rib_pred=1.35, acc_pred=0.79,
                k_min_pred=4)
    two = [dict(base), dict(base, c=6, acc_ridge=0.55, acc_t=0.83, rib=1.20, rib_pred=1.15),
           dict(base, c=3, acc_ridge=0.60, acc_t=0.86, rib=0.95, rib_pred=0.90)]
    v, _ = lib_verdict(two)
    ok.append(("verdict SUPPORT on a clean three-instance family", v.startswith("SUPPORT")))
    v, _ = lib_verdict([dict(base, excl_pred=[2, 4])])
    ok.append(("verdict not SUPPORT when the prediction over-excludes", not v.startswith("SUPPORT")))
    v, _ = lib_verdict([dict(base, rib_pred=0.50)])
    ok.append(("verdict not SUPPORT when the prediction is far off", not v.startswith("SUPPORT")))
    print("[FAMILY selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="FAMILY-LIBRARY: within-family amortisation of the reading")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--train-subset", type=float, default=TRAIN_SUBSET,
                    help="fraction of the (class-filtered) training split used to train the witness")
    ap.add_argument("--quick", action="store_true", help="3 instances, 2 epochs (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--verdict-only", action="store_true", help="re-validate from family.jsonl (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.quick:
        run_library([2, 4, 10], 2, 0.05, not args.fresh, device, quick=True)
    else:
        run_library(C_GRID, args.epochs, args.train_subset, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
