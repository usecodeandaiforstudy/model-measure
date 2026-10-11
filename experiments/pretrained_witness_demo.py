"""PRETRAINED-WITNESS: can a PRETRAINED model serve as the witness -- i.e. with NO target-side training?

Every witness in this project so far was trained on the target task (1-NN/linear probe on raw pixels, a
short CNN, the calibrated teacher). The realistic use of the tool is different: an engineer already has a
pretrained backbone and wants the requirement reading WITHOUT training anything on the target. This
experiment tests exactly that, under a controlled pretraining design (same architecture, only the
pretraining source changes), with the target-side ground truth taken entirely from the CP-H cache
(13-level student grid x 3 seeds) -- so the cost of every witness is explicit.

Usage:
    python experiments/pretrained_witness_demo.py            # ~10 min GPU (2 pretrainings + probes)
    python experiments/pretrained_witness_demo.py --quick    # 1 epoch pretraining (smoke)
    python experiments/pretrained_witness_demo.py --selftest # pure-function unit tests (no training)
    python experiments/pretrained_witness_demo.py --verdict-only

Pre-registration (2026-09-30, criteria before data)
-----------------------------------------------------------------------------------------------
(1) Target task: CIFAR-10 10-class (the CP-H task). NO target-side training happens anywhere in this
    experiment: the reference and the measured student grid are read from the CP-H cache.
(2) Witnesses (the "ladder"), all evaluated on the target test set:
      P0 raw-pixel ridge probe        -- the cheapest non-pretrained baseline (recomputed, cross-checked
                                         against the value witness_demo recorded)
      P1 random-init backbone + ridge -- no pretraining at all (control: architecture + random features)
      P2 MNIST-pretrained backbone + ridge   -- CROSS-domain pretraining (grayscale digits -> color objects)
      P3 rotation-pretrained backbone + ridge -- SAME-domain, label-free pretraining (4-way rotation
                                         prediction on CIFAR-10 images; no target labels are used)
      P4 rotation-pretrained backbone + 1-NN  -- the same features with a non-parametric readout
      P5 8-epoch CNN on the target (witness_demo's W3, read from witness.jsonl) -- a cheap TRAINED witness
      P6 the CP-H teacher (120 ep, acc ~0.80) -- the strongest trained witness (reference)
    Backbone = cifar_demo.SmallCNN(d=64, n_codes=256, width=96) (the CP-H architecture); the probe feature
    is the pre-VQ representation relu(fc1(features(x))) (dim 64). Pretraining budget: CIFAR-10/MNIST 20%
    subset (10000 images), 5 epochs, Adam 1e-3. Probe = closed-form ridge on standardized features
    (lambda = 1e-3*trace/d), train subset 10000, i.e. seconds of CPU.
(3) Conventions (fixed): R_IB = task_info_fano(1 - witness_acc, 10); K_min = ceil(e^R_IB); a K is excluded
    when log K < R_IB; coverage = |excluded & measured-failures| / |measured-failures| on the CP-H grid
    (relative-gap convention rel >= 0.50 fails, <= 0.10 succeeds, as in witness_demo); a FALSE EXCLUSION is
    an excluded K that the CP-H cache measured as a success; looseness = ln(K_wall)/R_IB with K_wall from
    the CP-H grid (measured_wall, rel_tol=0.10).
(4) Pre-registered criteria:
      P-A (soundness for EVERY witness): zero false exclusions across the whole ladder, the pretrained ones
          included. A violation would break the witness-relative reading in its most likely real use.
      P-B (a pretrained witness is USEFUL): at least one frozen pretrained backbone reaches acc >= 0.65
          (>= 80% of the CP-H teacher's ~0.80) with NO target-side training.
      P-C (domain gap, reported): acc(P3 same-domain) - acc(P2 cross-domain) >= 0.10.
      P-D (pretraining beats random features): acc(P3) - acc(P1) >= 0.15.
      P-E (cost-coverage table, reported): target-side cost vs (acc, R_IB, K_min, excluded set, coverage,
          looseness) for every rung.
    Verdict (pure function pw_verdict): SUPPORT if P-A and P-B and P-D hold; PARTIAL if P-A and exactly one
    of P-B/P-D; else MIXED/REFUTED -- recorded as-is.
(5) Relay: pretrained backbones cached under experiments/data/pw/<fingerprint>/; --fresh recomputes.
(6) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads
    (CIFAR-10 and MNIST are both in experiments/data/).
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (train_one_epoch, task_info_fano, DistillCapacity, measured_wall, looseness)
from cifar_demo import DATA_ROOT, SmallCNN
from cph_demo import load_cifar
from witness_demo import (cifar_cached_acc, cifar_ckpt_dir, cifar_raw_loaders,
                          excluded_set, coverage, false_exclusions, C_CIFAR, CPH_GRID)

N_CLASSES = C_CIFAR
D_REP = 64
WIDTH = 96
N_CODES = 256
PRETRAIN_SUBSET = 0.20
PRETRAIN_EPOCHS = 5
PROBE_TRAIN = 10000
NN_MEMORY = 2000
NN_QUERY = 1000
LR = 1e-3
SEED_PRETRAIN = 41000
SEED_PROBE = 42000

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "pw.jsonl")
LOG = os.path.join(_DATA, "pw.log")
REPORT = os.path.join(_DATA, "pw_report.txt")
CKPT_ROOT = os.path.join(_DATA, "pw")


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
# Frozen-feature extraction and the probe
# ---------------------------------------------------------------------------

def build_backbone(ckpt=None, device="cpu"):
    """SmallCNN with the pre-VQ representation exposed. ckpt=None -> random init (the P1 control)."""
    model = SmallCNN(d=D_REP, n_codes=N_CODES, width=WIDTH).to(device)
    if ckpt is not None and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device, limit=None, raw=False):
    """Pre-VQ representation relu(fc1(features(x))) -- the standard frozen-feature transfer protocol.
    raw=True returns the flattened pixels instead (the P0 raw-pixel baseline)."""
    xs, ys = [], []
    for x, y in loader:
        h = x.flatten(1).to(device) if raw else \
            F.relu(model.fc1(model.features(x.to(device)).flatten(1)))
        xs.append(h.cpu())
        ys.append(y)
        if limit is not None and sum(len(t) for t in ys) >= limit:
            break
    X = torch.cat(xs)
    y = torch.cat(ys)
    return (X[:limit] if limit else X), (y[:limit] if limit else y)


def ridge_probe(Xtr, ytr, Xte, yte, n_classes=N_CLASSES, lam_rel=1e-3):
    """Closed-form ridge on standardized features (the project's probe convention)."""
    Xtr, Xte = Xtr.double(), Xte.double()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), n_classes, dtype=torch.float64)
    Y[torch.arange(len(ytr)), ytr.long()] = 1.0
    d = Xtr.shape[1]
    A = Xtr.t() @ Xtr + lam_rel * float((Xtr ** 2).sum()) / d * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    return float(((Xte @ W).argmax(1) != yte.long()).float().mean().item())


def knn_features(Xtr, ytr, Xte, yte, memory=NN_MEMORY, query=NN_QUERY):
    """1-NN on frozen features (zero training), same subsampling convention as witness_demo."""
    Xm = F.normalize(Xtr[:memory].float(), dim=-1)
    ym = ytr[:memory].long()
    Xq = F.normalize(Xte[:query].float(), dim=-1)
    yq = yte[:query].long()
    return float((ym[torch.cdist(Xq, Xm).argmin(dim=1)] != yq).float().mean().item())


# ---------------------------------------------------------------------------
# Pretraining sources
# ---------------------------------------------------------------------------

def mnist_loader(subset=PRETRAIN_SUBSET):
    """Cross-domain source: MNIST digits adapted to the 3x32x32 input the backbone expects."""
    tf = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=3),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    n_keep = int(len(ds) * subset)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n_keep].tolist()
    return DataLoader(torch.utils.data.Subset(ds, idx), batch_size=128, shuffle=True)


class RotationTask(torch.utils.data.Dataset):
    """Same-domain, LABEL-FREE source: predict which of 4 rotations was applied (no CIFAR labels used)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        k = i % 4                       # deterministic and balanced (the label IS the rotation)
        return torch.rot90(x, k, dims=(1, 2)), k


def pretrain(source, epochs, device, ckpt_path, resume):
    """Train the backbone on a source task; returns the path (cached: relay mode)."""
    if resume and os.path.exists(ckpt_path):
        return ckpt_path
    torch.manual_seed(SEED_PRETRAIN + (0 if source == "mnist" else 1))
    model = SmallCNN(d=D_REP, n_codes=N_CODES, width=WIDTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if source == "mnist":
        loader = mnist_loader()
    else:
        tr, _ = load_cifar(batch_size=128, subset=PRETRAIN_SUBSET, workers=0)
        loader = DataLoader(RotationTask(tr.dataset), batch_size=128, shuffle=True)
    for _ in range(epochs):
        train_one_epoch(model, loader, opt, device=device)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Readout assembly and the pre-registered verdict
# ---------------------------------------------------------------------------

def witness_row(name, acc, acc_t, accs_students, grid, cost):
    """One rung of the ladder: the requirement reading and its coverage against the measured grid."""
    r_ib = task_info_fano(1.0 - acc, N_CLASSES)
    k_min = DistillCapacity().min_codebook_size(r_ib)
    rel = {K: (acc_t - accs_students[K]) / max(acc_t, 1e-12) for K in grid}
    fails = sorted(K for K in grid if rel[K] >= 0.50)
    succeeds = sorted(K for K in grid if rel[K] <= 0.10)
    excl = excluded_set(r_ib, grid)
    k_wall, _ = measured_wall(grid, accs_students, acc_t, rel_tol=0.10)
    return dict(name=name, acc=acc, cost=cost, r_ib=r_ib, k_min=k_min, excluded=excl,
                coverage=coverage(excl, fails), false_excl=false_exclusions(excl, succeeds),
                k_wall=k_wall, looseness=looseness(r_ib, k_wall), fails=fails, succeeds=succeeds)


def witness_threshold(k_target, n_classes=N_CLASSES):
    """Minimum witness accuracy for the certified rule to exclude K = k_target.

    R_IB(eps) = ln C - h_b(eps) - eps*ln(C-1) is decreasing in eps, and the rule excludes K iff
    ln K < R_IB(1 - acc). Solving R_IB(eps) = ln K by bisection gives eps*, so acc* = 1 - eps* is the
    accuracy a witness must BEAT before its reading excludes anything at K. This turns the negative
    result of this experiment into an actionable design rule (pure function, unit-testable)."""
    target = math.log(k_target)
    if target >= math.log(n_classes):
        return float("inf")                     # K >= C is never excluded by the requirement reading
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if task_info_fano(mid, n_classes) > target:
            lo = mid                            # exclusion still holds at this error -> push eps up
        else:
            hi = mid
    return 1.0 - lo


def pw_verdict(rows):
    """Pre-registered verdict (docstring (4))."""
    pretrained = [r for r in rows if r["name"] in ("P2_mnist_probe", "P3_rot_probe")]
    p_a = all(len(r["false_excl"]) == 0 for r in rows)
    p_b = any(r["acc"] >= 0.65 for r in pretrained)
    accs = {r["name"]: r["acc"] for r in rows}
    p_c = (accs.get("P3_rot_probe", 0.0) - accs.get("P2_mnist_probe", 0.0)) >= 0.10
    p_d = (accs.get("P3_rot_probe", 0.0) - accs.get("P1_random_probe", 0.0)) >= 0.15
    detail = [
        f"  P-A soundness for every witness (zero false exclusions): "
        f"{[(r['name'], len(r['false_excl'])) for r in rows]} -> {'PASS' if p_a else 'FAIL'}",
        f"  P-B a pretrained witness is useful (acc >= 0.65 with no target-side training): "
        f"{[(r['name'], round(r['acc'], 3)) for r in pretrained]} -> {'PASS' if p_b else 'FAIL'}",
        f"  P-C domain gap (same-domain rotation minus cross-domain MNIST >= 0.10): "
        f"{round(accs.get('P3_rot_probe', 0.0) - accs.get('P2_mnist_probe', 0.0), 3)} -> "
        f"{'PASS' if p_c else 'FAIL'}",
        f"  P-D pretraining beats random features (>= 0.15): "
        f"{round(accs.get('P3_rot_probe', 0.0) - accs.get('P1_random_probe', 0.0), 3)} -> "
        f"{'PASS' if p_d else 'FAIL'}",
        f"  P-E cost vs coverage (target-side cost, acc, R_IB, K_min, excluded, coverage, looseness):",
    ]
    for r in rows:
        detail.append(f"      {r['name']:16s} cost={r['cost']:14s} acc={r['acc']:.3f} "
                      f"R_IB={r['r_ib']:.2f} K_min={r['k_min']:3d} excl={r['excluded']} "
                      f"cov={r['coverage']:.2f} loose={None if r['looseness'] is None else round(r['looseness'], 2)} "
                      f"false={r['false_excl']}")
    if p_a and p_b and p_d:
        return ("SUPPORT: a frozen pretrained backbone is a usable witness with NO target-side training "
                "(P-B), pretraining (not just the architecture) is what buys the accuracy (P-D), and the "
                "witness-relative reading stays sound across the whole ladder (P-A)"), detail
    if p_a and (p_b or p_d):
        return "PARTIAL: soundness holds and one of the two pretrained-witness claims holds (recorded)", detail
    return "MIXED/REFUTED: the pretrained-witness claims did not hold (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(epochs, quick):
    raw = "|".join([f"ep{epochs}", f"sub{PRETRAIN_SUBSET}", f"w{WIDTH}", f"k{N_CODES}", f"d{D_REP}",
                    f"q{int(quick)}", "target=cifar10", "src=mnist+rotation"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def run_pw(epochs, resume, device, quick=False):
    fp = run_signature(epochs, quick)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 92)
    print("[PRETRAINED-WITNESS] can a pretrained model serve as the witness (no target-side training)?")
    print("=" * 92)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [target] CIFAR-10; reference + measured grid from the CP-H cache ({cifar_ckpt_dir()})")
    print(f"  [pretraining] subset={PRETRAIN_SUBSET} epochs={epochs} sources=MNIST(cross-domain) / "
          f"rotation(same-domain, label-free)")
    acc_t = cifar_cached_acc("teacher", 0)
    if acc_t is None:
        raise SystemExit("[PW] the CP-H cache is missing; run experiments/cph_demo.py first.")
    accs_students = {K: cifar_cached_acc("student", 0, K) for K in CPH_GRID}
    print(f"  [reference] CP-H teacher acc={acc_t:.3f}; student grid {CPH_GRID}")
    tr_raw, te_raw = cifar_raw_loaders()
    Xtr_raw, ytr_raw = extract_features(None, tr_raw, device, limit=PROBE_TRAIN, raw=True)
    Xte_raw, yte_raw = extract_features(None, te_raw, device, raw=True)
    t0 = time.time()
    acc_p0 = 1.0 - ridge_probe(Xtr_raw, ytr_raw, Xte_raw, yte_raw)
    t_p0 = time.time() - t0
    print(f"  P0 raw-pixel ridge probe: acc={acc_p0:.3f} [{t_p0:.0f}s, no pretraining] ; "
          f"witness_demo recorded W2={_witness_recorded('P0_raw_pixels_probe')} (cross-check)")
    rows = [witness_row("P0_raw_pixels_probe", acc_p0, acc_t, accs_students, CPH_GRID,
                        f"ridge ({t_p0:.0f}s)")]
    for name, src, tag in (("P2_mnist", "mnist", "cross-domain MNIST"),
                           ("P3_rot", "rotation", "same-domain rotation (label-free)")):
        path = pretrain(src, epochs, device, os.path.join(ckpt_dir, f"bb_{src}.pt"), resume)
        model = build_backbone(path, device)
        Xtr, ytr = extract_features(model, tr_raw, device, limit=PROBE_TRAIN)
        Xte, yte = extract_features(model, te_raw, device)
        t1 = time.time()
        acc_probe = 1.0 - ridge_probe(Xtr, ytr, Xte, yte)
        acc_nn = 1.0 - knn_features(Xtr, ytr, Xte, yte)
        dt = time.time() - t1
        cost = f"pretrain {epochs}ep + probe"
        print(f"  {tag:34s}: probe acc={acc_probe:.3f} ; 1-NN acc={acc_nn:.3f} "
              f"[{epochs} ep pretrain + {dt:.0f}s probe, no target labels]")
        rows.append(witness_row(f"{name}_probe", acc_probe, acc_t, accs_students, CPH_GRID, cost))
        rows.append(witness_row(f"{name}_1nn", acc_nn, acc_t, accs_students, CPH_GRID, cost))
    model = build_backbone(None, device)                      # P1: random init (the control)
    Xtr, ytr = extract_features(model, tr_raw, device, limit=PROBE_TRAIN)
    Xte, yte = extract_features(model, te_raw, device)
    acc_p1 = 1.0 - ridge_probe(Xtr, ytr, Xte, yte)
    print(f"  P1 random-init backbone + probe: acc={acc_p1:.3f} [no pretraining]")
    rows.append(witness_row("P1_random_probe", acc_p1, acc_t, accs_students, CPH_GRID,
                            "random init + probe"))
    for name, note in (("P5_target_8ep_cnn", "8 epochs on target"),
                       ("P6_cph_teacher", "120 epochs on target (reference)")):
        acc = _witness_recorded(name)
        if acc is not None:
            rows.append(witness_row(name, acc, acc_t, accs_students, CPH_GRID, note))
    order = ["P0_raw_pixels_probe", "P1_random_probe", "P2_mnist_probe", "P2_mnist_1nn",
             "P3_rot_probe", "P3_rot_1nn", "P5_target_8ep_cnn", "P6_cph_teacher"]
    rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    print("\n  --- cost vs coverage ---")
    for r in rows:
        print(f"    {r['name']:20s} acc={r['acc']:.3f} R_IB={r['r_ib']:5.2f} K_min={r['k_min']:3d} "
              f"excluded={str(r['excluded']):24s} coverage={r['coverage']:.2f} "
              f"false={r['false_excl']} looseness={None if r['looseness'] is None else round(r['looseness'], 2)}")
    print("\n  --- PRETRAINED-WITNESS verdict ---")
    verdict, detail = pw_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    jsonl_append(JSONL, dict(rows=rows, acc_t=acc_t, verdict=verdict, fp=fp))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[PRETRAINED-WITNESS] frozen pretrained backbone as the witness\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def _witness_recorded(name):
    """Read the target-trained witnesses from witness.jsonl (P5 = W3 8-ep CNN, P6 = W4 teacher,
    P0 = W2 linear probe for the cross-check)."""
    path = os.path.join(_DATA, "witness.jsonl")
    if not os.path.exists(path):
        return None
    key = {"P5_target_8ep_cnn": "W3", "P6_cph_teacher": "W4",
           "P0_raw_pixels_probe": "W2"}.get(name)
    if key is None:
        return None
    latest = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == "C":
            latest = r
    return None if latest is None else round(float(latest["accs"][key]), 3)


def verdict_only():
    """Offline re-verdict from pw.jsonl (no training)."""
    if not os.path.exists(JSONL):
        raise SystemExit("[PW] no pw.jsonl to re-verdict")
    rec = json.loads(open(JSONL, encoding="utf-8").read().strip().split("\n")[-1])
    rows = rec["rows"]
    print("[PRETRAINED-WITNESS offline re-verdict]")
    print("  witness-accuracy thresholds (the reading is empty below these): "
          + " ; ".join(f"K={K}: acc >= {witness_threshold(K):.3f}" for K in (2, 3, 4, 8))
          + f" ; K >= ceil(e^ln C) = {N_CLASSES} can NEVER be excluded on a {N_CLASSES}-class task")
    for r in rows:
        print(f"  {r['name']:20s} acc={r['acc']:.3f} R_IB={r['r_ib']:5.2f} K_min={r['k_min']:3d} "
              f"excluded={str(r['excluded']):24s} coverage={r['coverage']:.2f} false={r['false_excl']}")
    verdict, detail = pw_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")


def run_selftest():
    ok = []
    rows = [dict(name="P0_raw_pixels_probe", acc=0.30, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128],
                 verdict="FAIL: below K_min"),
            dict(name="P1_random_probe", acc=0.30, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128],
                 verdict="FAIL: below K_min"),
            dict(name="P2_mnist_probe", acc=0.45, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128],
                 verdict="FAIL: below K_min"),
            dict(name="P3_rot_probe", acc=0.72, cost="", r_ib=2.0, k_min=8, excluded=[2, 3],
                 coverage=0.5, false_excl=[], k_wall=128, looseness=2.5, fails=[2, 3, 4], succeeds=[128],
                 verdict="FAIL: below K_min")]
    v, _ = pw_verdict(rows)
    ok.append(("verdict SUPPORT when sound + useful + pretraining beats random", v.startswith("SUPPORT")))
    bad = [dict(r) for r in rows]
    bad[3]["false_excl"] = [16]
    v, _ = pw_verdict(bad)
    ok.append(("verdict not SUPPORT when a witness produces a false exclusion", not v.startswith("SUPPORT")))
    weak = [dict(r) for r in rows]
    weak[3]["acc"] = 0.40
    v, _ = pw_verdict(weak)
    ok.append(("verdict not SUPPORT when no pretrained witness is useful", not v.startswith("SUPPORT")))
    mix = [dict(r) for r in rows]
    mix[3]["acc"] = 0.62                                    # useful-ish, but not 0.65
    v, _ = pw_verdict(mix)
    ok.append(("verdict PARTIAL/MIXED when P-B fails but soundness holds", v.startswith(("PARTIAL", "MIXED"))))
    ok.append(("excluded_set uses log K < R_IB", excluded_set(2.0, [2, 4, 8, 16]) == [2, 4]))
    ok.append(("coverage counts only measured failures", abs(coverage([2, 4], [2, 4, 8]) - 2 / 3) < 1e-9))
    ok.append(("coverage is 1.0 when every failure is excluded", coverage([2, 4], [2, 4]) == 1.0))
    ok.append(("false_exclusions finds the overlap", false_exclusions([2, 4], [2, 8]) == [2]))
    th = {K: witness_threshold(K) for K in (2, 3, 4, 8)}
    ok.append(("witness thresholds are monotone in K", all(th[a] < th[b] for a, b in
                                                          zip([2, 3, 4], [3, 4, 8]))))
    ok.append(("the K=4 threshold is just above the CP-H teacher's accuracy",
               abs(th[4] - 0.807) < 0.01 and th[4] > 0.803))
    ok.append(("witness threshold for K=8 sits just under 1-R_IB inversion",
               abs(task_info_fano(1.0 - th[8], N_CLASSES) - math.log(8)) < 1e-6))
    ok.append(("K >= n_classes is never excluded", witness_threshold(16) == float("inf")))
    print("[PW selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="PRETRAINED-WITNESS: frozen pretrained backbone as the witness")
    ap.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--quick", action="store_true", help="1 pretraining epoch (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--verdict-only", action="store_true", help="re-verdict from pw.jsonl (no training)")
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
    run_pw(1 if args.quick else args.epochs, not args.fresh, device, quick=args.quick)
    print("\nDone.")


if __name__ == "__main__":
    main()
