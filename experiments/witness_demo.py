"""WITNESS: how witness strength controls the requirement bound R_IB and its exclusion coverage.

Motivation (2026-09-30 discussion): the requirement reading is *witness-relative* -- R_IB is obtained by
inverting Fano on an ACHIEVED error rate, so a weak witness yields a looser (smaller) lower bound and
misses exclusions. Two questions are open and have never been measured in this project:

  Q1 (witness ladder, task A = clean grid64): how much does the bound loosen as the witness gets cheaper,
     and do cheap witnesses ever produce an UNSOUND exclusion (excluding a student that measurably succeeds)?
  Q2 (label-only reading, task B = merged grid64): the alternative "read the requirement off the labels"
     proposal uses H(Y) = ln C, which is an UPPER bound on the requirement. On a task whose fine-grained
     distinction is unlearnable, that upper bound over-demands and should FALSELY EXCLUDE students that
     measurably match the reference. This script measures that false-exclusion count.

Usage:
    python experiments/witness_demo.py                 # both tasks, relay mode (default), ~25 min GPU
    python experiments/witness_demo.py --task A        # clean grid64 only
    python experiments/witness_demo.py --selftest      # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Tasks (encoding constants mirror cifar_demo.make_grid64_images; do not tune them):
    A "clean": 64 classes = 3 bands x 4 block positions, 150/class, sigma_pix=0.15, sigma_bright=0.1.
      H(Y) = ln 64 = 4.159 nats; the task is (near-)deterministic, so a strong witness approaches
      epsilon ~ 0 and the label-only reading ln C is expected to coincide with the strongest witness.
    B "merged": 8 templates (the 8 v = g*8 positions of the same encoding); every sample gets its group's
      template but a LABEL drawn uniformly from that group's 8 labels (g*8 + U{0..7}). Hence
      H(Y) = ln 64 = 4.159 while H(Y|X) = ln 8 = 2.079: the finest distinction is unlearnable, the Bayes
      error is ~0.875, and the intrinsic requirement is ~ln 8 nats -- while the label-only reading still
      demands ln 64.
(2) Witness ladder (per task, per seed), all on the SAME train/test split:
    W1  1-NN on raw flattened inputs, train -> test (zero training; memory subsampled to 4000)
    W2  linear probe on raw flattened inputs (standardized, closed-form ridge on one-hot targets; seconds)
    W3  GridCNN(d=64, K=256, euclid), 4 epochs (1/10 of the calibrated budget)
    W4  GridCNN(d=64, K=256, euclid), 40 epochs (the calibrated reference; also the KD teacher below)
    W0  label-only reading: requirement := ln C (no model; reported for contrast, not a witness).
    Note: capacity_lens.err_1nn is a leave-one-out representation gauge (a different quantity), so the
    W1 witness is computed here explicitly as train -> test nearest neighbour.
(3) Measured truth (both tasks): student grid distilled from W4 with the standard protocol
    (train_one_epoch(..., distill_from=W4), 40 ep, Adam 1e-3), same architecture, only K varies.
    Base grids (these define the fingerprint): A = [2,3,4,6,8,12,16,24,32,48,64]; B = [2,3,4,6,8,16,32,64].
    Top extension (--extend-top; amendment 2026-09-30, recorded AFTER the base run, criteria unchanged,
    the added K values are new data): A += [128,192,256]; B += [128,256]. Reason: the base grids stop
    exactly at the wall (K=64), so no student sits at the teacher's own capacity and -- because of the
    known KD tail decay -- the "measured success" set can only come out empty, which leaves T2 vacuous.
    Twin-student sizes make the success set reachable. Per the project convention (research-summary 10.4) the
    fingerprint covers only the BASE grids, so the extension reuses every cached base unit.
    Gap g(K) = acc_W4 - acc_student(K); the thresholds are RELATIVE to the reference accuracy
    (amendment 2026-09-30, recorded BEFORE the student grid was run: absolute thresholds are unreachable
    when the reference accuracy itself is ~0.125, as on task B, which would make the coverage criterion
    vacuous -- a definitional bug, not a tuning; the criteria set and their direction are unchanged):
      rel(K) = g(K) / acc_W4
      measured_success S = {K : rel(K) <= 0.10}   (student retains >= 90% of the reference accuracy)
      measured_fail    F = {K : rel(K) >= 0.50}   (student retains <= 50%; in-between = gray, in neither)
(4) Per-witness readouts: acc, eps = 1-acc, R_IB = task_info_fano(eps, C), K_min = ceil(e^R_IB),
    excluded set E(W) = {K in grid : log K < R_IB(W)}, coverage(W) = |E(W) INTERSECT F| / |F|.
    The acc part of g0_gate is REPORTED but is not used as a filter: on task B the calibrated acc gate
    cannot pass by construction (the task's own ceiling is ~0.125) and the Fano bound is nevertheless
    valid (it needs only an achieved error) -- this is exactly the regime where the label-only reading fails.
(5) Pre-registered criteria (pure functions; >=2/3 seeds for task A, >=1/2 for task B):
    T1 monotonicity: R_IB is non-decreasing along W1 -> W2 -> W3 -> W4 (tolerance 0.01).
    T2 soundness (the decisive safety property): for EVERY witness and seed, E(W) INTERSECT S = {} --
       no witness may exclude a student that measurably matches the reference.
    T3 coverage profile; classify STEEP if mean coverage(W4) - mean coverage(W1) >= 0.3, else SHALLOW.
    U1 (task B, the label-only reading): false-exclusion count of the rule "log K < ln C" (i.e. K < C)
       against the measured success set S, compared with the same count for the strongest witness W4.
       Prediction: the label-only rule produces >= 1 false exclusion on task B (the over-demand direction
       is unsafe), while W4 produces 0 on both tasks.
(6) Verdict (pure function witness_verdict): UNSOUND if T2 fails; else SUPPORT if T1 and STEEP; else
    SHALLOW if T1 and not STEEP (cheap witnesses already suffice -- good news for usability); else MIXED.
(7) Seed scheme: task A seeds 0,1,2; task B seeds 0,1 (data seed s). Unit seeds: W3 = 21000+100*s,
    W4 = 21001+100*s, students = 22000+100*s+K (disjoint from all existing families).
(8) Relay: per-unit checkpoints under experiments/data/witness/<fingerprint>/; --fresh recomputes.
    Fingerprint = (tasks, epochs, grid A, grid B); the seed count is excluded.
(9) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads
    (both tasks are generated in-process). A negative result is archived as-is.
(10) Task C = CIFAR-10 (real task; added 2026-09-30 as the third task, criteria unchanged). It is the
    regime where the cheap witnesses are genuinely weak and the reference error sits mid-range --
    the place where T2/T3 can actually be pressed:
      W1 1-NN on raw pixels (memory 2000 / query 1000, zero training)
      W2 closed-form ridge on standardized raw pixels (fixed 10000-sample train subset)
      W3 SmallCNN(d=64, K=256, width=96) with a SHORT budget (--cifar-cheap-epochs, default 8; Adam 1e-3,
         no cosine) -- note this is a cheaper protocol than the CP-H budget, by construction
      W4 = the CACHED CP-H calibrated teacher (SmallCNN width96 K=256, 120 ep cosine, acc ~0.799);
         the measured truth = the CACHED CP-H student grid (CALIBRATED_K, 13 levels x 3 seeds).
      So task C costs one short CNN run + two zero-training witnesses; the reference and the student grid
      are read from the CP-H cache (no training). If the cache is missing the script stops with a hint.
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
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (train_one_epoch, evaluate, task_info_fano, DistillCapacity, g0_gate,
                           excluded_set, coverage, false_exclusions)
from cp3_demo import build_model
from cifar_demo import DATA_ROOT, SmallCNN
from cph_demo import load_cifar, run_signature as cph_run_signature, CALIBRATED_K as CPH_GRID
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as _DL

C_TASK = 64
C_CIFAR = 10
N_PER_CLASS = 150
K_TEACHER = 256
EPOCHS_FULL = 40
EPOCHS_CHEAP = 4
LR = 1e-3
# task C (CIFAR-10) reuses the CP-H caches: same locked config as cph_demo's calibrated run
CIFAR_WIDTH = 96
CIFAR_EPOCHS = 120
CIFAR_CHEAP_EPOCHS = 8
CIFAR_NN_MEMORY = 2000
CIFAR_NN_QUERY = 1000
CIFAR_PROBE_TRAIN = 10000
CIFAR_MEAN_STD = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
BASE_GRID_A = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
BASE_GRID_B = [2, 3, 4, 6, 8, 16, 32, 64]
TOP_EXT_A = [128, 192, 256]      # twin-student sizes (see docstring (3)); only used with --extend-top
TOP_EXT_B = [128, 256]
N_GROUPS = 8
NN_MEMORY = 4000

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "witness.jsonl")
LOG = os.path.join(_DATA, "witness.log")
REPORT = os.path.join(_DATA, "witness_report.txt")
CKPT_ROOT = os.path.join(_DATA, "witness")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        self.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logging():
    """Mirror stdout/stderr into the run log. Called only for real runs (the selftest writes nothing)."""
    global _LOG
    _LOG = open(LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Tasks (encoding mirrors cifar_demo.make_grid64_images; do not tune)
# ---------------------------------------------------------------------------

def _templates():
    rows0, cols = (1, 7, 12), (2, 6, 10, 14)
    t = torch.zeros(C_TASK, 1, 16, 16)
    for v in range(C_TASK):
        v1, rem = divmod(v, 16)
        v2, v3 = divmod(rem, 4)
        for c, vi in enumerate((v1, v2, v3)):
            r0, cx = rows0[c], cols[vi]
            t[v, 0, r0:r0 + 3, cx - 1:cx + 2] = 1.0
    return t


def make_task(task, n_per_class=N_PER_CLASS, seed=0):
    """task 'A' = clean grid64; task 'B' = merged grid64 (8 templates, 8 unlearnable duplicate labels each)."""
    g = torch.Generator().manual_seed(seed)
    templates = _templates()
    per_group = n_per_class * (C_TASK // N_GROUPS)
    if task == "A":
        y = torch.arange(C_TASK).repeat_interleave(n_per_class)
        used = templates[y]
    else:
        group = torch.arange(N_GROUPS).repeat_interleave(per_group)
        dup = torch.randint(0, C_TASK // N_GROUPS, (len(group),), generator=g)
        y = group * (C_TASK // N_GROUPS) + dup
        used = templates[group * (C_TASK // N_GROUPS)]
    bright = 1.0 + 0.1 * torch.randn(len(y), 1, 1, 1, generator=g)
    x = used * bright + 0.15 * torch.randn(len(y), 1, 16, 16, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# ---------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------

@torch.no_grad()
def witness_1nn(xtr, ytr, xte, yte, memory=NN_MEMORY):
    """W1: train -> test 1-NN on raw flattened inputs (zero training; a valid achieved error)."""
    Xm = xtr.flatten(1)[:memory].float()
    ym = ytr[:memory].long()
    Xq = xte.flatten(1).float()
    pred = ym[torch.cdist(Xq, Xm).argmin(dim=1)]
    return float((pred != yte.long()).float().mean().item())


def witness_linear(xtr, ytr, xte, yte):
    """W2: closed-form ridge linear probe on standardized raw inputs (seconds; an achieved error)."""
    Xtr = xtr.flatten(1).double()
    Xte = xte.flatten(1).double()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), C_TASK, dtype=torch.float64)
    Y[torch.arange(len(ytr)), ytr.long()] = 1.0
    d = Xtr.shape[1]
    lam = 1e-3 * float((Xtr ** 2).sum()) / d
    A = Xtr.t() @ Xtr + lam * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    pred = (Xte @ W).argmax(dim=1)
    return float((pred != yte.long()).float().mean().item())


def train_model(xtr, ytr, xte, yte, K, epochs, seed, device, ckpt_path=None, resume=True, teacher=None):
    """Train one GridCNN(K) under the standard protocol; returns (acc, model). Cache hit -> (acc, None)."""
    if ckpt_path is not None and resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ck["acc"], None
    torch.manual_seed(seed)
    tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    te = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = build_model("cnn", K, "euclid").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device, distill_from=teacher)
    acc, _ = evaluate(model, te, device=device)
    if ckpt_path is not None:
        torch.save({"acc": acc, "state_dict": model.state_dict()}, ckpt_path)
    return acc, model


def rebuild(ckpt_path, K, device):
    """Reload a cached teacher for KD. MUST be put in eval() mode: BatchNorm in train mode would use
    batch statistics and produce different soft labels than the original run (cache-hit reproducibility)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = build_model("cnn", K, "euclid").to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def is_monotone(vals, tol=0.01):
    return all(b >= a - tol for a, b in zip(vals, vals[1:]))


def witness_verdict(per_task):
    """Pre-registered verdict assembly (docstring (5)/(6)); per_task: {task: summary dict}."""
    detail, t1_all, t2_all, steep_any = [], True, True, False
    ladder = ["W1", "W2", "W3", "W4"]
    for task, r in per_task.items():
        t1 = all(is_monotone([s["ribs"][w] for w in ladder]) for s in r["seeds"])
        t2 = all(not s["false_excl_any"] for s in r["seeds"])
        cov_w1 = [s["cov"]["W1"] for s in r["seeds"]]
        cov_w4 = [s["cov"]["W4"] for s in r["seeds"]]
        steep = (sum(cov_w4) / len(cov_w4)) - (sum(cov_w1) / len(cov_w1)) >= 0.3
        steep_any = steep_any or steep
        t1_all, t2_all = t1_all and t1, t2_all and t2
        detail.append(f"  task {task}: T1 monotone R_IB = {'PASS' if t1 else 'FAIL'}; "
                      f"T2 soundness (no false exclusion) = {'PASS' if t2 else 'FAIL'}; "
                      f"coverage W1={cov_w1} W4={cov_w4} -> {'STEEP' if steep else 'SHALLOW'}")
        detail.append("           R_IB by witness (median): " +
                      ", ".join(f"{w}={r['rib_med'][w]:.2f}" for w in ["W1", "W2", "W3", "W4"]) +
                      f"; K_min W4={r['kmin_med']['W4']}; label-only target K_min={n_classes(task)}")
        detail.append(f"           label-only false exclusions (task {task}): {r['labelonly_false']}; "
                      f"strongest-witness false exclusions: {r['w4_false']}")
    if not t2_all:
        return "UNSOUND: a witness excluded a student that measurably succeeds (investigate)", detail
    if t1_all and steep_any:
        return ("SUPPORT: witness strength materially controls exclusion coverage; cheap witnesses stay "
                "sound but miss exclusions (the measured cost of skipping the witness)"), detail
    if t1_all:
        return ("SHALLOW: even the cheapest witnesses reach (near-)full coverage on these tasks -- "
                "good news for usability; separating them needs a harder task"), detail
    return "MIXED: monotonicity and/or coverage criteria not met (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def n_classes(task):
    """Class count per task (task C is CIFAR-10)."""
    return C_CIFAR if task == "C" else C_TASK


def assemble_row(task, seed, accs_w, accs_students, grid, fp, first_line=None):
    """Shared readout assembly for all tasks (docstring (4)/(5)); returns (row, record, lines)."""
    n = n_classes(task)
    acc4 = accs_w["W4"]
    gaps = {K: acc4 - accs_students[K] for K in grid}
    rel = {K: gaps[K] / max(acc4, 1e-9) for K in grid}
    succeeds = sorted(K for K in grid if rel[K] <= 0.10)
    fails = sorted(K for K in grid if rel[K] >= 0.50)
    ribs, kmins, excl, cov, fexcl = {}, {}, {}, {}, {}
    for w, a in accs_w.items():
        ribs[w] = task_info_fano(1.0 - a, n)
        kmins[w] = DistillCapacity().min_codebook_size(ribs[w])
        excl[w] = excluded_set(ribs[w], grid)
        cov[w] = coverage(excl[w], fails)
        fexcl[w] = false_exclusions(excl[w], succeeds)
    label_only_rib = math.log(n)
    label_only_excl = excluded_set(label_only_rib, grid)
    label_only_false = false_exclusions(label_only_excl, succeeds)
    row = dict(task=task, seed=seed, accs=accs_w, gaps=gaps, rel=rel, succeeds=succeeds, fails=fails,
               ribs=ribs, kmins=kmins, excl=excl, cov=cov, false_excl=fexcl,
               false_excl_any=any(bool(v) for v in fexcl.values()),
               label_only_excl=label_only_excl, label_only_false=label_only_false, fp=fp)
    record = dict(task=task, seed=seed, accs=accs_w, ribs=ribs, kmins=kmins, succeeds=succeeds,
                  fails=fails, gap_by_K=gaps, rel_by_K=rel, false_excl=fexcl,
                  label_only_false=label_only_false, fp=fp)
    lines = []
    if first_line:
        lines.append(first_line)
    lines.append("           R_IB " + "  ".join(f"{w}={ribs[w]:.2f}" for w in ribs) +
                 f"  label-only={label_only_rib:.2f}")
    lines.append("           K_min " + "  ".join(f"{w}={kmins[w]}" for w in kmins) +
                 f"  (label-only target K_min={n})")
    lines.append(f"           measured: succeeds={succeeds} fails={fails}")
    lines.append("           excluded sets " + "  ".join(f"{w}={excl[w]}" for w in excl))
    lines.append("           coverage " + "  ".join(f"{w}={cov[w]:.2f}" for w in cov) +
                 "; false exclusions " + "  ".join(f"{w}={fexcl[w]}" for w in fexcl))
    lines.append(f"           label-only rule excludes {label_only_excl} -> false exclusions "
                 f"{label_only_false}")
    return row, record, lines


# ---------------------------------------------------------------------------
# Task C (CIFAR-10): witnesses; the reference and the student grid come from the CP-H cache
# ---------------------------------------------------------------------------

def cifar_ckpt_dir():
    """The CP-H calibrated-run checkpoint directory (same fingerprint recipe as pa3_demo)."""
    fp = cph_run_signature(False, False, CIFAR_EPOCHS, None, CIFAR_WIDTH, K_TEACHER, LR, 4, True, "cuda")
    return os.path.join(_DATA, "cph", fp)


def cifar_cached_acc(kind, s, K=None):
    """Read the recorded accuracy of a cached CP-H unit (teacher or student)."""
    name = f"cifar_teacher_s{s}.pt" if kind == "teacher" else f"cifar_student_s{s}_K{K}.pt"
    path = os.path.join(cifar_ckpt_dir(), name)
    if not os.path.exists(path):
        return None
    return float(torch.load(path, map_location="cpu", weights_only=True)["acc"])


def cifar_raw_loaders(batch_size=256):
    """Un-augmented CIFAR-10 loaders for the cheap witnesses (1-NN / linear probe must see raw inputs)."""
    mean, std = CIFAR_MEAN_STD
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
    return _DL(tr, batch_size=batch_size, shuffle=False), _DL(te, batch_size=batch_size, shuffle=False)


def witness_1nn_cifar(tr_loader, te_loader, memory=CIFAR_NN_MEMORY, query=CIFAR_NN_QUERY, device="cpu"):
    """W1 on CIFAR-10: train -> test 1-NN on raw pixels (subsampled; zero training)."""
    xm, ym, xq, yq = [], [], [], []
    for x, y in tr_loader:
        xm.append(x.flatten(1)); ym.append(y)
        if sum(len(t) for t in ym) >= memory:
            break
    for x, y in te_loader:
        xq.append(x.flatten(1)); yq.append(y)
        if sum(len(t) for t in yq) >= query:
            break
    Xm = torch.cat(xm)[:memory].to(device).float()
    ym = torch.cat(ym)[:memory].long().to(device)
    Xq = torch.cat(xq)[:query].to(device).float()
    yq = torch.cat(yq)[:query].long().to(device)
    pred = ym[torch.cdist(Xq, Xm).argmin(dim=1)]
    return float((pred != yq).float().mean().item())


def witness_linear_cifar(tr_loader, te_loader, n_train=CIFAR_PROBE_TRAIN):
    """W2 on CIFAR-10: closed-form ridge on standardized raw pixels (a fixed train subset)."""
    xs, ys = [], []
    for x, y in tr_loader:
        xs.append(x.flatten(1)); ys.append(y)
        if sum(len(t) for t in ys) >= n_train:
            break
    Xtr = torch.cat(xs)[:n_train].double()
    ytr = torch.cat(ys)[:n_train].long()
    xs, ys = [], []
    for x, y in te_loader:
        xs.append(x.flatten(1)); ys.append(y)
    Xte = torch.cat(xs).double()
    yte = torch.cat(ys).long()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), C_CIFAR, dtype=torch.float64)
    Y[torch.arange(len(ytr)), ytr] = 1.0
    d = Xtr.shape[1]
    lam = 1e-3 * float((Xtr ** 2).sum()) / d
    A = Xtr.t() @ Xtr + lam * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    pred = (Xte @ W).argmax(dim=1)
    return float((pred != yte).float().mean().item())


def train_cifar_cheap(epochs, seed, device, ckpt_path, resume=True):
    """W3 on CIFAR-10: SmallCNN(width 96, K=256), a short budget (Adam 1e-3, no cosine)."""
    if resume and os.path.exists(ckpt_path):
        return float(torch.load(ckpt_path, map_location="cpu", weights_only=True)["acc"])
    torch.manual_seed(seed)
    tr, te = load_cifar(subset=None, workers=0)
    model = SmallCNN(d=64, n_codes=K_TEACHER, width=CIFAR_WIDTH).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device)
    acc, _ = evaluate(model, te, device=device)
    torch.save({"acc": acc, "state_dict": model.state_dict()}, ckpt_path)
    return acc


def run_task_c(seeds, cheap_epochs, resume, device, ckpt_dir):
    """Task C: witnesses W1/W2 (zero training) + W3 (short CNN) + W4 = the cached CP-H teacher;
    the measured truth (student grid) is read from the cached CP-H units (no training)."""
    cache_dir = cifar_ckpt_dir()
    acc_t_cached = [cifar_cached_acc("teacher", s) for s in range(seeds)]
    acc_s_cached = [{K: cifar_cached_acc("student", s, K) for K in CPH_GRID} for s in range(seeds)]
    missing = [s for s in range(seeds) if acc_t_cached[s] is None
               or any(v is None for v in acc_s_cached[s].values())]
    if missing:
        raise SystemExit(f"[WITNESS] task C needs the CP-H cache at {cache_dir}; missing seeds {missing}. "
                         f"Run experiments/cph_demo.py first (relay mode).")
    print("\n" + "=" * 84)
    print(f"[WITNESS] task C (CIFAR-10, CP-H cache) -- {seeds} seeds, student grid {CPH_GRID}")
    print("=" * 84)
    print(f"  [reference] W4 = the cached CP-H teacher (acc {acc_t_cached}), no training; "
          f"student accuracies read from the cache (no training)")
    tr_raw, te_raw = cifar_raw_loaders()
    err1 = witness_1nn_cifar(tr_raw, te_raw, device=device)
    err2 = witness_linear_cifar(tr_raw, te_raw)
    acc1, acc2 = 1.0 - err1, 1.0 - err2
    print(f"  [cheap witnesses] W1 1-NN raw acc={acc1:.3f} ; W2 linear probe raw acc={acc2:.3f}")
    rows = []
    for s in range(seeds):
        acc3 = train_cifar_cheap(cheap_epochs, 23000 + s, device,
                                 os.path.join(ckpt_dir, f"tC_w3_s{s}.pt"), resume)
        accs_w = {"W1": acc1, "W2": acc2, "W3": acc3, "W4": acc_t_cached[s]}
        accs_students = acc_s_cached[s]
        _, acc_gate, _ = g0_gate(accs_w["W4"], 0.0, K_TEACHER)
        row, record, lines = assemble_row(
            "C", s, accs_w, accs_students, CPH_GRID, "cph-cache",
            first_line=f"  seed {s}: witness acc " + "  ".join(f"{w}={accs_w[w]:.3f}" for w in accs_w) +
                       f"  (W4 acc gate: {'PASS' if acc_gate else 'FAIL'})")
        rows.append(row)
        jsonl_append(JSONL, record)
        for line in lines:
            print(line)
    return dict(
        seeds=rows,
        rib_med={w: sorted(r["ribs"][w] for r in rows)[len(rows) // 2] for w in ["W1", "W2", "W3", "W4"]},
        kmin_med={w: sorted(r["kmins"][w] for r in rows)[len(rows) // 2] for w in ["W1", "W2", "W3", "W4"]},
        labelonly_false=sum(len(r["label_only_false"]) for r in rows),
        w4_false=sum(len(r["false_excl"]["W4"]) for r in rows),
    )


def run_signature(tasks, epochs, grid_a, grid_b):
    raw = "|".join([f"t{''.join(tasks)}", f"ep{epochs}",
                    f"ga{'/'.join(str(k) for k in grid_a)}",
                    f"gb{'/'.join(str(k) for k in grid_b)}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def run_task(task, seeds, grid, epochs, resume, device, ckpt_dir, fp):
    print("\n" + "=" * 84)
    print(f"[WITNESS] task {task} ({'clean grid64' if task == 'A' else 'merged grid64'}) -- "
          f"{seeds} seeds, student grid {grid}")
    print("=" * 84)
    rows = []
    for s in range(seeds):
        (xtr, ytr), (xte, yte) = make_task(task, N_PER_CLASS, s)
        t0 = time.time()
        acc1 = 1.0 - witness_1nn(xtr, ytr, xte, yte)
        acc2 = 1.0 - witness_linear(xtr, ytr, xte, yte)
        acc3, _ = train_model(xtr, ytr, xte, yte, K_TEACHER, EPOCHS_CHEAP, 21000 + 100 * s, device,
                              os.path.join(ckpt_dir, f"t{task}_w3_s{s}.pt"), resume)
        w4_path = os.path.join(ckpt_dir, f"t{task}_w4_s{s}.pt")
        acc4, teacher = train_model(xtr, ytr, xte, yte, K_TEACHER, epochs, 21001 + 100 * s, device,
                                    w4_path, resume)
        if teacher is None:
            teacher = rebuild(w4_path, K_TEACHER, device)
        _, acc_gate, _ = g0_gate(acc4, 0.0, K_TEACHER)   # acc gate only (codebook gate N/A here)
        accs = {}
        for K in grid:
            acc_s, _ = train_model(xtr, ytr, xte, yte, K, epochs, 22000 + 100 * s + K, device,
                                   os.path.join(ckpt_dir, f"t{task}_st_s{s}_K{K}.pt"), resume,
                                   teacher=teacher)
            accs[K] = acc_s
        accs_w = {"W1": acc1, "W2": acc2, "W3": acc3, "W4": acc4}
        row, record, lines = assemble_row(
            task, s, accs_w, accs, grid, fp,
            first_line=f"  seed {s}: witness acc " + "  ".join(f"{w}={accs_w[w]:.3f}" for w in accs_w) +
                       f"  (W4 acc gate: {'PASS' if acc_gate else 'FAIL'})   [{time.time() - t0:.0f}s]")
        row["acc_gate"] = bool(acc_gate)
        rows.append(row)
        jsonl_append(JSONL, record)
        for line in lines:
            print(line)
    return dict(
        seeds=rows,
        rib_med={w: sorted(r["ribs"][w] for r in rows)[len(rows) // 2] for w in ["W1", "W2", "W3", "W4"]},
        kmin_med={w: sorted(r["kmins"][w] for r in rows)[len(rows) // 2] for w in ["W1", "W2", "W3", "W4"]},
        labelonly_false=sum(len(r["label_only_false"]) for r in rows),
        w4_false=sum(len(r["false_excl"]["W4"]) for r in rows),
    )


def run_all(tasks, seeds_a, seeds_b, seeds_c, epochs, cheap_epochs, resume, device, grid_a, grid_b):
    # fingerprint covers only the BASE grids (project convention, research-summary 10.4): a top extension must
    # reuse every cached base unit. grid_a/grid_b may be base + top extension.
    fp = run_signature(tasks, epochs, BASE_GRID_A, BASE_GRID_B)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[WITNESS] witness strength vs requirement-bound tightness; fingerprint {fp}")
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print("  [ladder] W1 1-NN raw (0 ep) / W2 linear probe (~0 ep) / W3 CNN short budget / "
          "W4 CNN 40 ep (the calibrated reference); W0 label-only = ln C")
    per_task = {}
    if "A" in tasks:
        per_task["A"] = run_task("A", seeds_a, grid_a, epochs, resume, device, ckpt_dir, fp)
    if "B" in tasks:
        per_task["B"] = run_task("B", seeds_b, grid_b, epochs, resume, device, ckpt_dir, fp)
    if "C" in tasks:
        per_task["C"] = run_task_c(seeds_c, cheap_epochs, resume, device, ckpt_dir)
    print("\n  --- WITNESS verdict ---")
    verdict, detail = witness_verdict(per_task)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[WITNESS] witness strength vs requirement-bound tightness\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return per_task, verdict


def verdict_from_jsonl():
    """Offline re-verdict: rebuild the per-task summary from witness.jsonl and run the pre-registered
    witness_verdict. No training, no logging, no file writes (zero-training replay).
    If both a base-row and an extended-row exist for a (task, seed), the LAST row wins (it carries the
    wider grid, so its measured sets are the superset)."""
    latest = {}
    for line in open(JSONL, encoding="utf-8"):
        r = json.loads(line)
        latest[(r["task"], r["seed"])] = r
    per_task = {}
    for r in latest.values():
        grid = sorted(int(k) for k in r["gap_by_K"])
        excl = {w: excluded_set(r["ribs"][w], grid) for w in ["W1", "W2", "W3", "W4"]}
        seed = dict(ribs=r["ribs"],
                    false_excl_any=any(bool(v) for v in r["false_excl"].values()),
                    cov={w: coverage(excl[w], r["fails"]) for w in excl})
        d = per_task.setdefault(r["task"], dict(seeds=[], rib_med={}, kmin_med={},
                                                labelonly_false=0, w4_false=0))
        d["seeds"].append(seed)
        d["labelonly_false"] += len(r["label_only_false"])
        d["w4_false"] += len(r["false_excl"]["W4"])
    for t, d in per_task.items():
        rows = [r for r in latest.values() if r["task"] == t]
        for w in ["W1", "W2", "W3", "W4"]:
            d["rib_med"][w] = sorted(s["ribs"][w] for s in d["seeds"])[len(d["seeds"]) // 2]
            d["kmin_med"][w] = sorted(r["kmins"][w] for r in rows)[len(rows) // 2]
    verdict, detail = witness_verdict(per_task)
    print("[WITNESS offline re-verdict] rebuilt from", JSONL)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    return per_task, verdict


def run_selftest():
    ok = []
    ok.append(("excluded_set uses log K < R_IB", excluded_set(2.0, [2, 4, 8, 16]) == [2, 4]))
    ok.append(("is_monotone accepts non-decreasing", is_monotone([1.0, 1.5, 1.5])))
    ok.append(("is_monotone rejects a decrease", not is_monotone([1.5, 1.0])))
    ok.append(("coverage counts only measured failures", coverage([2, 4, 8], [2, 4, 16]) == 2 / 3))
    ok.append(("false_exclusions finds the overlap", false_exclusions([2, 4], [4, 8]) == [4]))
    per_sound = {"A": dict(seeds=[dict(ribs={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                                       false_excl_any=False, cov={"W1": 0.2, "W4": 1.0})],
                           rib_med={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                           kmin_med={"W4": 8}, labelonly_false=0, w4_false=0)}
    v, _ = witness_verdict(per_sound)
    ok.append(("verdict SUPPORT when sound + steep", v.startswith("SUPPORT")))
    per_shallow = {"A": dict(seeds=[dict(ribs={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                                         false_excl_any=False, cov={"W1": 0.95, "W4": 1.0})],
                             rib_med={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                             kmin_med={"W4": 8}, labelonly_false=0, w4_false=0)}
    v, _ = witness_verdict(per_shallow)
    ok.append(("verdict SHALLOW when coverage flat", v.startswith("SHALLOW")))
    per_bad = {"A": dict(seeds=[dict(ribs={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                                     false_excl_any=True, cov={"W1": 0.2, "W4": 1.0})],
                         rib_med={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                         kmin_med={"W4": 8}, labelonly_false=0, w4_false=0)}
    v, _ = witness_verdict(per_bad)
    ok.append(("verdict UNSOUND when a false exclusion appears", v.startswith("UNSOUND")))
    print("[WITNESS selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="WITNESS: witness strength vs requirement-bound tightness")
    ap.add_argument("--task", type=str, default="both", choices=["A", "B", "C", "both", "all"],
                    help="both = A+B (default); all = A+B+C (C = CIFAR-10, needs the CP-H cache)")
    ap.add_argument("--seeds-a", type=int, default=3)
    ap.add_argument("--seeds-b", type=int, default=2)
    ap.add_argument("--seeds-c", type=int, default=3, help="task C seeds (capped by the CP-H cache)")
    ap.add_argument("--cifar-cheap-epochs", type=int, default=CIFAR_CHEAP_EPOCHS,
                    help="W3 budget for task C (the short-budget CNN witness)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_FULL)
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--extend-top", action="store_true",
                    help="append twin-student sizes to the student grids (amendment; reuses base cache)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    ap.add_argument("--verdict-only", action="store_true",
                    help="rebuild the verdict from witness.jsonl (no training, no file writes)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_from_jsonl()
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tasks = {"both": ["A", "B"], "all": ["A", "B", "C"]}.get(args.task, [args.task])
    grid_a = BASE_GRID_A + TOP_EXT_A if args.extend_top else BASE_GRID_A
    grid_b = BASE_GRID_B + TOP_EXT_B if args.extend_top else BASE_GRID_B
    if args.extend_top:
        print(f"[WITNESS] top extension ON: grid A -> {grid_a} ; grid B -> {grid_b} "
              f"(base units reused from cache)")
    run_all(tasks, args.seeds_a, args.seeds_b, args.seeds_c, args.epochs,
            args.cifar_cheap_epochs, not args.fresh, device, grid_a, grid_b)
    print("\nDone.")


if __name__ == "__main__":
    main()
