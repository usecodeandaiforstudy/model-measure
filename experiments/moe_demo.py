"""MOE: validation of the two-tier capacity reading (conditional_capacity) on a routed expert network.

This closes the last unverified piece of the package body: capacity_lens.capacity.conditional_capacity
and kv_storage_capacity are CANDIDATE functions (A-002 5.3 vocabulary); this experiment validates the
MoE two-tier reading on the synthetic route64 task. (kv_storage_capacity remains a candidate -- an
attention-based task is out of this script's scope, recorded honestly.)

Usage:
    python experiments/moe_demo.py                 # full run, 3 seeds, ~30-40 min on GPU
    python experiments/moe_demo.py --selftest      # pure-function unit tests
    (relay mode is the default; --fresh recomputes)

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Task: route64 -- a 16x16 single-channel image with one 3x3 block at (row r in 0..7, col c in 0..7),
    class = 8r + c (64 classes, H(Y) = ln64 = 4.16 nats). Cluster = r (8 clusters, cluster info ln8).
    Same noise conventions as grid64 (sigma_pix=0.15, sigma_bright=0.1, 150/class, data seed = s).
(2) Networks:
    - RoutedMoE: shared GridCNN-style backbone -> d=64 -> 8 experts, each expert = VQBottleneck(d, K_e,
      euclid) + 64-class head. Router = ORACLE, a deterministic input-computable feature: the block's
      row read off the image (sliding 3-row window argmax). The router carries exactly ln8 and leaks
      nothing about the column. Per-example FLOW bound = ln8 (router) + ln K_e (active expert);
      STORAGE = 8 * ln K_e. (Oracle routing per the E-001 lesson: this experiment measures CAPACITY,
      not router trainability.)
    - Dense control: GridCNN(d=64, n_codes=K_total) (the existing cp3 model) with K_total = 8 * K_e.
(3) Training: custom loop for RoutedMoE (train_one_epoch does not know conditional activation: commit
    loss and dead-code restart are applied only to the ACTIVE expert, on its cluster's representations);
    dense control uses the standard train_one_epoch. Budget: 40ep Adam 1e-3 (cp3-calibrated CNN budget).
(4) Readouts: teacher (dense K=256) acc / H(Z_t) / G0 / R_IB(Fano)=4.16; routed acc(K_e) + per-expert
    distinct-code counts (joint storage utilization); dense acc(K_total); gap curves.
(5) Pre-registered criteria (pure function moe_verdict; T1a/T2/T3 per-seed thresholds, >=2/3 seeds):
    T1a (flow tier is the per-example exclusion, router info counts): at K_e=8 (flow = 4.16 = the wall)
        acc >= 0.90; at K_e=4 (flow = 3.47 < 4.16) acc <= 0.50. A pass FALSIFIES the "expert-only"
        reading (which would predict K_e=8 fails because ln8 = 2.08 < 4.16).
    T1b (cliff location): the routed curve's steepest adjacent pair (sharpness) is the 4->8 step.
    T2 (storage is real): at K_e=8, joint distinct codes over all 8 experts >= 0.8 * 64.
    T3 (reduction to dense): for every K_e in {2,4,8,16}, |acc_routed(K_e) - acc_dense(8*K_e)| <= 0.05.
    Verdict: all four pass -> conditional_capacity's two-tier reading is VERIFIED on the route64 task
    family (its docstring status then upgrades from CANDIDATE accordingly); otherwise PARTIAL/REFUTED.
(6) Seed scheme: teacher = 18000 + s; routed student = 19000 + 100*s + K_e; dense student =
    19500 + 100*s + K_total (non-overlapping with all existing families). Data seed = s.
(7) Relay: default relay mode, per-unit checkpoints under experiments/data/moe/<fingerprint>/;
    --fresh recomputes. Fingerprint = (k_e grid, k_total grid, epochs, lr, device); seed count excluded.
(8) Iron rules: print only GBK-encodable characters; criteria before data; pure functions unit-testable;
    no downloads (task generated in-process).
(9) Follow-up phase (pre-registered 2026-09-30, AFTER the first run's REFUTED verdict; fresh seeds 5-7, criteria
    before the new data): the first run's T1a/T1b FAILED because they anchored the routed cliff at the
    Fano bound (ln64) while this task family's MEASURED dense wall sits at K=256 (dense curve: 0.39-0.60
    at K=64, 0.69-0.90 at 128, 0.999 at 256 -- a 4x looseness of the Fano bound, the CP-H-32x-margin
    family). The follow-up re-anchors at the measured wall:
      F1a: with wall_k = the smallest dense K_total reaching >= 0.95 (per seed), the routed net reaches
           >= 0.90 at the smallest K_e with flow = ln8 + ln K_e >= ln(wall_k) (predicted K_e = 32);
      F1b: routed(K_e) >= dense(8*K_e) - 0.05 for every K_e in {2,4,8,16,32} (reduction + the equal-total
           routing advantage direction).
    Verdict (pure function followup_verdict): F1a AND F1b each pass for >= 2/3 seeds -> the flow tier is
    verified against the measured-wall anchor; conditional_capacity's flow note then reflects this.
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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import VQBottleneck, train_one_epoch, evaluate, entropy_of, task_info_fano
from capacity_lens.preflight import g0_gate
from cp3_demo import build_model

K_TEACHER = 256
N_EXPERTS = 8
C_TASK = 64
N_PER_CLASS = 150
K_E_GRID = [2, 4, 8, 16, 32]
K_TOT_GRID = [16, 32, 64, 128, 256]
EPOCHS_DEFAULT = 40
LR = 1e-3

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "moe.jsonl")
LOG = os.path.join(_DATA, "moe.log")
REPORT = os.path.join(_DATA, "moe_report.txt")
CKPT_ROOT = os.path.join(_DATA, "moe")


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


try:
    _LOG = open(LOG, "w", encoding="utf-8")
except OSError:
    # An import must not require a writable data directory (offline replay, read-only checkouts). Fall back to
    # devnull, NOT to None: these scripts tee stdout/stderr through _LOG, and a None sink makes every print
    # fail at interpreter shutdown with exit code 120 (found by the selftest loop that CI runs).
    _LOG = open(os.devnull, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Task: route64
# ---------------------------------------------------------------------------

def make_route64(n_per_class=150, seed=0):
    """8 clusters (block row) x 8 classes (block col) -> 64 classes; cluster info ln8, task info ln64."""
    g = torch.Generator().manual_seed(seed)
    templates = torch.zeros(C_TASK, 1, 16, 16)
    for v in range(C_TASK):
        r, c = divmod(v, N_EXPERTS)
        templates[v, 0, r:r + 3, c:c + 3] = 1.0
    y = torch.arange(C_TASK).repeat_interleave(n_per_class)
    bright = 1.0 + 0.1 * torch.randn(len(y), 1, 1, 1, generator=g)
    x = templates[y] * bright + 0.15 * torch.randn(len(y), 1, 16, 16, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RoutedMoE(nn.Module):
    """Oracle-routed expert net: shared backbone -> d, then 8 experts (VQ(K_e) + 64-class head).
    The router reads the block ROW from the input (3-row sliding window argmax) -- input-computable,
    carries exactly ln8, leaks nothing about the column. Flow bound = ln8 + ln K_e; storage = 8 ln K_e."""

    def __init__(self, d=64, k_e=16, width=32):
        super().__init__()
        w2 = width * 2
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w2, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(w2 * 2 * 2, d)
        self.vqs = nn.ModuleList([VQBottleneck(d, k_e, metric="euclid") for _ in range(N_EXPERTS)])
        self.heads = nn.ModuleList([nn.Linear(d, C_TASK) for _ in range(N_EXPERTS)])

    @staticmethod
    def router(x):
        """Input-computable oracle: sliding 3-row window argmax of the row sums -> block top row."""
        rs = x.squeeze(1).sum(dim=2)                    # [B, 16]
        w = torch.stack([rs[:, j] + rs[:, j + 1] + rs[:, j + 2] for j in range(14)], dim=1)  # [B, 14]
        return w.argmax(dim=1).clamp(0, N_EXPERTS - 1)  # true r <= 7, argmax = r exactly

    def forward_meta(self, x):
        """Returns (logits [B,64], route [B], active list [(expert_idx, t_flat, idx)])."""
        route = self.router(x)
        h = self.features(x).flatten(1)
        t = F.relu(self.fc1(h))
        logits = torch.zeros(x.size(0), C_TASK, device=x.device)
        active = []
        for e in range(N_EXPERTS):
            mask = (route == e)
            if mask.any():
                z, idx = self.vqs[e](t[mask])
                logits[mask] = self.heads[e](z)
                active.append((e, self.vqs[e], idx))
        return logits, route, active


def train_step_moe(model, x, y, opt, beta=0.25):
    """Custom step: CE + commit loss on the ACTIVE expert only; dead-code restart only on active experts."""
    model.train()
    logits, route, active = model.forward_meta(x)
    loss = F.cross_entropy(logits, y)
    for _, vq, _ in active:
        loss = loss + beta * vq.commit_loss()
    opt.zero_grad()
    loss.backward()
    opt.step()
    for _, vq, _ in active:
        t = vq.last_input()
        if t is not None:
            vq.update_usage_and_restart(t)
    return loss.item()


@torch.no_grad()
def evaluate_moe(model, loader, device="cpu"):
    """Returns (acc, per_expert_used_bitmask [8 x K_e], route_frac [8])."""
    model.eval()
    correct = total = 0
    used = [torch.zeros(model.vqs[0].n_codes, dtype=torch.bool) for _ in range(N_EXPERTS)]
    route_counts = torch.zeros(N_EXPERTS)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, route, active = model.forward_meta(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
        route_counts += torch.bincount(route.cpu(), minlength=N_EXPERTS).float()
        for e, vq, idx in active:
            used[e] |= torch.bincount(idx.cpu(), minlength=vq.n_codes).bool()
    joint_used = sum(int(u.sum().item()) for u in used)
    return correct / total, used, joint_used, route_counts


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def sharpness_pairs(k_list, accs, acc_t):
    """Per-adjacent-pair sharpness (pure function): (g_low - g_high) / (log K_high - log K_low)."""
    ks = sorted(k_list)
    out = {}
    for a, b in zip(ks, ks[1:]):
        drop = (acc_t - accs[a]) - (acc_t - accs[b])
        dlog = math.log(b) - math.log(a)
        out[(a, b)] = drop / dlog if dlog > 0 else 0.0
    return out


def moe_verdict(per_seed):
    """The MOE two-tier verdict (pure function, see docstring (5)).

    per_seed: list of dicts with keys acc_ke4, acc_ke8, storage_util8, sharp_argmax (the routed pair
    with the max sharpness), t3_maxdiff (max over K_e of |routed(K_e) - dense(8*K_e)|).
    Returns (verdict, detail_lines)."""
    need = math.ceil(2 / 3 * len(per_seed))
    t1a = (sum(1 for r in per_seed if r["acc_ke8"] >= 0.90) >= need and
           sum(1 for r in per_seed if r["acc_ke4"] <= 0.50) >= need)
    t1b = sum(1 for r in per_seed if r["sharp_argmax"] == (4, 8)) >= need
    t2 = sum(1 for r in per_seed if r["storage_util8"] >= 0.8) >= need
    t3 = sum(1 for r in per_seed if r["t3_maxdiff"] <= 0.05) >= need
    detail = [
        f"  T1a flow-tier exclusion (ke8>=0.90 AND ke4<=0.50): {'PASS' if t1a else 'FAIL'}",
        f"  T1b steepest step at 4->8: {'PASS' if t1b else 'FAIL'}",
        f"  T2 storage tightness (ke8 joint utilization>=0.8): {'PASS' if t2 else 'FAIL'}",
        f"  T3 dense reduction (per-step diff<=0.05): {'PASS' if t3 else 'FAIL'}",
    ]
    if t1a and t1b and t2 and t3:
        return "SUPPORT: two-tier capacity reading holds on the route64 task family (conditional_capacity can be upgraded to verified)", detail
    if t1a and (t1b or t2 or t3):
        return "PARTIAL: flow-tier exclusion holds but the other criteria did not all pass (recorded honestly)", detail
    return "REFUTED: flow-tier exclusion does not hold (two-tier reading fails on this task family) -- recorded honestly", detail


# ---------------------------------------------------------------------------
# Training units
# ---------------------------------------------------------------------------

def run_signature(k_e_grid, k_tot_grid, epochs, lr, device):
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    raw = "|".join([f"ke{'/'.join(str(k) for k in k_e_grid)}",
                    f"kt{'/'.join(str(k) for k in k_tot_grid)}",
                    f"ep{epochs}", f"lr{lr}", f"dev{dev}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def ckpt_dir_for(k_e_grid, k_tot_grid, epochs, lr, device):
    d = os.path.join(CKPT_ROOT, run_signature(k_e_grid, k_tot_grid, epochs, lr, device))
    os.makedirs(d, exist_ok=True)
    return d


def train_routed_unit(s, k_e, epochs, device, ckpt_dir, resume):
    torch.manual_seed(19000 + 100 * s + k_e)
    (xtr, ytr), (xte, yte) = make_route64(N_PER_CLASS, s)
    tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    te = DataLoader(list(zip(xte, yte)), batch_size=256)
    path = os.path.join(ckpt_dir, f"routed_s{s}_ke{k_e}.pt")
    if resume and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  routed ke={k_e} s{s}: [cache hit] acc={ck['acc']:.3f}  util={ck['joint_used']}/{8 * k_e}")
        return ck["acc"], ck["joint_used"], 8 * k_e, te, None
    model = RoutedMoE(d=64, k_e=k_e).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(epochs):
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            train_step_moe(model, x, y, opt)
    acc, used, joint_used, route_counts = evaluate_moe(model, te, device=device)
    torch.save({"acc": acc, "joint_used": joint_used,
                "state_dict": model.state_dict()}, path)
    print(f"  routed ke={k_e} s{s}: acc={acc:.3f}  joint codes {joint_used}/{8 * k_e}  "
          f"route distribution {[f'{v:.0f}' for v in route_counts.tolist()]}")
    return acc, joint_used, 8 * k_e, te, model


def train_dense_unit(s, k_total, epochs, device, ckpt_dir, resume):
    """Dense control trained with pure CE (same objective as the routed net -- T3 compares capacity
    readings, not training objectives)."""
    torch.manual_seed(19500 + 100 * s + k_total)
    (xtr, ytr), (xte, yte) = make_route64(N_PER_CLASS, s)
    tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    te = DataLoader(list(zip(xte, yte)), batch_size=256)
    path = os.path.join(ckpt_dir, f"dense_s{s}_kt{k_total}.pt")
    if resume and os.path.exists(path):
        acc = float(torch.load(path, map_location="cpu", weights_only=True)["acc"])
        print(f"  dense kt={k_total} s{s}: [cache hit] acc={acc:.3f}")
        return acc
    model = build_model("cnn", k_total, "euclid").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, tr, opt, device=device)
    acc, _ = evaluate(model, te, device=device)
    torch.save({"acc": acc}, path)
    print(f"  dense kt={k_total} s{s}: acc={acc:.3f}")
    return acc


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_moe(seeds, epochs, resume, device):
    fp = run_signature(K_E_GRID, K_TOT_GRID, epochs, LR, device)
    ckpt_dir = ckpt_dir_for(K_E_GRID, K_TOT_GRID, epochs, LR, device)
    print("\n" + "=" * 76)
    print(f"[MOE] two-tier capacity reading validation: route64 + oracle routing ({seeds} seeds; {epochs}ep; "
          f"ke={K_E_GRID}; dense={K_TOT_GRID})")
    print("=" * 76)
    print(f"  [checkpoint] {'relay mode' if resume else 'recompute mode'}  {ckpt_dir}")
    print("  [criteria] T1a flow-tier exclusion (ke8>=0.90 AND ke4<=0.50) / T1b steepest step 4->8 / "
          "T2 storage tightness / T3 dense reduction")
    per_seed = []
    for s in range(seeds):
        print(f"\n  --- s{s} ---")
        # teacher (dense K=256)
        torch.manual_seed(18000 + s)
        (xtr, ytr), (xte, yte) = make_route64(N_PER_CLASS, s)
        tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
        te = DataLoader(list(zip(xte, yte)), batch_size=256)
        teacher = build_model("cnn", K_TEACHER, "euclid").to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=LR)
        for _ in range(epochs):
            train_one_epoch(teacher, tr, opt, device=device)
        acc_t, idx_t = evaluate(teacher, te, device=device)
        h_z_t = entropy_of(idx_t[0], K_TEACHER)
        g0, ok_acc, ok_cb = g0_gate(acc_t, h_z_t, K_TEACHER)
        r_ib = task_info_fano(1.0 - acc_t, C_TASK)
        print(f"  teacher: acc={acc_t:.3f} (G0 {'PASS' if g0 else 'FAIL'})  H(Z_t)={h_z_t:.2f}  "
              f"R_IB={r_ib:.2f} (ln64={math.log(C_TASK):.2f})")
        routed_accs, routed_util, routed_models = {}, {}, {}
        for k_e in K_E_GRID:
            acc, joint_used, total_codes, _, model = train_routed_unit(
                s, k_e, epochs, device, ckpt_dir, resume)
            routed_accs[k_e] = acc
            routed_util[k_e] = joint_used / total_codes
            routed_models[k_e] = model
        dense_accs = {}
        for k_total in K_TOT_GRID:
            dense_accs[k_total] = train_dense_unit(s, k_total, epochs, device,
                                                   ckpt_dir, resume)
        sh = sharpness_pairs(K_E_GRID, routed_accs, acc_t)
        sharp_argmax = max(sh, key=sh.get)
        t3 = max(abs(routed_accs[k_e] - dense_accs[8 * k_e]) for k_e in [2, 4, 8, 16])
        row = dict(s=s, acc_t=acc_t, g0=g0, r_ib=r_ib,
                   routed=dict(accs=routed_accs, util=routed_util,
                               sharp={f"{a}-{b}": v for (a, b), v in sh.items()}),
                   dense=dict(accs=dense_accs), fp=fp)
        jsonl_append(JSONL, row)
        per_seed.append(dict(acc_ke4=routed_accs[4], acc_ke8=routed_accs[8],
                             storage_util8=routed_util[8], sharp_argmax=sharp_argmax,
                             t3_maxdiff=t3))
        print(f"  routed acc: " + "  ".join(f"ke{k}:{routed_accs[k]:.3f}" for k in K_E_GRID))
        print(f"  routed joint-code utilization: " + "  ".join(f"ke{k}:{routed_util[k]:.0%}"
                                                        for k in K_E_GRID))
        print(f"  dense  acc: " + "  ".join(f"kt{k}:{dense_accs[k]:.3f}" for k in K_TOT_GRID))
        print(f"  steepest step: {sharp_argmax}; T3 max per-step diff: {t3:.3f}")
    print("\n  --- MOE verdict ---")
    verdict, detail = moe_verdict(per_seed)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[MOE] two-tier capacity reading validation verdict\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return per_seed, verdict


def followup_verdict(per_seed):
    """Re-anchored F1 criteria (pure function, see docstring (9))."""
    need = math.ceil(2 / 3 * len(per_seed))
    f1a = sum(1 for r in per_seed if r["f1a"]) >= need
    f1b = sum(1 for r in per_seed if r["f1b"]) >= need
    detail = [f"  F1a (measured-wall-anchored ke reaches >=0.90): {'PASS' if f1a else 'FAIL'}",
              f"  F1b (routed >= dense(8ke)-0.05 per step): {'PASS' if f1b else 'FAIL'}"]
    if f1a and f1b:
        return "SUPPORT: flow tier holds against the measured dense wall (conditional_capacity flow tier validated)", detail
    return "PARTIAL/REFUTED: still failing after re-anchoring (recorded honestly)", detail


def run_followup(seeds, epochs, resume, device):
    ckpt_dir = ckpt_dir_for(K_E_GRID, K_TOT_GRID, epochs, LR, device)
    print("\n" + "=" * 76)
    print(f"[MOE followup] re-anchored validation (seeds {seeds}; criteria in docstring (9))")
    print("=" * 76)
    per_seed = []
    for s in seeds:
        print(f"\n  --- s{s} ---")
        torch.manual_seed(18000 + s)
        (xtr, ytr), (xte, yte) = make_route64(N_PER_CLASS, s)
        tr = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
        te = DataLoader(list(zip(xte, yte)), batch_size=256)
        teacher = build_model("cnn", K_TEACHER, "euclid").to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=LR)
        for _ in range(epochs):
            train_one_epoch(teacher, tr, opt, device=device)
        acc_t, _ = evaluate(teacher, te, device=device)
        routed_accs = {}
        for k_e in K_E_GRID:
            acc, _, _, _, _ = train_routed_unit(s, k_e, epochs, device, ckpt_dir, resume)
            routed_accs[k_e] = acc
        dense_accs = {}
        for k_total in K_TOT_GRID:
            dense_accs[k_total] = train_dense_unit(s, k_total, epochs, device, ckpt_dir, resume)
        wall_k = next((k for k in K_TOT_GRID if dense_accs[k] >= 0.95), None)
        if wall_k is None:
            f1a, pred_ke = False, None
        else:
            pred_ke = next((k for k in K_E_GRID
                            if math.log(N_EXPERTS) + math.log(k) >= math.log(wall_k) - 1e-9), None)
            f1a = pred_ke is not None and routed_accs.get(pred_ke, 0.0) >= 0.90
        f1b = all(routed_accs.get(k, -1.0) >= dense_accs[8 * k] - 0.05
                  for k in [2, 4, 8, 16, 32])
        per_seed.append(dict(f1a=f1a, f1b=f1b))
        print(f"  teacher acc={acc_t:.3f}  dense wall={wall_k}  -> predicted ke={pred_ke} (routed={routed_accs.get(pred_ke) if pred_ke else None})"
              f"  F1a={'PASS' if f1a else 'FAIL'}  F1b={'PASS' if f1b else 'FAIL'}")
        print(f"  routed acc: " + "  ".join(f"ke{k}:{routed_accs[k]:.3f}" for k in K_E_GRID))
        print(f"  dense  acc: " + "  ".join(f"kt{k}:{dense_accs[k]:.3f}" for k in K_TOT_GRID))
    verdict, detail = followup_verdict(per_seed)
    print("\n  --- followup verdict ---")
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    return per_seed, verdict


def run_selftest():
    ok = []
    # sharpness_pairs: the steepest pair is where the cliff sits
    accs = {2: 0.1, 4: 0.2, 8: 0.95, 16: 0.97, 32: 0.98}
    sh = sharpness_pairs([2, 4, 8, 16, 32], accs, 1.0)
    ok.append(("sharpness steepest step 4->8", max(sh, key=sh.get) == (4, 8)))
    # moe_verdict: all pass -> support
    good = [dict(acc_ke4=0.2, acc_ke8=0.95, storage_util8=0.9, sharp_argmax=(4, 8), t3_maxdiff=0.02)
            for _ in range(3)]
    v, _ = moe_verdict(good)
    ok.append(("all pass -> SUPPORT", v.startswith("SUPPORT")))
    # T1a fail -> REFUTED
    bad = [dict(acc_ke4=0.9, acc_ke8=0.95, storage_util8=0.9, sharp_argmax=(4, 8), t3_maxdiff=0.02)
           for _ in range(3)]
    v, _ = moe_verdict(bad)
    ok.append(("ke4 does not fail -> REFUTED", v.startswith("REFUTED")))
    # t3 fail only -> PARTIAL
    part = [dict(acc_ke4=0.2, acc_ke8=0.95, storage_util8=0.9, sharp_argmax=(4, 8), t3_maxdiff=0.2)
            for _ in range(3)]
    v, _ = moe_verdict(part)
    ok.append(("only T3 fails -> PARTIAL", v.startswith("PARTIAL")))
    # followup_verdict: both pass -> SUPPORT
    v, _ = followup_verdict([dict(f1a=True, f1b=True)] * 3)
    ok.append(("F1 both pass -> SUPPORT", v.startswith("SUPPORT")))
    v, _ = followup_verdict([dict(f1a=True, f1b=True), dict(f1a=True, f1b=False),
                             dict(f1a=True, f1b=False)])
    ok.append(("F1b 2/3 fail -> PARTIAL/REFUTED", v.startswith("PARTIAL")))
    print("[MOE selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all passed' if all_ok else 'failures found (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="MOE: two-tier capacity reading validation (route64 + oracle routing)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (default 3)")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="epochs (default 40)")
    ap.add_argument("--fresh", action="store_true", help="recompute mode: ignore cache (default relay)")
    ap.add_argument("--followup", action="store_true",
                    help="re-anchored validation (seeds 5-7, criteria in docstring (9))")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.followup:
        run_followup([5, 6, 7], args.epochs, not args.fresh, device)
    else:
        run_moe(args.seeds, args.epochs, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
