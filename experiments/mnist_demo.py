"""Capacity Lens validation experiments: P1/P2/P3/P4 + feature A (quantization) + feature B (distillation) + Fisher curve.

Usage:
    python experiments/mnist_demo.py              # all experiments (about 20 minutes on CPU)
    python experiments/mnist_demo.py --quick      # quick smoke test (about 2 minutes, no verdict)
    python experiments/mnist_demo.py --fast       # reduced budget but **outputs verdicts**
    python experiments/mnist_demo.py --only p4    # run a single experiment
    python experiments/mnist_demo.py --only quant --seeds 3   # multi-seed stability rerun for feature A/B
    python experiments/mnist_demo.py --only qat --seeds 3     # QAT vs PTQ three-arm comparison (artifact removal)
    python experiments/mnist_demo.py --only distill-mnist     # feature B v2 (MNIST archived protocol)

The success/failure criteria of each experiment are written in their respective verdict functions.

Output character convention: all **print content uses only GBK-encodable characters** (Chinese, ≤ ≥ × → are all fine, but
the U+2212 minus sign `−`, U+2713/2717 `✓✗`, etc. are not). Otherwise, redirecting the output to a file on Windows
will hard-crash with UnicodeEncodeError——this pitfall has already been stepped into twice.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Allow the script to import the parent package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from capacity_lens import (VQBottleneck, FisherEffectiveDim,
                           train_one_epoch, evaluate, entropy_of,
                           task_info_fano, joint_code,
                           discrete_capacity_logK, min_capacity_serial,
                           sum_capacity_parallel)
# Judgment pure functions moved 2026-09-30 to capacity_lens.preflight (the canonical home);
# re-exported here so every existing experiment script keeps importing from mnist_demo unchanged.
from capacity_lens.preflight import (g0_gate, distill_plateau_ok,
                                     distill_gap_k_star, ratio_verdict)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

# Data placement (verified directory layout; the tools do not download from the network):
#   experiments/data/MNIST/raw/train-images-idx3-ubyte   (+ the other 3 idx files)
# Note: this torchvision version reads the raw idx files directly, no processed/*.pt needed;
#       download=True returns immediately when the files already exist (offline-safe, verified no network calls).


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist(batch_size=128, subset=None):
    """MNIST. subset: fraction of the training set to keep (used in quick mode)."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    if subset is not None:
        n_keep = int(len(train) * subset)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train), generator=g)[:n_keep]
        train = Subset(train, idx.tolist())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_grid64(n_per_class=150, sigma=0.5, spacing=3.0, seed=0):
    """Synthetic task: a 4×4×4 grid 64-class Gaussian mixture with 3-dimensional input.

    Task information ≈ log 64 = 4.16 nats = 6 bits.
    Why not use MNIST for quantization validation: the MNIST task only carries
    log 10 ≈ 3.3 bits of information, so for a d=32 bottleneck q_min = 2^(3.3/32) ≈ 1.07
    and the lower bound degenerates (the formula is only meaningful for
    "small dimension, large information"). This task has R=6 bits, d=3 -> q_min = 2^2 = 4 levels.
    """
    g = torch.Generator().manual_seed(seed)
    coords = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(4)).float()
    centers = coords * spacing                       # [64, 3]
    n_classes = 64
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    x = centers[y] + sigma * torch.randn(len(y), 3, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Linear(784->d) -> ReLU -> VQBottleneck(d, K) -> Linear(d->10)."""

    def __init__(self, d, n_codes):
        super().__init__()
        self.fc1 = nn.Linear(784, d)
        self.vq = VQBottleneck(d, n_codes)
        self.fc2 = nn.Linear(d, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        z, idx = self.vq(h)
        return self.fc2(z), [idx]


class DeepVQNet(nn.Module):
    """Three narrow VQ bottlenecks in series, each with K_each codewords (total combinations K_each^3).

    Used for P2: compared against a single wide bottleneck (K = K_each^3) at the same parameter scale.
    DPI guarantees I(X; Z3) <= capacity of the min segment = log K_each.
    """

    def __init__(self, d, k_each):
        super().__init__()
        self.fc1 = nn.Linear(784, d)
        self.vq1 = VQBottleneck(d, k_each)
        self.fc2 = nn.Linear(d, d)
        self.vq2 = VQBottleneck(d, k_each)
        self.fc3 = nn.Linear(d, d)
        self.vq3 = VQBottleneck(d, k_each)
        self.fc4 = nn.Linear(d, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x)); z1, i1 = self.vq1(h1)
        h2 = F.relu(self.fc2(z1)); z2, i2 = self.vq2(h2)
        h3 = F.relu(self.fc3(z2)); z3, i3 = self.vq3(h3)
        return self.fc4(z3), [i1, i2, i3]


class Classifier3D(nn.Module):
    """Synthetic-task model: [VQ bottleneck (optional)] -> Linear(3->64).

    Design notes (lessons from the failed first version; both were capacity/alignment problems):
    - No ReLU hidden layer: 3 ReLU units can span at most 8 linear regions, which cannot
      distinguish 64 classes at all (measured acc≈0.17)——the bottleneck itself lacks capacity;
    - No learnable fc1: a randomly initialized linear map would rotate the grid, making the
      axis-aligned binning of per-dimension quantization misaligned with the clusters, and
      confounding the capacity conclusion under test. Linear(3->64) can already
      implement a nearest-centroid classifier (argmax is affinely realizable), which suffices for this task.
    """

    def __init__(self, with_vq=False, n_codes=256):
        super().__init__()
        self.vq = VQBottleneck(3, n_codes, metric="euclid") if with_vq else None
        self.fc = nn.Linear(3, 64)

    def forward(self, x):
        if self.vq is not None:
            z, idx = self.vq(x)
            return self.fc(z), [idx]
        return self.fc(x), None


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Quantizer (training/evaluation/entropy and other shared tools live in capacity_lens.training)
# ---------------------------------------------------------------------------

class PercentileQuantizer:
    """Per-dimension q-level percentile quantization (a post-training calibration style, the simplest
    form of TensorRT percentile calibration): the edges are the k/q quantiles of each dimension of the
    training distribution, and the reconstruction values are the per-level empirical means (one Lloyd step).

    The capacity is still q^d distinguishable states (d·log2 q bits), and the q_min formula is unchanged.
    Compared with min/max uniform quantization, percentile edges can align with the valleys of a
    multimodal distribution, avoiding the alignment failure of "bin boundaries landing near cluster centers".
    """

    def __init__(self, t_train, q):
        self.q = q
        d = t_train.shape[1]
        if q > 1:
            probs = torch.arange(1, q, dtype=t_train.dtype) / q
            self.edges = torch.quantile(t_train, probs, dim=0)   # [q-1, d]
        else:
            self.edges = torch.zeros(0, d)
        # The per-dimension, per-level empirical mean of the training set serves as the reconstruction value
        levels = self._bucketize(t_train)                        # [N, d]
        self.rec = torch.zeros(q, d)
        for j in range(d):
            for l in range(q):
                mask = levels[:, j] == l
                if mask.any():
                    self.rec[l, j] = t_train[mask, j].mean()

    def _bucketize(self, T):
        d = T.shape[1]
        return torch.stack([torch.bucketize(T[:, j].contiguous(), self.edges[:, j])
                            for j in range(d)], dim=1)            # [B, d] in [0, q-1]

    @torch.no_grad()
    def __call__(self, T):
        levels = self._bucketize(T)
        d = T.shape[1]
        return torch.stack([self.rec[levels[:, j], j] for j in range(d)], dim=1)


# ---------------------------------------------------------------------------
# P1: tightness of the capacity upper bound log K
# ---------------------------------------------------------------------------

def verify_p1(device="cpu", d=32, epochs=3, quick=False, fast=False):
    """With fixed width d, sweep the codebook size K and test whether the codebook entropy H(Z) grows
    logarithmically with log K with a slope approaching 1.

    Criterion: final-segment slope >= 0.7 passes; < 0.5 with low utilization -> collapse.
    """
    k_list = (4, 16, 64, 256) if (quick or fast) else (4, 8, 16, 32, 64, 128, 256)
    subset = 0.2 if (quick or fast) else None
    print("\n" + "=" * 60)
    print("[P1] Tightness of the capacity upper bound log K: H(Z) grows with log K, slope approaching 1")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)
    results = []

    for K in k_list:
        model = Net(d=d, n_codes=K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, device=device)
        acc, idx_lists = evaluate(model, test_loader, device=device)
        h_z = entropy_of(idx_lists[0], K)
        util = model.vq.codebook_utilization(idx_lists[0])
        n_restart = int(model.vq.restart_events.item())
        log_k = math.log(K)
        results.append((K, log_k, h_z, acc, util))
        print(f"  K={K:4d}  log K={log_k:5.2f}  H(Z)={h_z:5.2f}  "
              f"H(Z)/logK={h_z/log_k:5.2f}  util={util:4.0%}  restart={n_restart:6d}  acc={acc:.3f}")

    if len(results) >= 2 and not quick:
        (_, l0, h0, _, u0), (_, l1, h1, _, u1) = results[-2], results[-1]
        slope = (h1 - h0) / (l1 - l0)
        print(f"  [P1 criterion] final-segment slope = {slope:.2f} (tight if approaching 1)")
        if slope >= 0.7:
            verdict = "PASS: upper bound tight, tool usable"
        elif slope < 0.5 and (u0 < 0.5 or u1 < 0.5):
            verdict = "FAIL: codebook collapse, upper bound loose (needs codebook reset trick)"
        else:
            verdict = "PARTIAL: upper bound loose but no collapse"
        print(f"  [P1 verdict] {verdict}")
    return results


# ---------------------------------------------------------------------------
# P2: the min form for serial bottlenecks (wide vs deep)
# ---------------------------------------------------------------------------

def verify_p2(device="cpu", d=32, k_wide=256, k_each=6, epochs=3, quick=False, fast=False):
    """At the same parameter scale, 1 wide bottleneck (K=256) vs 3 narrow bottlenecks in series (each K=6, 6^3≈256).

    Criterion:
    - Success: acc_wide - acc_deep >= 0.10 (the task information is locked by the narrowest segment) and H_wide > log K_each;
    - Failure: gap < 0.05 (the series is not constrained by the narrowest segment -> the min form has no predictive power for the task).
    Honest note: H(Z3) <= log K_each is mathematically always true (Z3 has only K_each values); the non-trivial content
    is the locking effect on task accuracy and the fact that joint entropy cannot represent end-to-end capacity.
    """
    if quick:
        epochs = 1
    elif fast:
        epochs = 2
    print("\n" + "=" * 60)
    print(f"[P2] Serial min form: wide bottleneck (K={k_wide}) vs deep series (3xK={k_each})")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=0.3 if (quick or fast) else None)

    # (a) single wide bottleneck
    model_a = Net(d=d, n_codes=k_wide).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_one_epoch(model_a, train_loader, opt_a, device=device)
    acc_a, idx_a = evaluate(model_a, test_loader, device=device)
    h_a = entropy_of(idx_a[0], k_wide)
    print(f"  [wide] params {count_params(model_a):6d}  acc={acc_a:.3f}  H(Z)={h_a:.2f}  (log K={math.log(k_wide):.2f})")

    # (b) three narrow bottlenecks in series
    model_b = DeepVQNet(d=d, k_each=k_each).to(device)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_one_epoch(model_b, train_loader, opt_b, device=device)
    acc_b, idx_b = evaluate(model_b, test_loader, device=device)
    h_last = entropy_of(idx_b[2], k_each)
    h_joint = entropy_of(joint_code(idx_b, [k_each] * 3), k_each ** 3)
    util_last = model_b.vq3.codebook_utilization(idx_b[2])
    print(f"  [deep] params {count_params(model_b):6d}  acc={acc_b:.3f}  H(Z3)={h_last:.2f}  (<= log {k_each}={math.log(k_each):.2f}, always true)")
    print(f"       joint entropy H(Z1,Z2,Z3)={h_joint:.2f}  (<= log {k_each**3}={math.log(k_each**3):.2f})  last-segment utilization={util_last:.0%}")

    if not quick:
        gap = acc_a - acc_b
        print(f"  [P2 criterion] acc_wide - acc_deep = {gap:.3f}")
        if gap >= 0.10 and h_a > math.log(k_each):
            verdict = "PASS: task accuracy of the narrow series is locked by the narrowest segment (min form holds)"
        elif gap < 0.05:
            verdict = "FAIL: series accuracy not constrained by the narrowest segment (min form has no predictive power for the task; check whether VQ degenerated)"
        else:
            verdict = "PARTIAL: locking effect present but magnitude between 0.05-0.10"
        print(f"  [P2 verdict] {verdict}")
    return (acc_a, h_a), (acc_b, h_last, h_joint)


# ---------------------------------------------------------------------------
# P3: capacity matching accelerates convergence
# ---------------------------------------------------------------------------

def p3_verdict(results, max_epochs):
    """P3 verdict: gives a conclusion from [(d, median_epoch, mean_acc), ...] (does not depend on training, independently unit-testable).

    - Success: d* is an **interior point** (neither the lower nor the upper end of the sweep), and every d < d* is slower than d* by >= 1 epoch;
    - Partial: d* falls on the upper end of the sweep (monotone "larger is faster"; interior optimum not established);
    - Failure: d* is at the smallest end, or no width converges.

    Returns (verdict, detail_lines).
    """
    INF = max_epochs + 1
    epochs_map = [(d, e if e is not None else INF) for d, e, _ in results]
    acc_map = {d: a for d, _, a in results}
    d_min = min(d for d, _ in epochs_map)
    d_max = max(d for d, _ in epochs_map)
    best_d, best_e = min(epochs_map, key=lambda r: r[1])
    n_wall = sum(1 for _, e in epochs_map if e >= INF)
    detail = [
        f"  fastest convergence: d*={best_d} ({best_e} epoch); widths hitting the capacity wall (not converged) = {n_wall}"
        f"; sweep range [{d_min}, {d_max}]",
        "  same-direction reading (more robust): end-of-budget accuracy per width = "
        + ", ".join(f"d={d}:{acc_map[d]:.3f}" for d, _ in epochs_map),
    ]
    if best_e >= INF:
        return "FAIL: no width converged within budget (task beyond capability overall)", detail
    if best_d == d_max:
        return (f"PARTIAL: capacity wall exists (small d cannot converge), but interior optimum not established"
                f"-- d*={best_d} is the sweep upper end; sweep further up to confirm whether it falls back"), detail
    if best_d == d_min:
        return "FAIL: d* at the smallest end (no capacity wall, monotonically faster as d grows)", detail
    if all(e >= best_e + 1 for d, e in epochs_map if d < best_d):
        return "PASS: small d hit the capacity wall and slowed (or did not converge), middle d* fastest (capacity matching holds)", detail
    return "PARTIAL: d* is an interior point but small d not significantly slower", detail


def verify_p3(device="cpu", k_fixed=256, max_epochs=10, quick=False, fast=False, seeds=3):
    """With a fixed codebook size (large enough not to be a bottleneck), sweep the width d and measure
    the number of epochs to converge to 90% accuracy.

    Criterion:
    - Success: the fastest d* is an interior point (neither the lower nor the upper end of the sweep), and every configuration d < d* is slower than d* by >= 1 epoch;
    - Partial: d* falls on the upper end of the sweep (the monotone form "larger d is faster"; interior optimum not established);
    - Failure: d* is at the smallest end (no capacity wall).

    Multiple seeds (fixed 2026-09-19): the "convergence epoch" of a single run is extremely sensitive to
    random initialization——measured for the same d=32: converges in 4 epochs in full mode, >8 not converged
    in fast mode (same data, same seed, but a different d_list leads to a different model creation order).
    Hence each d runs `seeds` different initializations and takes the **median** (unconverged counted as +∞)
    as the representative value, and the per-seed results are printed so the dispersion can be reviewed.

    Honest note: the slowdown at large d may not appear in this setting (MLP+MNIST); it is only an observation,
    not a criterion; in addition, each configuration also reports the "end-of-budget accuracy", which is a
    much more robust same-direction reading than the convergence verdict.
    """
    if quick:
        d_list, max_epochs, subset, seeds = (2, 8, 32), 3, 0.3, 1
    elif fast:
        # Pitfalls of fast (measured 2026-09-19):
        # (1) The first version used "6 epochs × 30% subset": total steps 6×147=882 ≪ the 4×469=1876
        #     needed for d=32 to converge on the full set → no width converged, and the verdict output a
        #     false failure of "task beyond capability". So it was changed to full data + 8 epochs (3752 steps).
        # (2) The sweep must cover the large d that "may fall back", otherwise when d* lands on the upper end
        #     it is impossible to distinguish an interior optimum from the monotone form.
        # (3) A single seed is not reproducible → use multiple seeds and take the median.
        d_list, max_epochs, subset, seeds = (2, 8, 32, 64), 8, None, 3
    else:
        d_list, max_epochs, subset, seeds = (2, 4, 8, 16, 32, 64), 10, None, 3
    print("\n" + "=" * 60)
    print(f"[P3] Capacity matching: bottleneck width d should have an interior optimal convergence point ({seeds} seeds per width, median)")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)
    results = []

    for d in d_list:
        seed_epochs, seed_accs = [], []
        for s in range(seeds):
            torch.manual_seed(1000 * s + d)  # decorrelate across configurations and seeds
            model = Net(d=d, n_codes=k_fixed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            converge_epoch = None
            acc = 0.0
            for ep in range(1, max_epochs + 1):
                train_one_epoch(model, train_loader, optimizer, device=device)
                acc, _ = evaluate(model, test_loader, device=device)
                if acc >= 0.90 and converge_epoch is None:
                    converge_epoch = ep
            seed_epochs.append(converge_epoch)
            seed_accs.append(acc)
        # Representative value: median (unconverged participates in sorting as max_epochs+1, the lower bound of +∞)
        vals = sorted(e if e is not None else max_epochs + 1 for e in seed_epochs)
        med = vals[len(vals) // 2]
        median_epoch = None if med > max_epochs else med
        mean_acc = sum(seed_accs) / len(seed_accs)
        results.append((d, median_epoch, mean_acc))
        shown = [(">" if e is None else str(e)) for e in seed_epochs]
        status = f"{median_epoch} epoch" if median_epoch else f">{max_epochs} (not reached 90%)"
        print(f"  d={d:4d}  per-seed convergence epoch {shown} -> median {status}; end acc={mean_acc:.3f}")

    if not quick:
        verdict, detail = p3_verdict(results, max_epochs)
        for line in detail:
            print(line)
        print(f"  [P3 verdict] {verdict}")
    return results


# ---------------------------------------------------------------------------
# P4: residual/bypass topology invalidates the serial min bound (v2 first item)
# ---------------------------------------------------------------------------

class BranchVQNet(nn.Module):
    """Three narrow bottlenecks in series + one **parallel branch** (the minimal model of a residual/bypass topology).

        logits = fc4(z3) + head_branch(b)

    branch="vq"       : b = VQBottleneck(d, k_wide)(h0)  —— discrete wide-bottleneck branch
    branch="identity" : b = h0                            —— continuous bypass (the simplest form of a residual network)

    Compared with DeepVQNet, **the bottleneck chain is identical** (three segments each K=k_each, serial min = log k_each);
    the only difference is this parallel branch. The two topology rules in capacity.py give opposite predictions:
        min_capacity_serial([C_chain, C_branch])  -> still log k_each (predicts unchanged accuracy)
        sum_capacity_parallel([C_chain, C_branch]) -> C_chain + C_branch (predicts a large increase)
    This experiment is exactly to decide which of these two rules is right.
    """

    def __init__(self, d, k_each, branch, k_wide=256):
        super().__init__()
        if branch not in ("vq", "identity"):
            raise ValueError(f"branch must be 'vq' or 'identity', got {branch!r}")
        self.branch = branch
        self.fc1 = nn.Linear(784, d)
        self.vq1 = VQBottleneck(d, k_each)
        self.fc2 = nn.Linear(d, d)
        self.vq2 = VQBottleneck(d, k_each)
        self.fc3 = nn.Linear(d, d)
        self.vq3 = VQBottleneck(d, k_each)
        self.fc4 = nn.Linear(d, 10)
        if branch == "vq":
            self.vq_b = VQBottleneck(d, k_wide)
            self.head_b = nn.Linear(d, 10)
        else:
            self.head_b = nn.Linear(d, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h0 = F.relu(self.fc1(x))
        z1, i1 = self.vq1(h0)
        h2 = F.relu(self.fc2(z1)); z2, i2 = self.vq2(h2)
        h3 = F.relu(self.fc3(z2)); z3, i3 = self.vq3(h3)
        if self.branch == "vq":
            zb, ib = self.vq_b(h0)
            return self.fc4(z3) + self.head_b(zb), [i1, i2, i3, ib]
        return self.fc4(z3) + self.head_b(h0), [i1, i2, i3]


def p4_verdict(acc_serial, acc_branch, acc_wide, min_gain=0.30, tight=0.10):
    """P4 verdict: gives a conclusion from the measured accuracy of each topology (does not depend on training, independently unit-testable).

    acc_serial: accuracy of the pure series 3×K=k_each (i.e. the "locked" level predicted by the min rule)
    acc_branch: accuracy after adding one parallel branch (the sum rule predicts it should rise substantially)
    acc_wide  : accuracy of a single wide bottleneck K=k_wide (the level the wide branch alone can reach in the parallel version)

    - Success: gain >= min_gain and the parallel version is close to the single wide bottleneck (>= acc_wide - tight)
            -> the serial min fails for the parallel topology, and the direction given by additivity is correct;
    - Failure: gain < 0.10 (the parallel branch barely helped -> additivity does not hold);
    - Partial: gain is significant but does not reach the single wide bottleneck's level (additivity is overestimated).

    Returns (verdict, detail_lines).
    """
    gain = acc_branch - acc_serial
    deficit = acc_wide - acc_branch
    detail = [
        f"  pure series acc={acc_serial:.3f} | plus parallel branch acc={acc_branch:.3f} "
        f"| single wide bottleneck acc={acc_wide:.3f}",
        f"  gain = {gain:+.3f}; deficit of parallel version vs single wide bottleneck = {-deficit:+.3f}"
        f" (approaching 0 means end-to-end capacity is contributed by the parallel branch; serial min is no longer the constraint)",
    ]
    if gain >= min_gain and deficit <= tight:
        return ("PASS: the same serial bottleneck capacity (min = log k_each) no longer locks accuracy once a parallel branch is added"
                "-- the topology must be judged before reporting a min upper bound"), detail
    if gain < 0.10:
        return "FAIL: parallel branch did not lift accuracy (serial min still holds under parallel topology; additivity does not hold)", detail
    return (f"PARTIAL: gain {gain:.3f} significant, but {deficit:.3f} below the single wide bottleneck (> tolerance {tight})"
            f"; the additivity upper bound is too optimistic"), detail


def verify_p4(device="cpu", d=32, k_each=6, k_wide=256, epochs=3, quick=False, fast=False):
    """P4: does a residual/bypass topology invalidate the serial min bound (v2 first item).

    Protocol: same task (MNIST), same budget, train four topologies,
    with **the bottleneck chain part completely identical** (three segments each K=k_each → serial min = log k_each):
      (a) pure series 3×K=k_each                —— no bypass, min rule applies;
      (b) (a) + parallel discrete wide bottleneck K=k_wide     —— has bypass, min/sum rules give opposite predictions;
      (c) (a) + parallel continuous bypass (residual identity)—— the simplest form of a residual network;
      (d) single wide bottleneck K=k_wide                —— reference: what level the wide branch can reach alone.

    Criterion: gain = acc(b) - acc(a) >= 0.30 passes; < 0.10 fails (detailed criterion in p4_verdict).

    Honest notes:
    - Both branches of (b) are **discrete bottlenecks**, so the objection that "the parallel version only won
      because it added a continuous path" does not apply; (c) is the continuous limit, whose bypass capacity
      is not given by log K.
    - Therefore this experiment proves "**the min bound fails when there is a bypass, and the direction should
      be given by additivity**", not "the end-to-end capacity is exactly equal to the sum of the branches'
      log K"——the latter's rigorous quantification requires an independent estimate of branch capacities
      (e.g. using the Fisher effective dimension to characterize the effective dimensionality of the continuous
      bypass, a v2 follow-up).
    """
    if quick:
        epochs = 1
    elif fast:
        epochs = 2
    subset = 0.3 if (quick or fast) else None
    print("\n" + "=" * 60)
    print(f"[P4] Residual/bypass topology: with the same serial bottleneck chain (3xK={k_each}), does the min bound still constrain accuracy when a bypass exists")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)

    c_chain = min_capacity_serial([discrete_capacity_logK(k_each)] * 3)
    c_branch = discrete_capacity_logK(k_wide)
    print(f"  [capacity] serial chain C_chain = min_capacity_serial([log {k_each}]x3) = {c_chain:.2f} nats")
    print(f"  [topology] serial rule misapplied: min_capacity_serial([C_chain, log {k_wide}])"
          f" = {min_capacity_serial([c_chain, c_branch]):.2f} nats  -> predicts accuracy unchanged (wrong)")
    print(f"  [topology] parallel rule:  sum_capacity_parallel([C_chain, log {k_wide}])"
          f" = {sum_capacity_parallel([c_chain, c_branch]):.2f} nats  -> predicts accuracy lift")

    def run(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_one_epoch(model, train_loader, opt, device=device)
        return evaluate(model, test_loader, device=device)

    acc_a, idx_a = run(DeepVQNet(d=d, k_each=k_each).to(device))
    print(f"  [a pure series ] acc={acc_a:.3f}   H(Z3)={entropy_of(idx_a[2], k_each):.2f}"
          f" (<= log {k_each}={math.log(k_each):.2f})")

    model_b = BranchVQNet(d=d, k_each=k_each, branch="vq", k_wide=k_wide).to(device)
    acc_b, idx_b = run(model_b)
    print(f"  [b parallel wide bottleneck] acc={acc_b:.3f}   H(Z3)={entropy_of(idx_b[2], k_each):.2f} (bottleneck chain side unchanged)")

    acc_c, idx_c = run(BranchVQNet(d=d, k_each=k_each, branch="identity").to(device))
    print(f"  [c residual identity] acc={acc_c:.3f}   H(Z3)={entropy_of(idx_c[2], k_each):.2f}")

    acc_d, idx_d = run(Net(d=d, n_codes=k_wide).to(device))
    print(f"  [d single wide bottleneck] acc={acc_d:.3f}   H(Z)={entropy_of(idx_d[0], k_wide):.2f}"
          f" (<= log {k_wide}={math.log(k_wide):.2f})")

    if not quick:
        verdict, detail = p4_verdict(acc_a, acc_b, acc_d)
        for line in detail:
            print(line)
        print(f"  [P4 verdict] {verdict}")
        print(f"  [P4 same-direction observation] continuous bypass acc={acc_c:.3f} (continuous limit of a residual network,"
              f" lift {acc_c - acc_a:+.3f})")
    return (acc_a, idx_a), (acc_b, idx_b), acc_c, (acc_d, idx_d)


# ---------------------------------------------------------------------------
# Feature A: quantization lower bound (synthetic task, small dimension, large information)
# ---------------------------------------------------------------------------

def quant_cliff(acc_by_q, acc_cont, tol=0.02):
    """Measured cliff = the smallest quantization level that preserves the baseline (acc_cont - tol); returns None if no level preserves it."""
    ok = [q for q in sorted(acc_by_q) if acc_by_q[q] >= acc_cont - tol]
    return min(ok) if ok else None


def quant_verdict(q_min, q_cliff, acc_cont, acc_half, octave=2.0):
    """Verdict for a single run of feature A (pure function, independently unit-testable).

    - Pass: the measured cliff falls within one octave of q_min, and half the cliff level really drops >= 5%;
    - Fail: the cliff is outside the octave (>2× means the lower bound is too optimistic/too tight, underestimating demand;
            <0.5× means too pessimistic/too loose), or half the cliff level still preserves the baseline
            (the cliff is not pinched, so there is no real capacity wall).

    Returns (verdict, detail_lines).
    """
    lo, hi = q_min / octave, q_min * octave
    drop = acc_cont - acc_half
    detail = [f"  measured cliff={q_cliff}, predicted q_min={q_min:.2f}, octave window [{lo:.1f}, {hi:.1f}];"
              f" drop at half the cliff level = {drop:.3f} (>=0.05 counts as a drop)"]
    if drop < 0.05:
        return "FAIL: half the cliff level still preserves the baseline (cliff not pinched; quantizer has no real capacity wall)", detail
    if lo <= q_cliff <= hi:
        return f"PASS: measured cliff {q_cliff} falls within one octave of q_min", detail
    if q_cliff > hi:
        return f"FAIL: measured cliff {q_cliff} far above q_min (lower bound too optimistic/tight; actual demand underestimated)", detail
    return f"FAIL: measured cliff {q_cliff} far below q_min (lower bound too pessimistic/loose)", detail


def verify_quant(quick=False, fast=False, d=3, seeds=3):
    """Cliff prediction for q_min = 2^(R_IB/d).

    Protocol:
    1. Train a VQ probe -> R_IB = I(Z;Y) (task-relevant information; **do not use H(Z)**, see below);
    2. Train a continuous classifier -> apply post-training percentile quantization to the input, sweep q and measure accuracy;
    3. Criterion: the measured cliff (the smallest level that preserves the baseline) falls within one octave of q_min,
       and half the cliff level really drops >= 5%.

    Why R_IB uses I(Z;Y) rather than H(Z):
    - What quantization must preserve is "the information in the input that relates to the label", I(X;Y) ≈ H(Y)
      (log 64 = 4.16 nats for this task);
    - H(Z) is the representation entropy (free to grow up to log K = 5.55 nats), which overestimates systematically,
      pushing q_min up to 6.05 while the true cliff is at q=4 (does not pass);
    - I(Z;Y) uses the Fano lower bound (given by the probe error rate, no binning bias), giving q_min ≈ 4,
      consistent with the cliff. Using plug-in mutual information directly does not work: N=1920 ≪ K·C=16384,
      the upward bias is severe, and the Miller-Madow correction pushes it down to 0.

    Training configuration note: softmax CE converges slowly on unnormalized inputs with large coordinates
    (the logits need to reach ~10² magnitude; lr=1e-3 × 20ep measured only reaches acc≈0.33), so lr=1e-2,
    60 epochs is used (measured acc≈0.97).
    """
    epochs = 25 if (quick or fast) else 60
    n_per_class = 50 if (quick or fast) else 150
    lr = 1e-2
    seeds = 1 if quick else max(1, seeds)
    print("\n" + "=" * 60)
    print(f"[Feature A] Quantization lower bound q_min = 2^(R_IB/d) (synthetic 64-class task, d=3; {seeds} seeds)")
    print("=" * 60)

    records, per_seed = [], []
    for s in range(seeds):
        # Both the data realization and the initialization vary with the seed (the seed only changes randomness, not the task's nominal information log 64)
        torch.manual_seed(1000 + s)
        (x_train, y_train), (x_test, y_test) = make_grid64(n_per_class=n_per_class, seed=s)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=256)

        # 1. VQ probe measures R_IB = I(Z;Y) (task-relevant information)
        #    Note: H(Z) (codebook entropy) cannot be used as a proxy——H(Z) is the representation entropy,
        #    which systematically overestimates the task information
        #    (in the synthetic task H(Z) can reach 5.4 nats > the task's own log64 = 4.16 nats), pushing q_min too high.
        model_vq = Classifier3D(with_vq=True, n_codes=256)
        opt = torch.optim.Adam(model_vq.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(model_vq, train_loader, opt)
        acc_vq, idx_lists = evaluate(model_vq, test_loader)
        h_z = entropy_of(idx_lists[0], 256)
        # Use the Fano lower bound (given by the error rate) rather than plug-in mutual information:
        # N=1920 ≪ K·C=16384, plug-in is biased upward and the Miller-Madow correction is biased downward
        # (measured correction to 0), while Fano has no binning bias and is conservative in direction.
        r_ib_nats = task_info_fano(1.0 - acc_vq, 64, h_y=math.log(64))
        r_ib_bits = r_ib_nats / math.log(2)
        util_vq = model_vq.vq.codebook_utilization(idx_lists[0])
        q_min = 2 ** (r_ib_bits / d)

        # 2. Continuous classifier + post-training percentile quantization
        model_c = Classifier3D(with_vq=False)
        opt = torch.optim.Adam(model_c.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(model_c, train_loader, opt)
        acc_cont, _ = evaluate(model_c, test_loader)

        q_safe = max(2, math.ceil(q_min))
        q_unsafe = max(2, math.floor(q_min / 2))
        # The sweep grid must cover the levels used by the criterion; otherwise the criterion takes a default value and "necessarily passes" (a defect of the old version)
        q_scan = tuple(sorted({2, 3, 4, 5, 6, 8, 12, 16, q_safe, q_unsafe, q_safe + 1}))
        acc_by_q = {}
        for q in q_scan:
            if q < 2:
                continue
            quant = PercentileQuantizer(x_train, q)
            with torch.no_grad():
                out = model_c.fc(quant(x_test))
                acc_by_q[q] = (out.argmax(dim=1) == y_test).float().mean().item()

        q_cliff = quant_cliff(acc_by_q, acc_cont)
        half = max(2, (q_cliff or 2) // 2)
        acc_half = acc_by_q.get(half, acc_cont)
        if q_cliff is None:
            verdict_s, obs = "FAIL: no level preserves the baseline (quantizer or baseline problem)", None
        else:
            verdict_s, _ = quant_verdict(q_min, q_cliff, acc_cont, acc_half)
            # The cliff must be "pinched" (half the level really drops >= 5%) to count as a reliable observation; otherwise it does not enter the summary
            obs = q_cliff if (acc_cont - acc_half) >= 0.05 else None

        print(f"  seed {s}: probe acc={acc_vq:.3f}(util {util_vq:.0%}) -> R_IB >= {r_ib_nats:.2f} nats = "
              f"{r_ib_bits:.2f} bits; continuous acc={acc_cont:.3f}; q_min={q_min:.2f} -> measured cliff={q_cliff}")
        print("    " + "  ".join(f"q{q}:{acc_by_q[q]:.3f}" for q in sorted(acc_by_q)))
        if not quick:
            print(f"    [seed {s}] {verdict_s}"
                  + ("" if obs is not None else " (cliff unreliable, excluded from summary)"))
        records.append((s, q_min, obs))
        per_seed.append((s, r_ib_bits, q_min, q_cliff, acc_cont, acc_by_q))

    if not quick:
        if seeds > 1:
            print(f"  [Feature A multi-seed summary] {seeds} seeds (data realization and initialization both vary with the seed)")
        verdict, detail = ratio_verdict(records, "measured cliff / predicted q_min")
        for line in detail:
            print(line)
        print(f"  [Feature A verdict] {verdict}")
    return per_seed


# ---------------------------------------------------------------------------
# QAT: quantization-aware training (a v2 extension of feature A——removing PTQ calibration artifacts)
# ---------------------------------------------------------------------------

class QATNet(nn.Module):
    """QAT-uniform arm: per-dimension q-level quantization with a fixed percentile grid, training the classifier with quantization enabled.

    The only difference from PTQ (PercentileQuantizer): PTQ trains a continuous classifier first and applies
    quantization afterwards (the classifier has never seen the quantization grid); this arm puts **the same**
    percentile edges and empirical-mean reconstruction values into the training loop. The quantization parameters
    (buffers, non-learnable) are exactly the same as in PTQ——so the attribution of "artifact removal amount"
    falls cleanly on **classifier adaptation**; the contribution of learning the quantization parameters itself
    is measured separately by the QATVQNet arm (the difference between the two arms). There are no learnable
    parameters upstream, so STE is irrelevant here (gradients only need to flow to fc).
    """

    def __init__(self, t_train, q, n_classes=64):
        super().__init__()
        self.q = q
        d = t_train.shape[1]
        if q > 1:
            probs = torch.arange(1, q, dtype=t_train.dtype) / q
            edges = torch.quantile(t_train, probs, dim=0)      # [q-1, d]
        else:
            edges = torch.zeros(0, d)
        self.register_buffer("edges", edges)
        levels = torch.stack([torch.bucketize(t_train[:, j].contiguous(), edges[:, j])
                              for j in range(d)], dim=1)        # [N, d]
        rec = torch.zeros(q, d)
        for j in range(d):
            for l in range(q):
                mask = levels[:, j] == l
                if mask.any():
                    rec[l, j] = t_train[mask, j].mean()
        self.register_buffer("rec", rec)
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        levels = torch.stack([torch.bucketize(x[:, j].contiguous(), self.edges[:, j])
                              for j in range(x.shape[1])], dim=1)
        xq = torch.stack([self.rec[levels[:, j], j] for j in range(x.shape[1])], dim=1)
        return self.fc(xq), None


class QATVQNet(nn.Module):
    """QAT-vq arm: one VQ(1, q) per dimension (learnable codeword positions + commit loss + dead-code reset).

    The quantization parameters and the classifier are learned jointly; the codebook is initialized with the same
    quantiles as PTQ/QAT-uniform (same starting point; the only difference is "whether the codewords are
    subsequently moved by gradients"). The capacity is still q^d distinguishable states and the q_min formula is
    unchanged. The difference from QAT-uniform = the contribution of learning the quantization parameters.
    train_one_epoch automatically applies commit loss and dead-code reset to all VQBottlenecks within this class,
    with no extra wiring needed.
    """

    def __init__(self, t_train, q, n_classes=64):
        super().__init__()
        d = t_train.shape[1]
        self.vqs = nn.ModuleList()
        for j in range(d):
            vq = VQBottleneck(1, q, metric="euclid")
            with torch.no_grad():
                probs = (torch.arange(q, dtype=t_train.dtype) + 0.5) / q
                vq.codebook.copy_(torch.quantile(t_train[:, j:j + 1], probs, dim=0))
            self.vqs.append(vq)
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        cols = [self.vqs[j](x[:, j:j + 1])[0].squeeze(-1) for j in range(x.shape[1])]
        return self.fc(torch.stack(cols, dim=1)), None


def qat_verdict(q_min, acc_cont, acc_by_q, tol=0.02, wall_drop=0.05):
    """QAT verdict (pure function, independently unit-testable). acc_by_q: {q: acc} (the curve of one QAT arm).

    J1 artifact removal: all levels with q >= ceil(q_min) have acc >= acc_cont - tol
       (PTQ has anomalous drops at safe levels——the calibration artifact of percentile edges splitting clusters);
    J2 capacity wall preserved: the level q = ceil(q_min) - 1 is still below the baseline by >= wall_drop
       (d*log q < R_IB, I(Z;Y) <= H(Z) = d*log q < R_IB, Fano forces errors——
       this is a mathematical wall at the DPI level; QAT only changes "whether the classifier adapts to the grid",
       not the information-theoretic lower bound, and must not rescue it).

    Returns (verdict, j1, j2, detail_lines). j2=None means the grid did not cover the wall level.
    """
    q_safe = max(2, math.ceil(q_min))
    q_wall = q_safe - 1
    detail = []
    artifact = [q for q in sorted(acc_by_q) if q >= q_safe and acc_by_q[q] < acc_cont - tol]
    j1 = not artifact
    detail.append(f"  J1 artifact removal: {'PASS' if j1 else 'FAIL'}"
                  + (f" (safe levels q{artifact} still below baseline-{tol})" if artifact
                     else f" (all levels q>={q_safe} preserve baseline-{tol})"))
    if q_wall in acc_by_q:
        drop = acc_cont - acc_by_q[q_wall]
        j2 = drop >= wall_drop
        detail.append(f"  J2 capacity wall preserved: {'PASS' if j2 else 'FAIL'} (level q={q_wall} drop {drop:.3f},"
                      f" threshold {wall_drop} -- mathematical wall of DPI+Fano; QAT must not rescue it)")
    else:
        j2 = None
        detail.append(f"  J2 capacity wall preserved: UNDECIDABLE (grid does not cover wall level q={q_wall})")
    if j1 and j2:
        v = "PASS: safe-level artifact removed, and capacity wall not rescued"
    elif j1 and j2 is None:
        v = "PARTIAL: J1 passes, but wall level not in grid (protocol gap)"
    else:
        failed = [name for name, bad in (("J1", not j1), ("J2", j2 is False)) if bad]
        v = f"FAIL: {' and '.join(failed)} did not pass"
    return v, j1, j2, detail


def verify_qat(quick=False, fast=False, seeds=3, d=3):
    """QAT vs PTQ three-arm comparison (a v2 extension of feature A, removing PTQ calibration artifacts).

    Background: PTQ percentile calibration has a systematic artifact at q=5/6
    (reproduced across three seeds: acc 0.466-0.564, below the 0.96+ of q=4)——each dimension has 4 equal-mass
    modes; when q is a multiple of 4 the percentile edges land exactly in the mode valleys, while at q=5/6
    the edges split into the interior of the clusters.
    This experiment tests two things:
    1. Whether QAT (training with quantization enabled) can remove this artifact (J1);
    2. Whether the capacity wall (the DPI+Fano hard wall of d*log q < R_IB when q < q_min) is preserved by
       QAT (J2)——QAT cannot change the information-theoretic lower bound, only "whether the classifier adapts to the grid".

    Three arms (same data realization, same seed convention):
    - PTQ: continuous classifier + post-hoc percentile quantization (the feature A protocol as-is, control baseline);
    - QAT-uniform: the same percentile grid put into the training loop, quantization parameters fixed (buffers),
      only the classifier adapts——the artifact removal amount is attributed to "training adaptation";
    - QAT-vq: one VQ(1,q) per dimension, learnable codebook——the difference from QAT-uniform is attributed to
      "quantization parameter learning".

    Criterion (pre-registered): pass = J1 and J2 both hold; the multi-seed summary requires >= 2/3 seeds to pass.
    """
    epochs = 25 if quick else (40 if fast else 60)
    n_per_class = 50 if quick else (100 if fast else 150)
    lr = 1e-2
    seeds = 1 if quick else max(1, seeds)
    print("\n" + "=" * 60)
    print(f"[QAT] Quantization-aware training vs post-training quantization: artifact removal and capacity wall preservation (synthetic 64-class task; {seeds} seeds)")
    print("=" * 60)
    print("  [background] PTQ percentile calibration has a systematic artifact at q=5/6 (edges split clusters; reproduced across three seeds).")
    print("  [criterion] J1 artifact removal: all QAT levels with q >= ceil(q_min) preserve baseline-0.02;")
    print("         J2 capacity wall preserved: level q=ceil(q_min)-1 still >=0.05 below baseline (DPI+Fano")
    print("         hard wall -- QAT only changes classifier adaptation, not the information-theoretic lower bound).")
    print("         PASS = J1 and J2; multi-seed summary requires >= 2/3 seeds to pass.")

    per_seed, arm_pass = [], []
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        (x_train, y_train), (x_test, y_test) = make_grid64(n_per_class=n_per_class, seed=s)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=256)

        # 1. Probe (same as feature A: the Fano lower bound of R_IB -> q_min)
        model_vq = Classifier3D(with_vq=True, n_codes=256)
        opt = torch.optim.Adam(model_vq.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(model_vq, train_loader, opt)
        acc_vq, _ = evaluate(model_vq, test_loader)
        r_ib_nats = task_info_fano(1.0 - acc_vq, 64, h_y=math.log(64))
        q_min = 2 ** (r_ib_nats / math.log(2) / d)

        # 2. PTQ arm: continuous classifier + post-hoc percentile quantization (control; acc_cont is also the baseline for all three arms)
        model_c = Classifier3D(with_vq=False)
        opt = torch.optim.Adam(model_c.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(model_c, train_loader, opt)
        acc_cont, _ = evaluate(model_c, test_loader)

        q_safe = max(2, math.ceil(q_min))
        q_scan = tuple(sorted({2, 3, 4, 5, 6, 8, 12, 16, q_safe, q_safe + 1}))
        acc_ptq = {}
        for q in q_scan:
            quant = PercentileQuantizer(x_train, q)
            with torch.no_grad():
                out = model_c.fc(quant(x_test))
                acc_ptq[q] = (out.argmax(dim=1) == y_test).float().mean().item()

        # 3. QAT two arms: train each q from scratch (same percentile starting point as PTQ/QAT-uniform)
        acc_qat_u, acc_qat_v = {}, {}
        for q in q_scan:
            torch.manual_seed(7000 + 100 * s + q)
            m_u = QATNet(x_train, q)
            opt = torch.optim.Adam(m_u.parameters(), lr=lr)
            for _ in range(epochs):
                train_one_epoch(m_u, train_loader, opt)
            acc_qat_u[q], _ = evaluate(m_u, test_loader)

            torch.manual_seed(8000 + 100 * s + q)
            m_v = QATVQNet(x_train, q)
            opt = torch.optim.Adam(m_v.parameters(), lr=lr)
            for _ in range(epochs):
                train_one_epoch(m_v, train_loader, opt)
            acc_qat_v[q], _ = evaluate(m_v, test_loader)

        # 4. Three-arm comparison table + verdict
        print(f"  seed {s}: probe acc={acc_vq:.3f} -> R_IB >= {r_ib_nats:.2f} nats,"
              f" q_min={q_min:.2f} (q_safe={q_safe}); continuous baseline acc={acc_cont:.3f}")
        print(f"    {'q':>4} {'PTQ':>7} {'QAT-u':>7} {'QAT-vq':>7}")
        for q in q_scan:
            mark = " <- artifact level" if (q >= q_safe and acc_ptq[q] < acc_cont - 0.05) else ""
            print(f"    {q:>4} {acc_ptq[q]:7.3f} {acc_qat_u[q]:7.3f} {acc_qat_v[q]:7.3f}{mark}")
        if not quick:
            v_u, j1_u, j2_u, det_u = qat_verdict(q_min, acc_cont, acc_qat_u)
            v_v, j1_v, j2_v, det_v = qat_verdict(q_min, acc_cont, acc_qat_v)
            print(f"    [seed {s}] QAT-uniform: {v_u}")
            for line in det_u:
                print(line)
            print(f"    [seed {s}] QAT-vq: {v_v}")
            for line in det_v:
                print(line)
            arm_pass.append((s, bool(j1_u and j2_u), bool(j1_v and j2_v)))
        per_seed.append((s, q_min, acc_cont, acc_ptq, acc_qat_u, acc_qat_v))

    if not quick and seeds > 1:
        print(f"\n  [QAT multi-seed summary] {seeds} seeds")
        for i, arm in ((1, "QAT-uniform"), (2, "QAT-vq")):
            n_pass = sum(1 for r in arm_pass if r[i])
            n_need = math.ceil(2 / 3 * seeds)
            concl = "PASS" if n_pass >= n_need else "FAIL"
            print(f"    {arm}: {n_pass}/{seeds} seeds (J1 and J2 both hold)"
                  f" -> {concl} (threshold {n_need}/{seeds})")
        for i, arm in ((4, "QAT-uniform"), (5, "QAT-vq")):
            recovers = []
            for rec in per_seed:
                _, q_min_, acc_c_, ptq_, qat_u_, qat_v_ = rec
                qs_ = max(2, math.ceil(q_min_))
                art = [q for q in ptq_ if q >= qs_ and acc_c_ - ptq_[q] >= 0.05]
                if art:
                    accs_ = qat_u_ if i == 4 else qat_v_
                    recovers.append(sum(accs_[q] - ptq_[q] for q in art) / len(art))
            if recovers:
                print(f"    {arm} artifact removal amount (mean recovery on safe levels where PTQ drops >=0.05):"
                      f" mean {sum(recovers) / len(recovers):.3f},"
                      f" range [{min(recovers):.3f}, {max(recovers):.3f}]")
    return per_seed


# ---------------------------------------------------------------------------
# Feature B: distillation capacity lower bound (MNIST)
# ---------------------------------------------------------------------------

def verify_distill(device="cpu", d=16, k_teacher=256, quick=False, fast=False, seeds=3):
    """The **gap-curve** criterion for K_min = ceil(e^R_IB) (v2 reworked version).

    Protocol:
    1. Train teacher -> R_IB = H(Z_teacher) -> K_min;
    2. Distill students of different K (CE + KL(T=3)), measure acc_s(K), compute the gap g(K) = acc_t - acc_s(K);
    3. Criterion (**relative** rather than absolute): K*(gap) (= the smallest K where the gap reaches the reference
       plateau g_ref = g(K_max) + tol) satisfies K*/2 <= K_min <= 2K* (within one octave) -> pass.
       Precondition: the plateau must already be established (see distill_plateau_ok); otherwise it is recorded as
       "unlocated" and does not enter the pass.

    R_IB uses H(Z_teacher) (the teacher's representation rate) rather than I(Z;Y): distillation transfers the
    teacher's soft labels / representation structure, whose information can exceed the class-label information
    H(Y) (only 2.3 nats for 10 classes), and the derivation "capacity must accommodate the rate being transferred"
    corresponds to the representation rate. (An early single run saw H(Z)=5.26 nats -> K_min=192 matching a turning
    point at 194, but that match failed the multi-seed check——see below.)
    (The quantization setting of feature A is different: quantization must preserve the information in the input
    that relates to the label, so I(Z;Y) is used there.)

    Budget alignment (fixed 2026-09-19): the student's number of epochs must be >= the teacher's, otherwise the
    student systematically underfits, K* is pushed up and the criterion necessarily fails——the earlier
    ep_student=2 < ep_teacher=3 was exactly this defect.

    **Criterion rework (2026-09-19, v2)**: the old criterion used the absolute bar "acc_s >= teacher - 0.03";
    measured teacher rising from 0.864 to 0.884 (only 0.020 stronger) flipped K* from 132 into "no one in the whole
    grid meets the bar"——the threshold is of the same order as the effect under test, so whether the criterion holds
    is decided by the teacher's incidental strength. Reworked as the **gap curve**: using the gap of the largest-K
    student in the same run as the reference (it is the least capacity-limited), the criterion depends only on
    g(K) - g_ref, and the teacher's strength is canceled out. The script still prints the K* of the **old absolute
    criterion** as a comparison column, so the difference before and after the rework is directly visible.
    """
    ep_teacher = 1 if quick else 3
    ep_student = ep_teacher  # budget alignment: the student must not have fewer epochs than the teacher
    seeds = 1 if quick else max(1, seeds)
    subset = 0.3 if (quick or fast or seeds > 1) else None
    print("\n" + "=" * 60)
    print(f"[Feature B] Distillation capacity lower bound K_min = ceil(e^R_IB) (MNIST; {seeds} seeds)")
    print("=" * 60)
    if seeds > 1 and not (quick or fast):
        print("  [budget] Multi-seed mode automatically uses the 30% subset: 7 models x multiple seeds is not feasible under the full budget.")
        print("         The purpose of this rerun is to check **stability** (whether the K* location is stable, whether the plateau is stable),")
        print("         not to re-calibrate the exact K_min value -- for the latter use a single-seed full run.")
    print("  [criterion] Gap curve: g(K)=acc_teacher - acc_student(K); reference plateau g_ref=g(K_max);")
    print("         K*(gap) = smallest K where the gap first falls within g_ref+tol; if the plateau is not established, refuse to report K*.")

    train_loader, test_loader = get_mnist(subset=subset)

    records, curves, k_list = [], [], None
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        teacher = Net(d=d, n_codes=k_teacher).to(device)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for _ in range(ep_teacher):
            train_one_epoch(teacher, train_loader, opt, device=device)
        acc_t, idx_t = evaluate(teacher, test_loader, device=device)
        r_ib = entropy_of(idx_t[0], k_teacher)
        k_min = max(2, math.ceil(math.exp(r_ib)))

        if k_list is None:
            # The grid is anchored on the first seed's K_min and shared by all seeds——only then can
            # "the gap of the same K under different seeds" be compared side by side.
            # The two top anchors (192/256) are necessary for the **plateau reference**: without a K
            # "above the plateau", it is impossible to judge whether g_ref is already the irreducible gap
            # (plateau detection: see distill_plateau_ok).
            if quick:
                k_list = sorted({2, 8, k_min, min(k_min * 2, k_teacher), k_teacher})
            else:
                k_list = sorted({32, max(2, k_min - 2), k_min,
                                 min(k_min + 2, k_teacher),
                                 min(192, k_teacher), k_teacher})
            print(f"  [K grid] anchored on seed 0's K_min={k_min}: {k_list}")

        accs = {}
        for K in k_list:
            torch.manual_seed(2000 + 100 * s + K)
            student = Net(d=d, n_codes=K).to(device)
            opt = torch.optim.Adam(student.parameters(), lr=1e-3)
            for _ in range(ep_student):
                train_one_epoch(student, train_loader, opt, device=device, distill_from=teacher)
            acc_s, _ = evaluate(student, test_loader, device=device)
            accs[K] = acc_s

        plateau = distill_plateau_ok(k_list, accs)
        k_star = distill_gap_k_star(acc_t, k_list, accs) if plateau else None
        # K* of the old absolute criterion, only as a comparison column (the difference before and after the rework is obvious at a glance)
        k_star_abs = min([K for K in k_list if accs[K] >= acc_t - 0.03], default=None)
        records.append((s, k_min, k_star))
        curves.append((s, acc_t, k_min, accs))

        gaps = {K: acc_t - accs[K] for K in k_list}
        g_ref = gaps[k_list[-1]]
        print(f"  seed {s}: teacher acc={acc_t:.3f}  R_IB={r_ib:.2f} nats -> K_min={k_min}"
              f"; g_ref=g({k_list[-1]})={g_ref:.3f}")
        print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}" for K in k_list))
        print(f"    plateau established={'yes' if plateau else 'no'}; **K*(gap)={k_star}**"
              f"; [reference] old absolute criterion K*(teacher-0.03)={k_star_abs} (pass line {acc_t - 0.03:.3f})")

    # The per-K × per-seed **gap** matrix: directly exposes "whether the gap curve is stable"
    heads = " ".join(f"{'seed' + str(s):>9}" for s, _, _, _ in curves)
    print(f"\n  gap matrix g(K)=teacher-acc  {'K':>4} {heads} {'mean':>9}")
    for K in k_list:
        row = " ".join(f"{acc_t - accs[K]:9.3f}" for _, acc_t, _, accs in curves)
        mean = sum(acc_t - accs[K] for _, acc_t, _, accs in curves) / len(curves)
        print(f"  {'':>22} {K:>4} {row} {mean:9.3f}")
    print("  teacher acc per seed: "
          + ", ".join(f"seed{s}={acc_t:.3f}" for s, acc_t, _, _ in curves))
    print("  plateau reference g_ref per seed: "
          + ", ".join(f"seed{s}={acc_t - accs[k_list[-1]]:.3f}"
                      for s, acc_t, _, accs in curves))

    if not quick:
        if seeds > 1:
            print(f"\n  [Feature B multi-seed summary] {seeds} seeds (criterion: gap curve)")
        verdict, detail = ratio_verdict(records, "measured K*(gap) / predicted K_min")
        for line in detail:
            print(line)
        print(f"  [Feature B verdict] {verdict}")
    return records, curves


# ---------------------------------------------------------------------------
# Feature B v3: distillation capacity lower bound (grid64 hard-wall task)
# ---------------------------------------------------------------------------

def verify_distill_grid(quick=False, fast=False, seeds=3, k_teacher=256):
    """The gap-curve criterion for K_min = ceil(e^R_IB) (v3: reworked version on the grid64 hard-wall task).

    Why rework (the root cause of v2/MNIST being judged "operationalization failure"): the 10-class MNIST task
    only needs log 10 = 2.3 nats, so a codebook with K >= 32 is already informationally more than enough, and the
    accuracy-capacity curve is smooth with no knee——all the wiggles on the curve are optimization/generalization
    noise, with no locatable capacity wall.
    v3 switches to a task with a **hard information wall**: a 4x4x4 grid, 64 classes, H(Y) = log 64 = 4.16 nats.
    When the student codebook K < 64, log K < H(Y); by I(Z;Y) <= H(Z) <= log K and the Fano inequality, the student
    **mathematically cannot** preserve the teacher's accuracy; when K >= 64 the information obstacle disappears.
    The sharpness of the cliff is guaranteed by the task structure, not decided by training noise.

    R_IB definition change (the second change in v3; v2 used H(Z_teacher)):
    - KD objective = CE + KL(teacher logits). The KL term requires the student to reproduce the teacher's output
      distribution, in which the rate about the task is I(Z_t;Y), **not** the representation entropy H(Z_t)——
      the teacher codebook (K=256)'s H(Z_t) can approach log 256 = 5.55 nats, and the part beyond log 64 is
      irrelevant to the task; the student does not need to pay for it.
    - In v2 with H(Z_t), K_min=134 on MNIST happened to be close to K*=132; multiple seeds have shown that was a
      coincidence on a smooth curve. v3 switches to the Fano lower bound of I(Z_t;Y) (given by the teacher error
      rate, no binning bias, conservative in direction), unified with feature A: **both quantization and
      distillation must preserve task-relevant information**.
    - Honest note: Fano gives the necessary capacity lower bound for "preserving the teacher's accuracy", not the
      sufficient capacity for "fully reproducing the teacher's soft labels (including dark knowledge)"; this
      experiment claims the former.

    Budget: on grid64, 7680 training samples x a small model, 60 epochs completes in a few minutes, and multi-seed
    full needs no 30% subset compromise like the MNIST version. The student's epochs/lr are strictly aligned with
    the teacher (the budget alignment principle, a v2 lesson).

    If the curve is still smooth on this task (gaps of K=32 and K=64 are close, cliff not locatable), then feature B
    is removed entirely from the claimable list——that too is a clean theoretical conclusion.
    """
    epochs = 25 if quick else (40 if fast else 60)
    n_per_class = 50 if quick else (100 if fast else 150)
    lr = 1e-2  # needed for convergence on large-coordinate inputs (consistent with feature A)
    seeds = 1 if quick else max(1, seeds)
    wall = 64  # hard-wall location = number of classes: log 64 = H(Y)
    print("\n" + "=" * 60)
    print(f"[Feature B] Distillation capacity lower bound K_min = ceil(e^R_IB) (grid64 hard-wall task; {seeds} seeds)")
    print("=" * 60)
    print("  [v3 rework] v2(MNIST) judged 'operationalization failure': 10 classes need only 2.3 nats, and codebooks")
    print("        with K>=32 are already informationally more than enough; the curve is smooth with no knee and no")
    print("        locatable capacity wall. v3 switches to a 64-class synthetic task:")
    print("        H(Y)=log64=4.16 nats; when K<64, logK<H(Y), I(Z;Y)<=logK<H(Y), and")
    print("        Fano forces the student to lose accuracy -- cliff sharpness is guaranteed by the task structure.")
    print("  [R_IB] Fano lower bound of I(Z_t;Y) (given by the teacher error rate). v2 used H(Z_t) (representation entropy),")
    print("        now deprecated: H(Z_t) of a teacher codebook K=256 can exceed log64, and the excess is irrelevant to the task.")
    print("  [criterion] Gap curve: g(K)=acc_teacher - acc_student(K); g_ref=g(K_max);")
    print("         K*(gap) = smallest K with g(K) <= g_ref+tol; if the plateau is not established, refuse to report K*.")

    records, curves, k_list = [], [], None
    for s in range(seeds):
        # Both the data realization and the initialization vary with the seed (the seed only changes randomness, not the task's nominal information log 64)
        torch.manual_seed(4000 + s)
        (x_train, y_train), (x_test, y_test) = make_grid64(n_per_class=n_per_class, seed=s)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=256)

        # 1. teacher (isomorphic to the feature A probe: VQ(K=256) + Linear(3->64))
        teacher = Classifier3D(with_vq=True, n_codes=k_teacher)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(teacher, train_loader, opt)
        acc_t, idx_t = evaluate(teacher, test_loader)
        h_z_t = entropy_of(idx_t[0], k_teacher)  # comparison print: representation entropy vs task information
        r_ib = task_info_fano(1.0 - acc_t, 64, h_y=math.log(64))
        k_min = max(2, math.ceil(math.exp(r_ib)))

        # 2. student grid: anchored on seed 0's K_min; fine levels 48-96 ensure the knee is visible
        #    (a lesson from v2 counter-evidence 3: large holes in the grid degrade the octave test); the top 192/256
        #    are the plateau reference——g_ref must already be on the irreducible gap (plateau detection: see distill_plateau_ok)
        if k_list is None:
            if quick:
                k_list = sorted({8, 32, 64, 256, k_min})
            else:
                k_list = sorted({8, 16, 32, 48, 56, 64, 72, 96, 128, 192, 256,
                                 k_min, max(2, k_min - 2), min(k_teacher, k_min + 2)})
            print(f"  [K grid] anchored on seed 0's K_min={k_min} (densified near the hard wall K=64): {k_list}")

        # 3. Distillation of the student for each K (CE + KL(T=3), budget strictly aligned with the teacher)
        accs = {}
        for K in k_list:
            torch.manual_seed(5000 + 100 * s + K)
            student = Classifier3D(with_vq=True, n_codes=K)
            opt = torch.optim.Adam(student.parameters(), lr=lr)
            for _ in range(epochs):
                train_one_epoch(student, train_loader, opt, distill_from=teacher)
            acc_s, _ = evaluate(student, test_loader)
            accs[K] = acc_s

        plateau = distill_plateau_ok(k_list, accs)
        k_star = distill_gap_k_star(acc_t, k_list, accs) if plateau else None
        records.append((s, k_min, k_star))
        curves.append((s, acc_t, k_min, accs))

        gaps = {K: acc_t - accs[K] for K in k_list}
        g_ref = gaps[k_list[-1]]
        print(f"  seed {s}: teacher acc={acc_t:.3f}  R_IB(Fano)={r_ib:.2f} nats -> K_min={k_min}"
              f"  [reference H(Z_t)={h_z_t:.2f} nats]")
        print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}" for K in k_list))
        print("    deficit R_IB-logK: " + "  ".join(f"{K}:{r_ib - math.log(K):+.2f}" for K in k_list)
              + " (positive=insufficient information)")
        print(f"    plateau established={'yes' if plateau else 'no'}; **K*(gap)={k_star}**"
              f"; [hard wall] K={wall}; g_ref=g({k_list[-1]})={g_ref:.3f}")

    # The per-K x per-seed gap matrix: directly exposes "whether the gap curve is stable" (isomorphic to v2)
    heads = " ".join(f"{'seed' + str(s):>9}" for s, _, _, _ in curves)
    print(f"\n  gap matrix g(K)=teacher-acc  {'K':>4} {heads} {'mean':>9}")
    for K in k_list:
        row = " ".join(f"{acc_t - accs[K]:9.3f}" for _, acc_t, _, accs in curves)
        mean = sum(acc_t - accs[K] for _, acc_t, _, accs in curves) / len(curves)
        print(f"  {'':>22} {K:>4} {row} {mean:9.3f}")
    print("  teacher acc per seed: " + ", ".join(f"seed{s}={a:.3f}" for s, a, _, _ in curves))

    if not quick:
        if seeds > 1:
            print(f"\n  [Feature B multi-seed summary] {seeds} seeds (criterion: gap curve, grid64 hard-wall task)")
        verdict, detail = ratio_verdict(records, "measured K*(gap) / predicted K_min")
        for line in detail:
            print(line)
        print(f"  [Feature B verdict] {verdict}")
    return records, curves


# ---------------------------------------------------------------------------
# Fisher effective-dimension curve (v1 module)
# ---------------------------------------------------------------------------

def fisher_demo(device="cpu", d=32, k=256, epochs=3, quick=False, fast=False):
    """v1 acceptance: report the evolution curve of the effective dimension during training.

    Criterion:
    - Engineering acceptance (hard criterion): 0 < d_eff <= B (number of cached batches), finite, non-zero, reproducible curve;
    - Scientific observation (soft observation): whether d_eff shrinks during training (entering a low-dimensional basin).
    """
    if quick or fast:
        epochs, subset = 2, 0.3
    else:
        subset = None
    print("\n" + "=" * 60)
    print("[Fisher] Effective dimension d_eff (gradient participation ratio) evolution during training")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)
    model = Net(d=d, n_codes=k).to(device)
    fed = FisherEffectiveDim(model, max_batches=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = F.cross_entropy(out, y)
            for m in model.modules():
                if isinstance(m, VQBottleneck):
                    loss = loss + 0.25 * m.commit_loss()
            opt.zero_grad()
            loss.backward()
            fed.collect()
            opt.step()
        d_eff = fed.effective_dim()
        fed.reset()
        acc, _ = evaluate(model, test_loader, device=device)
        curve.append((ep, d_eff, acc))
        print(f"  epoch {ep}: d_eff = {d_eff:6.2f}  acc = {acc:.3f}")

    if not quick and len(curve) >= 2:
        finite = all(de is not None and 0 < de <= 32 for _, de, _ in curve)
        first, last = curve[0][1], curve[-1][1]
        trend = "shrinking (entering a low-dimensional basin)" if last < first else "not shrinking"
        print(f"  [Fisher criterion] curve finite and non-zero: {'yes' if finite else 'no (bug)'}")
        print(f"  [Fisher observation] d_eff first-to-last {first:.2f} -> {last:.2f}: {trend}")
        if not finite:
            print("  [Fisher verdict] FAIL: numerical anomaly (0, out of range, or NaN)")
        else:
            print(f"  [Fisher verdict] PASS (engineering acceptance); trend: {trend}")
    return curve


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Capacity Lens validation experiments")
    ap.add_argument("--quick", action="store_true", help="Quick smoke mode (reduced budget, no verdict output)")
    ap.add_argument("--fast", action="store_true",
                    help="Reduced budget mode (20-30%% subset + fewer epochs), **still outputs verdicts**; "
                         "for quickly getting directional conclusions when long-task environments are unreliable")
    ap.add_argument("--only", choices=["p1", "p2", "p3", "p4", "quant", "qat", "distill",
                                       "distill-mnist", "fisher"],
                    default=None,
                    help="Run only the specified experiment. distill=Feature B v3 (grid64 hard-wall task, current protocol); "
                         "distill-mnist=Feature B v2 (MNIST, archived protocol, superseded by v3); "
                         "qat=quantization-aware training vs PTQ three-arm comparison (a v2 extension of Feature A)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of seeds for Feature A / Feature B (default 3; --quick forces 1). "
                         "Multiple seeds are used to check the stability of a single run's conclusions")
    args = ap.parse_args()
    quick = args.quick
    fast = args.fast and not quick
    seeds = args.seeds

    torch.manual_seed(0)  # reproducibility (the Fisher acceptance criterion requires the same curve for the same seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "quick (smoke, no verdict)" if quick else ("fast (reduced budget, with verdict)" if fast else "full (complete validation)")
    print(f"device: {device}  mode: {mode}")

    if args.only in (None, "p1"):
        verify_p1(device=device, quick=quick, fast=fast)
    if args.only in (None, "p2"):
        verify_p2(device=device, quick=quick, fast=fast)
    if args.only in (None, "p3"):
        verify_p3(device=device, quick=quick, fast=fast)
    if args.only in (None, "p4"):
        verify_p4(device=device, quick=quick, fast=fast)
    if args.only in (None, "quant"):
        verify_quant(quick=quick, fast=fast, seeds=seeds)
    if args.only in (None, "qat"):
        verify_qat(quick=quick, fast=fast, seeds=seeds)
    if args.only in (None, "distill"):
        verify_distill_grid(quick=quick, fast=fast, seeds=seeds)
    if args.only == "distill-mnist":
        verify_distill(device=device, quick=quick, fast=fast, seeds=seeds)
    if args.only in (None, "fisher"):
        fisher_demo(device=device, quick=quick, fast=fast)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
